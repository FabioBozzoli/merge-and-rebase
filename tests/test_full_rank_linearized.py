"""Full-rank linearized (NTK) fine-tuning of a CausalLM: a frozen W0 plus one trainable dW per parameter.

What must hold: the forward is f(x; W0) + J(x; W0) . dW with dW as the tangent, the gradient reaches only dW,
the strategy wires the fp32-master sharded AdamW, and the HF export holds W0 + dW.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from test_causal_lm_linearized_training import IDS_FOR_GRADS, VOCAB, _resume_dir, _run_resumable, _tiny_causal_lm
from torch.func import functional_call

from merge_and_rebase.finetune import train_text
from merge_and_rebase.finetune.forward_mode import apply_training_forward_mode, attach_full_rank_deltas
from merge_and_rebase.finetune.sharded_adamw import ShardedMasterAdamW
from merge_and_rebase.utils.linearization import forward_ad_safe_attention_context


def _bound_fp64_with_random_deltas():
    model = _tiny_causal_lm().double()
    deltas = attach_full_rank_deltas(model)
    torch.manual_seed(3)
    with torch.no_grad():
        for d in deltas.values():
            d.normal_(std=0.02)
    apply_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        device=torch.device("cpu"),
        output_transform=lambda out: out.logits,
        output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
    )
    return model, deltas


def test_full_rank_forward_and_gradients_match_the_oracle() -> None:
    model, deltas = _bound_fp64_with_random_deltas()
    ids = IDS_FOR_GRADS
    base = {k.replace("/", "."): model.get_parameter(k.replace("/", ".")) for k in deltas}

    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    train_text._causal_lm_loss(logits, ids).backward()
    grads = {k: d.grad.clone() for k, d in deltas.items()}
    assert all(not p.requires_grad and p.grad is None for p in base.values())

    # Oracle: d/d eps of f(W0 + eps * dW) at eps = 0 on an unbound copy, differentiated w.r.t. dW directly.
    oracle = _tiny_causal_lm().double()
    oracle_deltas = {k: d.detach().clone().requires_grad_(True) for k, d in deltas.items()}

    def g(eps):
        params = {n: base[n].detach() + eps * oracle_deltas[n.replace(".", "/")] for n in base}
        return functional_call(oracle, params, kwargs={"input_ids": ids, "attention_mask": torch.ones_like(ids)}).logits

    e0 = torch.zeros((), dtype=torch.float64)
    with forward_ad_safe_attention_context(torch.device("cpu")):
        f0, df = torch.func.jvp(g, (e0,), (torch.ones_like(e0),))
    ref = f0 + df
    train_text._causal_lm_loss(ref, ids).backward()

    torch.testing.assert_close(logits, ref, rtol=1e-10, atol=1e-10)
    assert len(grads) == len(base)
    for k, d in oracle_deltas.items():
        torch.testing.assert_close(grads[k], d.grad, rtol=1e-8, atol=1e-12)


def test_full_rank_is_exactly_the_pretrained_model_at_zero_delta() -> None:
    model, deltas = _bound_fp64_with_random_deltas()
    with torch.no_grad():
        for d in deltas.values():
            d.zero_()
    ids = IDS_FOR_GRADS
    torch.testing.assert_close(
        model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits,
        _tiny_causal_lm().double()(input_ids=ids).logits,
        rtol=1e-12,
        atol=1e-12,
    )


def test_causal_full_strategy_needs_linearized_and_uses_the_sharded_master_adamw() -> None:
    common = dict(
        strategy="full",
        optimizer_name="adamw",
        lr=1e-3,
        weight_decay=0.0,
        warmup_length=0,
        steps=4,
        device=torch.device("cpu"),
        model_kind="causal_lm",
    )
    with pytest.raises(ValueError, match="peft_lora"):
        train_text._configure_text_strategy(model=_tiny_causal_lm(), strategy_cfg={}, **common)

    model, opt, _, info, _ = train_text._configure_text_strategy(
        model=_tiny_causal_lm(), strategy_cfg={"forward_mode": "linearized_ntk"}, **common
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert isinstance(opt, ShardedMasterAdamW)
    assert trainable and all(n.startswith("ntk_delta.") for n in trainable)
    assert info["trainable_params"] == sum(p.numel() for n, p in model.named_parameters() if not n.startswith("ntk_delta."))


def test_full_rank_run_exports_w0_plus_the_best_delta(tmp_path, monkeypatch) -> None:
    from transformers import AutoModelForCausalLM

    summary, _ = _run_resumable(
        tmp_path,
        monkeypatch,
        strategy="full",
        strategy_cfg={"forward_mode": "linearized_ntk"},
        epochs=1,
        lr=1e-2,
    )
    hf_dir = tmp_path / "dummy" / "resume_causal" / "full__linearized_ntk_hf"
    reloaded = AutoModelForCausalLM.from_pretrained(hf_dir)
    assert not any("ntk_delta" in n for n in reloaded.state_dict())

    stash = torch.load(tmp_path / "dummy" / "resume_causal" / "full__linearized_ntk_best_trainable.pt", weights_only=False)
    assert stash["best_step"] == summary["best_step"]
    base = _tiny_causal_lm()
    moved = 0
    for key, d in stash["trainable_state"].items():
        name = key[len("ntk_delta.") :].replace("/", ".")
        torch.testing.assert_close(reloaded.get_parameter(name), base.get_parameter(name) + d, rtol=0, atol=1e-6)
        moved += int(d.abs().max() > 0)
    assert moved > 0

    meta = json.loads((hf_dir / "merge_and_rebase_meta.json").read_text())
    assert meta["forward_mode"] == "linearized_ntk"
    assert meta["full_rank_roundtrip_rel_err"]["max"] < 1e-3  # fp32 model: the round trip is near exact
    assert int(reloaded.config.vocab_size) == VOCAB


def _tiny_llama(tie: bool):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA, as in Llama-3.2
        pad_token_id=0,
        eos_token_id=1,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    return LlamaForCausalLM(config).double().eval()


@pytest.mark.parametrize("tie", [True, False])
def test_layerwise_recompute_matches_the_whole_model_jvp(tie) -> None:
    """Same logits and dW gradients, on a right-padded batch, with and without tied embeddings."""
    ids = IDS_FOR_GRADS.clone()
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0  # right padding, as the token-budget micro-batches have
    labels = ids.masked_fill(mask == 0, -100)
    results = []
    for recompute in (False, True):
        model = _tiny_llama(tie)
        deltas = attach_full_rank_deltas(model)
        torch.manual_seed(3)
        with torch.no_grad():
            for d in deltas.values():
                d.normal_(std=0.02)
        apply_training_forward_mode(
            model=model,
            forward_mode="linearized_ntk",
            device=torch.device("cpu"),
            output_transform=lambda out: out.logits,
            output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
            recompute=recompute,
        )
        logits = model(input_ids=ids, attention_mask=mask).logits
        train_text._causal_lm_loss(logits, labels).backward()
        results.append((logits.detach(), {k: d.grad.clone() for k, d in deltas.items()}))
    (y, g), (y_rc, g_rc) = results
    keep = mask.bool()  # pad positions' logits are never supervised
    torch.testing.assert_close(y_rc[keep], y[keep], rtol=1e-10, atol=1e-10)
    assert set(g_rc) == set(g)
    for k in g:
        torch.testing.assert_close(g_rc[k], g[k], rtol=1e-9, atol=1e-12)


def test_epoch_deltas_and_a_warm_start_from_one(tmp_path, monkeypatch) -> None:
    """fp32 dW per epoch; a new run started from one (lr 0: nothing moves) exports W0 + exactly that dW."""
    from transformers import AutoModelForCausalLM

    run = dict(strategy="full", strategy_cfg={"forward_mode": "linearized_ntk"}, epochs=2, save_delta_every_epoch=True)
    _run_resumable(tmp_path / "a", monkeypatch, lr=1e-2, **run)
    task_dir = tmp_path / "a" / "dummy" / "resume_causal"
    epoch2 = torch.load(task_dir / "full__linearized_ntk_epoch2_delta.pt", weights_only=True)
    assert (task_dir / "full__linearized_ntk_epoch1_delta.pt").exists() and epoch2["epoch"] == 2
    assert all(d.dtype == torch.float32 for d in epoch2["trainable_state"].values())

    summary, _ = _run_resumable(
        tmp_path / "b", monkeypatch, lr=0.0, init_from=str(task_dir / "full__linearized_ntk_epoch2_delta.pt"), **run
    )
    assert summary["trainable"]["init_from"].endswith("epoch2_delta.pt")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "b" / "dummy" / "resume_causal" / "full__linearized_ntk_hf")
    base = _tiny_causal_lm()
    for key, d in epoch2["trainable_state"].items():
        name = key[len("ntk_delta.") :].replace("/", ".")
        torch.testing.assert_close(reloaded.get_parameter(name), base.get_parameter(name) + d, rtol=0, atol=1e-6)


def test_read_full_rank_delta_takes_hf_names_and_skips_tied_aliases(tmp_path) -> None:
    model = _tiny_llama(tie=True)
    params = list(attach_full_rank_deltas(model).values())
    torch.manual_seed(1)
    plain = {n: torch.randn_like(p, dtype=torch.float32) for n, p in model.named_parameters() if not n.startswith("ntk_delta.")}
    plain["lm_head.weight"] = plain["model.embed_tokens.weight"]  # what a tied model's state_dict diff holds
    torch.save(plain, tmp_path / "plain.pt")
    got = train_text._read_full_rank_delta(tmp_path / "plain.pt", model, params)
    names = [k.replace("/", ".") for k in model.ntk_delta]
    assert all(torch.equal(g, plain[n]) for g, n in zip(got, names, strict=True))

    torch.save({**plain, "model.not_a_param": torch.zeros(1)}, tmp_path / "extra.pt")
    with pytest.raises(ValueError, match="matches no full-rank dW"):
        train_text._read_full_rank_delta(tmp_path / "extra.pt", model, params)
    torch.save({k: v for k, v in plain.items() if k != "model.norm.weight"}, tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="no dW for 1"):
        train_text._read_full_rank_delta(tmp_path / "missing.pt", model, params)


def test_full_rank_capped_then_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch) -> None:
    run = dict(strategy="full", strategy_cfg={"forward_mode": "linearized_ntk"}, lr=1e-2)
    _run_resumable(tmp_path / "straight", monkeypatch, max_steps=7, **run)
    _run_resumable(tmp_path / "first", monkeypatch, max_steps=5, **run)  # 5 = the epoch boundary
    summary, _ = _run_resumable(
        tmp_path / "second",
        monkeypatch,
        max_steps=7,
        resume_from=str(_resume_dir(tmp_path / "first") / "step_0000005.pt"),
        **run,
    )
    assert summary["global_update_step"] == 7
    a = torch.load(_resume_dir(tmp_path / "straight") / "step_0000007.pt", weights_only=False)
    b = torch.load(_resume_dir(tmp_path / "second") / "step_0000007.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    for key in ("master", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(b["optimizer"][key], a["optimizer"][key], rtol=1e-5, atol=1e-7)
    assert b["optimizer"]["step"] == a["optimizer"]["step"] == 7
