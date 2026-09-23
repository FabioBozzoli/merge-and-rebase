"""Linearized (NTK) LoRA fine-tuning of a CausalLM.

The four things that can silently go wrong and produce a model that trains
without error but is not the model anyone wanted:

1. the linearized forward is not actually f(x; W0) + J(x; W0) . dW,
2. the gradient does not reach the LoRA factors (or leaks into base weights),
3. the -100 response mask does not reach the NTK residual, so prompt tokens are
   supervised too,
4. save_format='hf' writes something AutoModelForCausalLM cannot load.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.func import functional_call

from merge_and_rebase.data.text_loaders import (
    CausalExample,
    CausalTaskData,
    build_causal_tokenized_loader,
    split_causal_task_data,
)
from merge_and_rebase.finetune import train_text
from merge_and_rebase.finetune.forward_mode import apply_training_forward_mode
from merge_and_rebase.models.text_lm import TextBuildConfig
from merge_and_rebase.utils.linearization import forward_ad_safe_attention_context

VOCAB = 64
TARGET_MODULES = ["q_proj", "v_proj"]


def _tiny_causal_lm(hidden_size: int = 16, num_layers: int = 2, seed: int = 0):
    """Mirrors tests.test_text_rebase._tiny_qwen, for the causal head."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def _linearized_peft_causal_lm(*, r: int = 4, lora_alpha: int = 8, seed: int = 0):
    model = get_peft_model(
        _tiny_causal_lm(seed=seed),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=list(TARGET_MODULES),
            bias="none",
        ),
    )
    apply_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        device=torch.device("cpu"),
        output_transform=lambda out: out.logits,
        output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
    )
    # Warm the adapter after binding, as a resumed run does; the expansion point
    # is the frozen base weights either way.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)
    return model


def _tiny_tokenizer():
    """Word-level stand-in: enough for the -100 boundary, no HF download."""

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def _encode(self, text, add_special_tokens):
            # crc32, not hash(): str hashing is salted per process, so a spawned
            # rank would tokenize the same text differently.
            ids = [(zlib.crc32(w.encode()) % (VOCAB - 2)) + 2 for w in str(text).split()]
            return [2] + ids if add_special_tokens else ids

        def __call__(self, text, add_special_tokens=True, **kwargs):
            if isinstance(text, (list, tuple)):
                return {"input_ids": [self._encode(t, add_special_tokens) for t in text]}
            return {"input_ids": self._encode(text, add_special_tokens)}

    return _Tok()


def test_linearized_forward_is_first_order_expansion_around_pretrained_weights() -> None:
    """f_lin(x) == f(x; W0) + d/de f(x; W0 + e.dW)|_{e=0}, dW = s.B.A.

    Checked in two halves against an *independent* plain forward of the
    unwrapped pretrained model, so a wrong expansion point and a wrong tangent
    fail separately:
      - with lora_B == 0 the linearized forward must be exactly f(x; W0);
      - the remainder must match a central difference along dW.
    """
    model = _linearized_peft_causal_lm()
    model.double()
    plain = _tiny_causal_lm().double()  # same seed -> same W0, no adapters

    names = model._ntk_linearized_names
    assert all("lora_" not in name for name in names)
    assert any(name.endswith("q_proj.base_layer.weight") for name in names)

    # dW per host, from the adapter factors, keyed by the plain model's names.
    tangent = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "base_layer") and "default" in getattr(mod, "lora_A", {}):
            plain_name = name.removeprefix("base_model.model.") + ".weight"
            a, b = mod.lora_A["default"].weight.detach(), mod.lora_B["default"].weight.detach()
            tangent[plain_name] = float(mod.scaling["default"]) * b @ a
    w0 = {n: p.detach() for n, p in plain.named_parameters()}
    assert tangent and max(float(t.abs().max()) for t in tangent.values()) > 0.0

    input_ids = torch.randint(2, VOCAB, (2, 6))
    attention_mask = torch.ones_like(input_ids)

    def _plain(eps: float) -> torch.Tensor:
        params = {n: w0[n] + eps * t for n, t in tangent.items()}
        out = functional_call(
            plain, params, args=(), kwargs={"input_ids": input_ids, "attention_mask": attention_mask}, strict=False
        )
        return out.logits.detach()

    with torch.no_grad():
        actual = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Expansion point: zero update must reproduce the pretrained forward.
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
        at_theta0 = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(at_theta0, _plain(0.0), atol=1e-10, rtol=0.0)

    # Tangent: J . dW against a central difference. The floor here is the
    # forward's own ~1e-8 numerical noise amplified by 1/h, not the jvp -- the
    # error is O(h^2) down to h ~ 3e-3 and grows again below it.
    jvp_term = actual - at_theta0
    fd_term = (_plain(3e-3) - _plain(-3e-3)) / (2 * 3e-3)
    rel_err = float((jvp_term - fd_term).abs().max() / jvp_term.abs().max())
    assert rel_err < 1e-3, rel_err


def test_gradient_reaches_lora_factors_and_not_base_weights() -> None:
    model = _linearized_peft_causal_lm()
    input_ids = torch.randint(2, VOCAB, (2, 6))
    labels = input_ids.clone()

    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    train_text._causal_lm_loss(logits, labels).backward()

    named = dict(model.named_parameters())
    lora_a = [n for n in named if "lora_A" in n]
    lora_b = [n for n in named if "lora_B" in n]
    assert lora_a and lora_b

    for name in lora_a + lora_b:
        assert named[name].grad is not None, name
        assert float(named[name].grad.abs().sum()) > 0.0, name

    for name, param in named.items():
        if "lora_" in name:
            continue
        assert not param.requires_grad, name
        assert param.grad is None, name


def test_prompt_masked_rows_contribute_no_gradient() -> None:
    """A row whose labels are all -100 must not move the LoRA factors.

    This is the response-only objective: if the mask did not reach the NTK
    residual r = softmax(z) - onehot(y), the fully-masked row would still push.
    """
    model = _linearized_peft_causal_lm()
    input_ids = torch.randint(2, VOCAB, (2, 6))
    attention_mask = torch.ones_like(input_ids)

    supervised = input_ids.clone()
    supervised[1, :] = -100  # row 1 is prompt-only

    def _grads(labels: torch.Tensor) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        train_text._causal_lm_loss(logits, labels).backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    masked = _grads(supervised)

    # Same objective with row 1 removed entirely: identical gradients.
    row0_only = input_ids[:1]
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=row0_only, attention_mask=torch.ones_like(row0_only)).logits
    train_text._causal_lm_loss(logits, row0_only.clone()).backward()
    alone = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    assert set(masked) == set(alone)
    for name in masked:
        assert torch.allclose(masked[name], alone[name], atol=1e-6, rtol=1e-4), name


def test_causal_loader_masks_prompt_and_supervises_response() -> None:
    data = CausalTaskData(
        task="t",
        examples=[CausalExample(prompt="a b c", response="d e") for _ in range(10)],
        meta={},
    )
    tokenized = build_causal_tokenized_loader(
        task_data=data, tokenizer=_tiny_tokenizer(), batch_size=2, max_length=32
    )
    batch = next(iter(tokenized.loader))

    labels = batch["labels"][0]
    ids = batch["input_ids"][0]
    supervised = labels != -100
    assert int(supervised.sum()) == 3  # "d", "e", eos
    assert bool(supervised[-3:].all()) and not bool(supervised[:-3].any())
    assert torch.equal(ids[supervised], labels[supervised])

    train_data, val_data = split_causal_task_data(data, val_fraction=0.2)
    assert len(train_data.examples) == 8
    assert len(val_data.examples) == 2


def test_hf_export_round_trips_through_automodel(tmp_path, monkeypatch) -> None:
    from transformers import AutoModelForCausalLM

    ids = torch.randint(2, VOCAB, (2, 6))
    labels = ids.clone()
    labels[:, :3] = -100
    batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}]

    monkeypatch.setattr(
        train_text.TextLM,
        "build",
        staticmethod(lambda build_cfg: SimpleNamespace(model=_tiny_causal_lm(), tokenizer=object())),
    )
    monkeypatch.setattr(
        train_text,
        "_build_task_loaders",
        lambda **kwargs: (
            SimpleNamespace(loader=batches),
            SimpleNamespace(loader=batches),
            SimpleNamespace(loader=batches),
            {"model_kind": "causal_lm", "train": {}, "validation": {}},
        ),
    )

    summary, head_payload = train_text.train_task(
        task="dummy_causal",
        build_cfg=TextBuildConfig(
            model_name_or_path="dummy",
            model_arch="auto",
            device="cpu",
            dtype="fp32",
            model_kind="causal_lm",
            num_labels=1,
            trust_remote_code=False,
            use_fast_tokenizer=True,
        ),
        strategy="peft_lora",
        strategy_cfg={
            "forward_mode": "linearized_ntk",
            "peft": {"target_modules": list(TARGET_MODULES), "r": 4, "lora_alpha": 8},
        },
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        warmup_length=0,
        optimizer_name="sgd",
        clip_grad_norm=0.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        max_length=6,
        head_num_labels=0,
        early_stopping=False,
        early_stopping_patience=3,
        seed=0,
        deterministic=False,
        device="cpu",
        out_dir=tmp_path,
        save_format="hf",
        task_cfg={},
    )

    assert head_payload == {}
    assert summary["forward_mode"] == "linearized_ntk"
    assert "val_loss" in summary["metrics"] and "val_ppl" in summary["metrics"]

    hf_dir = tmp_path / "dummy" / "dummy_causal" / "peft_lora__linearized_ntk_hf"
    reloaded = AutoModelForCausalLM.from_pretrained(hf_dir)
    assert int(reloaded.config.vocab_size) == VOCAB
    assert not any("lora" in n for n in reloaded.state_dict())

    meta = json.loads((hf_dir / "merge_and_rebase_meta.json").read_text())
    assert meta["forward_mode"] == "linearized_ntk"
    assert "f(x; W0) + J(x; W0) . dW" in meta["linearized_warning"]


def test_binding_order_does_not_move_the_expansion_point() -> None:
    """The primal is the frozen base weight itself, so binding and THEN loading a
    trained adapter (the resume order) computes the same function as binding a
    model whose adapter is already warm -- both expand around W0."""
    import copy

    cold = get_peft_model(
        _tiny_causal_lm(),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=list(TARGET_MODULES),
            bias="none",
        ),
    )
    warm = copy.deepcopy(cold)
    torch.manual_seed(3)
    b_vals = {n: 0.1 * torch.randn_like(p) for n, p in cold.named_parameters() if "lora_B" in n}

    def _load_b(model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in b_vals:
                    p.copy_(b_vals[n])

    _load_b(warm)
    for model in (cold, warm):
        apply_training_forward_mode(
            model=model,
            forward_mode="linearized_ntk",
            device=torch.device("cpu"),
            output_transform=lambda out: out.logits,
            output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
        )
    _load_b(cold)

    ids = torch.randint(2, VOCAB, (2, 8))
    torch.testing.assert_close(cold(input_ids=ids).logits, warm(input_ids=ids).logits)


def test_step_level_eval_lets_early_stopping_fire_inside_one_epoch(tmp_path, monkeypatch) -> None:
    """A single pass over 591K rows evaluates once at epoch end, which leaves
    early stopping nothing to act on. train.eval_every_n_steps is what makes it
    real for a one-epoch run."""
    ids = torch.randint(2, VOCAB, (2, 6))
    labels = ids.clone()
    labels[:, :3] = -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    batches = [batch] * 6

    losses = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    calls: list[float] = []

    def _fake_eval(model, loader, device, dist_info=None):
        loss = next(losses)
        calls.append(loss)
        return {"val_loss": loss, "val_ppl": float(torch.exp(torch.tensor(loss))), "val_token_acc": 0.0}

    monkeypatch.setattr(train_text, "_eval_causal", _fake_eval)
    monkeypatch.setattr(
        train_text.TextLM,
        "build",
        staticmethod(lambda build_cfg: SimpleNamespace(model=_tiny_causal_lm(), tokenizer=object())),
    )
    monkeypatch.setattr(
        train_text,
        "_build_task_loaders",
        lambda **kwargs: (
            SimpleNamespace(loader=batches),
            SimpleNamespace(loader=batches),
            SimpleNamespace(loader=batches),
            {"model_kind": "causal_lm"},
        ),
    )

    summary, _ = train_text.train_task(
        task="stop_causal",
        build_cfg=TextBuildConfig(
            model_name_or_path="dummy",
            model_arch="auto",
            device="cpu",
            dtype="fp32",
            model_kind="causal_lm",
            num_labels=1,
            trust_remote_code=False,
            use_fast_tokenizer=True,
        ),
        strategy="peft_lora",
        strategy_cfg={
            "forward_mode": "linearized_ntk",
            "peft": {"target_modules": list(TARGET_MODULES), "r": 4, "lora_alpha": 8},
        },
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        warmup_length=0,
        optimizer_name="sgd",
        clip_grad_norm=0.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        max_length=6,
        head_num_labels=0,
        early_stopping=True,
        early_stopping_patience=1,
        eval_every_n_steps=1,
        seed=0,
        deterministic=False,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        task_cfg={},
    )

    # Stopped after the first non-improving eval, not after all 6 batches.
    assert calls == [1.0, 2.0]
    assert summary["metrics"]["val_loss"] == 1.0  # best checkpoint is the first
    assert summary["hparams"]["eval_every_n_steps"] == 1


# --- step-level checkpoint / resume -----------------------------------------
#
# An LR sweep caps each trial at train.max_steps, and the winner resumes from
# its step-N checkpoint. The resumed run must be the *same* run: same LoRA
# factors, optimizer moments, LR curve (warmup included) and data order.


def _resume_loaders(n_rows: int = 20, batch_size: int = 2, seed: int = 0, rank: int = 0, world_size: int = 1):
    data = CausalTaskData(
        task="resume_causal",
        examples=[
            CausalExample(prompt=f"q{i} alpha beta", response=f"r{i} gamma delta eps") for i in range(n_rows)
        ],
        meta={},
    )
    tok = _tiny_tokenizer()
    train = build_causal_tokenized_loader(
        task_data=data,
        tokenizer=tok,
        batch_size=batch_size,
        max_length=16,
        shuffle=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    val = build_causal_tokenized_loader(task_data=data, tokenizer=tok, batch_size=batch_size, max_length=16)
    return train, val


def _run_resumable(tmp_path, monkeypatch, **overrides):
    info = overrides.get("dist_info")
    train, val = _resume_loaders(
        batch_size=overrides.get("batch_size", 2),
        rank=getattr(info, "rank", 0),
        world_size=getattr(info, "world_size", 1),
    )
    monkeypatch.setattr(
        train_text.TextLM,
        "build",
        staticmethod(lambda build_cfg: SimpleNamespace(model=_tiny_causal_lm(), tokenizer=object())),
    )
    monkeypatch.setattr(
        train_text,
        "_build_task_loaders",
        lambda **kwargs: (train, val, val, {"model_kind": "causal_lm"}),
    )
    kwargs = dict(
        task="resume_causal",
        build_cfg=TextBuildConfig(
            model_name_or_path="dummy",
            model_arch="auto",
            device="cpu",
            dtype="fp32",
            model_kind="causal_lm",
            num_labels=1,
            trust_remote_code=False,
            use_fast_tokenizer=True,
        ),
        strategy="peft_lora",
        strategy_cfg={
            "forward_mode": "linearized_ntk",
            "peft": {"target_modules": list(TARGET_MODULES), "r": 4, "lora_alpha": 8},
        },
        epochs=2,
        lr=0.01,
        weight_decay=0.0,
        warmup_length=4,
        scheduler_name="cosine",
        optimizer_name="adamw",
        clip_grad_norm=1.0,
        accumulate_grad_batches=2,  # 10 batches/epoch -> 5 steps/epoch, 10 total
        batch_size=2,
        num_workers=0,
        max_length=16,
        head_num_labels=0,
        early_stopping=False,
        early_stopping_patience=3,
        seed=0,
        deterministic=False,
        device="cpu",
        out_dir=tmp_path,
        save_format="hf",
        task_cfg={},
    )
    kwargs.update(overrides)
    return train_text.train_task(**kwargs)


def _resume_dir(root):
    return root / "dummy" / "resume_causal" / "resume"


def test_resumable_sampler_offset_replays_the_tail_of_the_same_permutation() -> None:
    sampler = train_text.ResumableRandomSampler(11, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, start_index=4)
    assert list(sampler) == full[4:]
    assert len(sampler) == 7
    sampler.set_epoch(3)
    assert list(sampler) != full  # a new epoch reshuffles


def test_max_steps_caps_evaluates_and_writes_a_resume_checkpoint(tmp_path, monkeypatch) -> None:
    summary, _ = _run_resumable(tmp_path, monkeypatch, max_steps=3)

    assert summary["global_update_step"] == 3
    assert summary["stop_reason"] == "max_steps"
    assert summary["best_step"] == 3  # the cap step was evaluated
    assert "val_loss" in summary["metrics"]
    # The cap must not shrink the LR horizon, or a continuation starts at lr=0.
    assert summary["total_steps"] == 10

    rdir = _resume_dir(tmp_path)
    ckpt = torch.load(rdir / "step_0000003.pt", weights_only=False)
    assert (rdir / "resume_last.pt").resolve() == (rdir / "step_0000003.pt").resolve()
    assert ckpt["micro_batches_consumed_in_epoch"] == 6 and ckpt["epoch"] == 1
    assert ckpt["fingerprint"]["total_steps"] == 10
    assert (tmp_path / "dummy" / "resume_causal" / "peft_lora__linearized_ntk_best_ep.pt").exists()
    assert (tmp_path / "dummy" / "resume_causal" / "peft_lora__linearized_ntk_best_trainable.pt").exists()


@pytest.mark.parametrize("split_step", [2, 5])  # 2: inside warmup; 5: exactly at the epoch boundary
def test_capped_then_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch, split_step) -> None:
    from merge_and_rebase.finetune.schedulers import cosine_lr

    straight_dir, first_dir, second_dir = tmp_path / "straight", tmp_path / "first", tmp_path / "second"
    _run_resumable(straight_dir, monkeypatch, max_steps=7)
    _run_resumable(first_dir, monkeypatch, max_steps=split_step)
    summary, _ = _run_resumable(
        second_dir,
        monkeypatch,
        max_steps=7,
        resume_from=str(_resume_dir(first_dir) / f"step_{split_step:07d}.pt"),
    )
    assert summary["global_update_step"] == 7

    a = torch.load(_resume_dir(straight_dir) / "step_0000007.pt", weights_only=False)
    b = torch.load(_resume_dir(second_dir) / "step_0000007.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    assert a["micro_batches_consumed_in_epoch"] == b["micro_batches_consumed_in_epoch"] == 4
    assert set(a["trainable_state"]) == set(b["trainable_state"])
    for name, value in a["trainable_state"].items():
        torch.testing.assert_close(b["trainable_state"][name], value, rtol=1e-5, atol=1e-6)
    for pid, state in a["optimizer"]["state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(b["optimizer"]["state"][pid][key], state[key], rtol=1e-5, atol=1e-7)

    # LR continuity: the last step (index 6) used the schedule's value, not a
    # restarted warmup.
    probe = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    cosine_lr(probe, 0.01, warmup_length=4, steps=10)(6)
    expected_lr = probe.param_groups[0]["lr"]
    assert b["optimizer"]["param_groups"][0]["lr"] == pytest.approx(expected_lr)
    assert a["optimizer"]["param_groups"][0]["lr"] == pytest.approx(expected_lr)


def test_resume_rejects_a_checkpoint_from_a_different_run(tmp_path, monkeypatch) -> None:
    _run_resumable(tmp_path / "first", monkeypatch, max_steps=2)
    ckpt = str(_resume_dir(tmp_path / "first") / "step_0000002.pt")

    with pytest.raises(ValueError, match="rows_per_step"):
        _run_resumable(tmp_path / "b", monkeypatch, resume_from=ckpt, accumulate_grad_batches=1)
    with pytest.raises(ValueError, match="lr"):
        _run_resumable(tmp_path / "c", monkeypatch, resume_from=ckpt, lr=0.02)

    summary, _ = _run_resumable(
        tmp_path / "d", monkeypatch, resume_from=ckpt, lr=0.02, resume_allow_lr_change=True, max_steps=3
    )
    assert summary["global_update_step"] == 3


def test_bs1_checkpoint_resumes_at_bs2_with_the_same_rows_per_step(tmp_path, monkeypatch) -> None:
    """bs=1 x 4 and bs=2 x 2 consume the same 4 rows per optimizer step, and the
    gradient is an exact token-weighted sum over them, so a bs=1 run continues at
    bs=2 onto the weights an uninterrupted bs=2 run reaches (padding is masked)."""
    _run_resumable(tmp_path / "straight", monkeypatch, max_steps=4, batch_size=2, accumulate_grad_batches=2)
    _run_resumable(tmp_path / "first", monkeypatch, max_steps=2, batch_size=1, accumulate_grad_batches=4)
    summary, _ = _run_resumable(
        tmp_path / "second",
        monkeypatch,
        max_steps=4,
        batch_size=2,
        accumulate_grad_batches=2,
        resume_from=str(_resume_dir(tmp_path / "first") / "step_0000002.pt"),
    )
    assert summary["global_update_step"] == 4

    a = torch.load(_resume_dir(tmp_path / "straight") / "step_0000004.pt", weights_only=False)
    b = torch.load(_resume_dir(tmp_path / "second") / "step_0000004.pt", weights_only=False)
    assert a["rows_consumed_in_epoch"] == b["rows_consumed_in_epoch"]
    for name, value in a["trainable_state"].items():
        torch.testing.assert_close(b["trainable_state"][name], value, rtol=1e-4, atol=1e-6)


def test_sigusr1_checkpoints_at_the_next_step_and_exits(tmp_path, monkeypatch) -> None:
    import os
    import signal

    real_loss = train_text._causal_lm_loss_sum
    calls = {"n": 0}

    def _loss_then_signal(logits, labels):
        calls["n"] += 1
        if calls["n"] == 3:  # mid-window of step 2 (accumulate=2)
            os.kill(os.getpid(), signal.SIGUSR1)
        return real_loss(logits, labels)

    monkeypatch.setattr(train_text, "_causal_lm_loss_sum", _loss_then_signal)
    summary, _ = _run_resumable(tmp_path, monkeypatch)

    assert summary["status"] == "preempted"
    assert summary["global_update_step"] == 2  # finished the window it was in
    assert (_resume_dir(tmp_path) / "step_0000002.pt").exists()
    assert not (tmp_path / "dummy" / "resume_causal" / "peft_lora__linearized_ntk_hf").exists()


# --- the linearization against an independent oracle --------------------------
#
# Oracle: g(eps) = f_peft(x; W0, A, eps * B) through PEFT's OWN forward. Then
# g(0) + g'(0) = f(x; W0) + J_W(x; W0) . (s B A) exactly, with the LoRA scale
# applied by PEFT and no weight-space tangent anywhere -- a computation that
# shares nothing with the implementation under test.

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _warm_peft_fp64(**lora_kwargs):
    model = get_peft_model(
        _tiny_causal_lm().double(),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=ALL_TARGETS,
            bias="none",
            **lora_kwargs,
        ),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.3)
            elif "lora_A" in name:
                p.add_(0.1 * torch.randn_like(p))
    return model


def _bind(model):
    apply_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        device=torch.device("cpu"),
        output_transform=lambda out: out.logits,
        output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
    )
    return model


def _oracle_logits(model, ids, extra=None):
    """extra: {param name: (p0, p)} for non-LoRA params, tangent p - p0."""
    extra = extra or {}
    b = {n: p for n, p in model.named_parameters() if "lora_B" in n}

    def g(eps):
        params = {n: eps * p for n, p in b.items()}
        params.update({n: p0 + eps * (p - p0) for n, (p0, p) in extra.items()})
        return functional_call(
            model, params, args=(), kwargs={"input_ids": ids, "attention_mask": torch.ones_like(ids)}, strict=False
        ).logits

    e0 = torch.zeros((), dtype=torch.float64)
    with forward_ad_safe_attention_context(torch.device("cpu")):  # CPU flash SDPA has no forward AD
        f0, df = torch.func.jvp(g, (e0,), (torch.ones_like(e0),))
    return f0 + df


def _loss_and_grads(model, logits_fn):
    logits = logits_fn()
    model.zero_grad(set_to_none=True)
    train_text._causal_lm_loss(logits, IDS_FOR_GRADS).backward()
    return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


IDS_FOR_GRADS = torch.randint(2, VOCAB, (2, 10), generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("use_rslora", [False, True])
def test_linearization_matches_the_oracle_forward_and_gradients(use_rslora) -> None:
    import copy

    model = _warm_peft_fp64(use_rslora=use_rslora)
    oracle = copy.deepcopy(model)
    _bind(model)
    ids = IDS_FOR_GRADS

    y, g = _loss_and_grads(model, lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits)
    y_ref, g_ref = _loss_and_grads(oracle, lambda: _oracle_logits(oracle, ids))

    torch.testing.assert_close(y, y_ref, rtol=1e-10, atol=1e-10)
    assert set(g) == set(g_ref) and len(g) == 28  # A and B of 7 modules x 2 layers
    for name, grad in g_ref.items():
        torch.testing.assert_close(g[name], grad, rtol=1e-8, atol=1e-10)


def test_linearization_is_linear_in_the_update_and_exact_at_zero() -> None:
    """f_lin(2B) - 2 f_lin(B) + f_lin(0) == 0 while the plain PEFT forward curves,
    and at B == 0 it is exactly the pretrained model."""
    model = _bind(_warm_peft_fp64())
    b_values = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    ids = torch.randint(2, VOCAB, (2, 10))

    def _at(scale):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in b_values:
                    p.copy_(scale * b_values[n])
            return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

    y0, y1, y2 = _at(0.0), _at(1.0), _at(2.0)
    assert (y2 - 2 * y1 + y0).abs().max().item() < 1e-10
    torch.testing.assert_close(y0, _tiny_causal_lm().double()(input_ids=ids).logits, rtol=1e-10, atol=1e-10)


def test_non_lora_trainable_params_enter_as_p_minus_p0() -> None:
    """A trainable non-LoRA param (here lm_head, like a modules_to_save head) is
    linearized around its value at bind time, alongside the LoRA update."""
    import copy

    model = _warm_peft_fp64()
    head = "base_model.model.lm_head.weight"
    dict(model.named_parameters())[head].requires_grad = True
    oracle = copy.deepcopy(model)
    head0 = dict(oracle.named_parameters())[head].detach().clone()
    _bind(model)

    torch.manual_seed(2)
    with torch.no_grad():
        dict(model.named_parameters())[head].add_(0.05 * torch.randn_like(head0))
    head_now = dict(model.named_parameters())[head].detach().clone()

    ids = torch.randint(2, VOCAB, (2, 10))
    y = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    y_ref = _oracle_logits(oracle, ids, extra={head: (head0, head_now)})
    torch.testing.assert_close(y, y_ref, rtol=1e-10, atol=1e-10)


# --- data parallelism --------------------------------------------------------
#
# N ranks each cover 1/N of the accumulation window, so the effective batch and
# the LR schedule are independent of N. After the per-step all-reduce every rank
# must hold the same weights as a single process would.


def _dp_worker(rank: int, world_size: int, tmp_dir: str) -> None:
    import os

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29517", WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK="0"
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from merge_and_rebase.utils.distributed import DistInfo

        # Rank 0 writes the checkpoint the parent reads; tensors are not sent back
        # over a queue, whose shared-memory handles do not survive the process exit.
        _run_resumable(
            Path(tmp_dir) / f"w{world_size}",
            _DummyMonkeypatch(),
            max_steps=3,
            dist_info=DistInfo(rank=rank, world_size=world_size),
        )
    finally:
        dist.destroy_process_group()


class _DummyMonkeypatch:
    """setattr-only stand-in for pytest's monkeypatch, usable in a child process."""

    def setattr(self, target, name, value):  # noqa: D102 - mirrors monkeypatch.setattr
        setattr(target, name, value)


def test_two_rank_data_parallel_matches_one_process(tmp_path) -> None:
    import torch.multiprocessing as mp

    single, _ = _run_resumable(tmp_path / "w1", _DummyMonkeypatch(), max_steps=3)
    assert single["hparams"]["world_size"] == 1
    single_state = torch.load(
        tmp_path / "w1" / "dummy" / "resume_causal" / "resume" / "step_0000003.pt", weights_only=False
    )["trainable_state"]

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_dp_worker, args=(r, 2, str(tmp_path))) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    dp_state = torch.load(
        tmp_path / "w2" / "dummy" / "resume_causal" / "resume" / "step_0000003.pt", weights_only=False
    )["trainable_state"]
    assert set(dp_state) == set(single_state)
    for name, value in single_state.items():
        torch.testing.assert_close(dp_state[name], value, rtol=1e-5, atol=1e-7)


def test_sampler_shards_disjointly_and_evenly_across_ranks() -> None:
    shards = []
    for rank in range(3):
        s = train_text.ResumableRandomSampler(11, seed=5, rank=rank, world_size=3)
        s.set_epoch(1)
        shards.append(list(s))
    assert [len(x) for x in shards] == [3, 3, 3]  # the ragged tail is dropped, so ranks stay in lockstep
    flat = [i for shard in shards for i in shard]
    assert len(set(flat)) == 9
    ref = train_text.ResumableRandomSampler(11, seed=5)
    ref.set_epoch(1)
    assert set(flat) <= set(ref.permutation())
    s0 = train_text.ResumableRandomSampler(11, seed=5, rank=0, world_size=3)
    s0.set_epoch(1, start_index=1)
    assert list(s0) == shards[0][1:]


def _dp_resume_worker(rank: int, world_size: int, tmp_dir: str, resume_from: str) -> None:
    import os

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29523", WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK="0"
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from merge_and_rebase.utils.distributed import DistInfo

        _run_resumable(
            Path(tmp_dir) / "resumed_w2",
            _DummyMonkeypatch(),
            max_steps=4,
            resume_from=resume_from,
            dist_info=DistInfo(rank=rank, world_size=world_size),
        )
    finally:
        dist.destroy_process_group()


def test_checkpoint_from_one_rank_resumes_on_two_ranks(tmp_path) -> None:
    """The sweep runs on 4 GPUs and the continuation on 8: a checkpoint must
    carry over to a different world size and land on exactly the same weights."""
    import torch.multiprocessing as mp

    _run_resumable(tmp_path / "straight", _DummyMonkeypatch(), max_steps=4)
    _run_resumable(tmp_path / "first", _DummyMonkeypatch(), max_steps=2)
    ckpt = str(tmp_path / "first" / "dummy" / "resume_causal" / "resume" / "step_0000002.pt")
    assert torch.load(ckpt, weights_only=False)["rows_consumed_in_epoch"] == 8  # 2 steps x 2 micro x bs 2

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_dp_resume_worker, args=(r, 2, str(tmp_path), ckpt)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    def _state(sub):
        path = tmp_path / sub / "dummy" / "resume_causal" / "resume" / "step_0000004.pt"
        return torch.load(path, weights_only=False)["trainable_state"]

    straight, resumed = _state("straight"), _state("resumed_w2")
    assert set(straight) == set(resumed)
    for name, value in straight.items():
        torch.testing.assert_close(resumed[name], value, rtol=1e-5, atol=1e-7)


def test_rslora_scales_by_alpha_over_sqrt_rank() -> None:
    model = _tiny_causal_lm()
    model, _, _, _, peft_out = train_text._configure_text_strategy(
        model=model,
        strategy="peft_lora",
        strategy_cfg={"peft": {"target_modules": ["q_proj"], "r": 16, "lora_alpha": 16, "use_rslora": True}},
        optimizer_name="adamw",
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=0,
        scheduler_name="constant",
        steps=1,
        device=torch.device("cpu"),
        model_kind="causal_lm",
    )
    scalings = {
        float(v)
        for m in model.modules()
        if hasattr(m, "scaling") and isinstance(m.scaling, dict)
        for v in m.scaling.values()
    }
    assert len(scalings) == 1 and scalings.pop() == pytest.approx(16 / 16**0.5)  # 4.0, not alpha/r = 1.0
    assert peft_out["use_rslora"] is True
