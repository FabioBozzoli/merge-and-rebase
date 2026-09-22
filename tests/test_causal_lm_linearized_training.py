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
from merge_and_rebase.utils.peft_materialization import materialized_peft_param_map

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
    # PEFT zero-inits lora_B, which is what makes the bind-time snapshot equal
    # the pretrained weights. Move off zero only afterwards.
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
            ids = [(abs(hash(w)) % (VOCAB - 2)) + 2 for w in str(text).split()]
            return [2] + ids if add_special_tokens else ids

        def __call__(self, text, add_special_tokens=True, **kwargs):
            if isinstance(text, (list, tuple)):
                return {"input_ids": [self._encode(t, add_special_tokens) for t in text]}
            return {"input_ids": self._encode(text, add_special_tokens)}

    return _Tok()


def test_linearized_forward_is_first_order_expansion_around_pretrained_weights() -> None:
    """f_lin(x) == f(x; W0) + d/de f(x; W0 + e.dW)|_{e=0}.

    Checked in two halves against an *independent* plain-forward path, so a
    wrong expansion point and a wrong tangent fail separately:
      - with lora_B == 0 the linearized forward must be exactly f(x; W0);
      - the remainder must match a central difference along dW = s.B.A.
    """
    model = _linearized_peft_causal_lm()
    model.double()
    linearized = model._linearized_module
    linearized.ref_module.double()
    linearized.theta0 = tuple(t.double() for t in linearized.theta0)
    linearized.buffer_values = tuple(
        b.double() if b.is_floating_point() else b for b in linearized.buffer_values
    )

    assert all("lora_" not in name for name in linearized.param_names)
    assert any(name.endswith("q_proj.base_layer.weight") for name in linearized.param_names)

    input_ids = torch.randint(2, VOCAB, (2, 6))
    attention_mask = torch.ones_like(input_ids)

    theta0 = dict(zip(linearized.param_names, linearized.theta0, strict=True))
    buffers = dict(zip(linearized.buffer_names, linearized.buffer_values, strict=True))
    params_now = materialized_peft_param_map(model)
    tangent = {n: params_now[n].double().detach() - theta0[n] for n in linearized.param_names}
    assert max(float(t.abs().max()) for t in tangent.values()) > 0.0

    def _plain(eps: float) -> torch.Tensor:
        param_map = {n: theta0[n] + eps * tangent[n] for n in linearized.param_names}
        out = functional_call(
            linearized.ref_module,
            (param_map, buffers),
            args=(),
            kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
            strict=False,
        )
        return out.logits.detach()

    with torch.no_grad():
        actual = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Expansion point: zero tangent must reproduce the pretrained forward.
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


def test_linearized_bind_rejects_a_warm_adapter() -> None:
    """Binding after loading a trained adapter would make the expansion point
    'pretrained + frozen adapter' and double-count the delta."""
    model = get_peft_model(
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
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.1)

    with pytest.raises(RuntimeError, match="lora_B == 0"):
        apply_training_forward_mode(
            model=model,
            forward_mode="linearized_ntk",
            device=torch.device("cpu"),
            output_transform=lambda out: out.logits,
        )


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

    def _fake_eval(model, loader, device):
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
