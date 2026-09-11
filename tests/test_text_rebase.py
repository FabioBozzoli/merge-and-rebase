from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.data.text_loaders import NLIExample, NLITaskData, build_nli_tokenized_loader
from merge_and_rebase.eval import text_rebase
from merge_and_rebase.rebase.methods import theseus as theseus_mod
from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.text import (
    TextEncoderShim,
    alias_inputs_loader,
    attach_local_labels,
    balanced_indices,
    block_index_of,
    block_modules,
    head_intermediate_linears,
    num_residual_blocks,
    steer_text_correction_context,
    text_param_filter,
    text_parameter_blocks,
)
from merge_and_rebase.rebase.text.steer_text import _head_as_identity

transformers = pytest.importorskip("transformers")

VOCAB = 64
EOS = 1
PAD = 0
NUM_LABELS = 3


# --------------------------------------------------------------------------
# Tiny in-memory models and data
# --------------------------------------------------------------------------


def _tiny_t5(d_model: int = 32, num_layers: int = 2, seed: int = 0):
    from transformers import T5Config, T5ForSequenceClassification

    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=VOCAB,
        d_model=d_model,
        d_ff=2 * d_model,
        d_kv=8,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=2,
        num_labels=NUM_LABELS,
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
        dropout_rate=0.0,
    )
    return T5ForSequenceClassification(config).eval()


def _tiny_qwen(hidden_size: int = 32, num_layers: int = 2, seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForSequenceClassification

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_labels=NUM_LABELS,
        pad_token_id=PAD,
        max_position_embeddings=64,
    )
    return Qwen2ForSequenceClassification(config).eval()


class _DictDataset(Dataset):
    """Fixed-length tokenized batches: ids in [2, VOCAB) plus a trailing eos.

    T5ForSequenceClassification requires exactly one eos per row, and Qwen finds
    the pooled position via pad_token_id, so ids 0 and 1 are reserved.
    """

    def __init__(self, n: int, length: int = 6, seed: int = 0, classes: int = 2) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(2, VOCAB, (n, length), generator=g)
        self.input_ids[:, -1] = EOS
        self.attention_mask = torch.ones_like(self.input_ids)
        # Local labels 0..classes-1, each class guaranteed present many times.
        self.local = torch.arange(n) % classes
        # Head-space labels: a two-way task on a three-way head uses {0, 2}.
        self.head_class_ids = [0, 2] if classes == 2 else list(range(classes))
        self.labels = torch.tensor([self.head_class_ids[int(y)] for y in self.local])

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}


def _loader(dataset: _DictDataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def _text_loaders(n: int = 16, seed: int = 0) -> SimpleNamespace:
    train, test = _DictDataset(n, seed=seed), _DictDataset(n, seed=seed + 1)
    return SimpleNamespace(
        train=_loader(train),
        test=_loader(test),
        local_labels={"train": train.local.tolist(), "test": test.local.tolist()},
        mask_class=sorted(set(train.head_class_ids)),
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_steer_text_registered() -> None:
    assert "steer_text" in list_methods()
    assert get_method("steer_text").name == "steer_text"


def test_unsupported_methods_are_named_with_a_reason() -> None:
    for name in ("transfusion", "steer", "bico_gradin"):
        assert name in text_rebase._UNSUPPORTED_METHODS
        assert len(text_rebase._UNSUPPORTED_METHODS[name]) > 40


class _RecordingTokenizer:
    """Fake tokenizer recording the exact positional args it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        n = len(args[0])
        return {"input_ids": [[0]] * n, "attention_mask": [[1]] * n}

    def pad(self, features, return_tensors=None):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _tiny_snli_task_data() -> NLITaskData:
    return NLITaskData(
        task="snli",
        examples=[NLIExample(premise="a cat sits", hypothesis="an animal rests", label=0)],
        labels=["entailment", "neutral", "contradiction"],
        label_texts=["entailment", "neutral", "contradiction"],
        meta={},
    )


def test_build_nli_tokenized_loader_defaults_to_pair_encoding() -> None:
    tokenizer = _RecordingTokenizer()
    build_nli_tokenized_loader(task_data=_tiny_snli_task_data(), tokenizer=tokenizer, max_length=32)
    assert len(tokenizer.calls) == 1
    (premises, hypotheses) = tokenizer.calls[0]
    assert premises == ["a cat sits"]
    assert hypotheses == ["an animal rests"]


def test_build_nli_tokenized_loader_applies_a_single_string_template() -> None:
    # A checkpoint fine-tuned on "premise: X hypothesis: Y" as one string
    # performs at chance under the default pair encoding (see
    # scripts/probe_nli_input_format.py) -- this is the fix.
    tokenizer = _RecordingTokenizer()
    build_nli_tokenized_loader(
        task_data=_tiny_snli_task_data(),
        tokenizer=tokenizer,
        max_length=32,
        premise_hypothesis_template="premise: {premise} hypothesis: {hypothesis}",
    )
    assert len(tokenizer.calls) == 1
    (texts,) = tokenizer.calls[0]
    assert texts == ["premise: a cat sits hypothesis: an animal rests"]


def test_is_hub_model_reference_distinguishes_hub_ids_from_local_paths(tmp_path) -> None:
    # A full HF Hub checkpoint (e.g. varun-v-rao/t5-base-snli): no local file,
    # has a slash, no weight-file extension.
    assert text_rebase._is_hub_model_reference("varun-v-rao/t5-base-snli")
    # A local delta file living alongside a shared base -- even one that
    # happens to contain a slash in its path -- is never a Hub reference.
    local = tmp_path / "sub" / "full_best_ep.pt"
    local.parent.mkdir()
    local.write_bytes(b"not a real checkpoint, just needs to exist")
    assert not text_rebase._is_hub_model_reference(str(local))
    # A hub-style path string that ends in a weight extension is a filename,
    # not a bare repo id, even though it wasn't found on disk.
    assert not text_rebase._is_hub_model_reference("someorg/somerepo/full_best_ep.pt")
    # No slash at all: neither shape.
    assert not text_rebase._is_hub_model_reference("full_best_ep.pt")


# --------------------------------------------------------------------------
# val/test carve
# --------------------------------------------------------------------------


def _fake_task_data(n: int, task: str = "rte") -> NLITaskData:
    return NLITaskData(
        task=task,
        examples=[NLIExample(premise=f"p{i}", hypothesis=f"h{i}", label=i % 2) for i in range(n)],
        labels=["entailment", "contradiction"],
        label_texts=["entailment", "contradiction"],
        meta={"task": task},
    )


def test_val_test_carve_is_disjoint_and_reproducible(monkeypatch) -> None:
    def fake_build(*, task, split, max_samples=None):  # noqa: ARG001
        return _fake_task_data(50 if split == "train" else 100, task=task)

    monkeypatch.setattr(text_rebase, "build_nli_task_data", fake_build)

    splits = text_rebase._build_task_splits(
        task="rte", eval_split="test", val_fraction=0.2, max_train_samples=None, max_eval_samples=None
    )
    val_ids = {ex.premise for ex in splits["val"].examples}
    test_ids = {ex.premise for ex in splits["test"].examples}

    assert len(val_ids) == 20
    assert len(test_ids) == 80
    assert not (val_ids & test_ids), "val and test must be disjoint (no leakage into the alpha search)"
    assert len(splits["train"].examples) == 50

    again = text_rebase._build_task_splits(
        task="rte", eval_split="test", val_fraction=0.2, max_train_samples=None, max_eval_samples=None
    )
    assert [ex.premise for ex in again["val"].examples] == [ex.premise for ex in splits["val"].examples]


def test_val_fraction_that_carves_nothing_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(text_rebase, "build_nli_task_data", lambda **kw: _fake_task_data(10))
    with pytest.raises(ValueError, match="carves"):
        text_rebase._build_task_splits(
            task="rte", eval_split="test", val_fraction=0.01, max_train_samples=None, max_eval_samples=None
        )


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


def test_shim_exposes_the_hf_model_under_theseus_module_lookup() -> None:
    model = _tiny_t5()
    shim = TextEncoderShim(model, pad_token_id=PAD)

    assert theseus_mod._visual_module(shim) is model
    # Hook keys must line up with the state-dict / delta keyspace, otherwise
    # theseus._precompute_transforms silently matches nothing.
    module_names = {n for n, _ in model.named_modules() if n}
    for key in model.state_dict():
        assert key.rsplit(".", 1)[0] in module_names

    # No fused OpenCLIP attention -> the qkv split/merge path stays a no-op.
    assert theseus_mod._has_fused_mha(model) is False


def test_shim_encode_image_and_forward_both_run() -> None:
    model = _tiny_t5()
    shim = TextEncoderShim(model, pad_token_id=PAD)
    dataset = _DictDataset(4)
    batch = _collate([dataset[i] for i in range(4)])

    with torch.no_grad():
        logits = shim.encode_image(batch["input_ids"])
        direct = shim(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert logits.shape == (4, NUM_LABELS)
    assert direct.shape == (4, NUM_LABELS)


def test_alias_inputs_loader_is_extractable_by_theseus() -> None:
    loader = alias_inputs_loader(_loader(_DictDataset(8)))
    batch = next(iter(loader))
    assert "inputs" in batch
    extracted = theseus_mod._extract_model_inputs(batch)
    assert torch.equal(extracted, batch["input_ids"])
    # _iter_random_dataset_batches needs all three of these.
    assert loader.dataset is not None and loader.batch_size == 4 and callable(loader.collate_fn)


def test_attach_local_labels_enables_class_balanced_calibration() -> None:
    dataset = _DictDataset(16, classes=2)
    attach_local_labels(dataset, dataset.local.tolist())

    labels = theseus_mod._dataset_labels(dataset)
    assert labels is not None
    assert torch.equal(labels, dataset.local)

    picked = theseus_mod._class_balanced_indices(dataset, shots_per_class=3, seed=0)
    assert picked.numel() == 6
    assert sorted(int(dataset.local[i]) for i in picked.tolist()) == [0, 0, 0, 1, 1, 1]


def test_attach_local_labels_rejects_wrong_length() -> None:
    dataset = _DictDataset(8)
    with pytest.raises(ValueError, match="entries but dataset has"):
        attach_local_labels(dataset, [0, 1, 0])


def test_head_space_labels_would_break_class_balanced_calibration() -> None:
    """Guards the reason attach_local_labels insists on local ids."""
    dataset = _DictDataset(16, classes=2)
    attach_local_labels(dataset, dataset.labels.tolist())  # head space {0, 2}: class 1 is empty
    with pytest.raises(ValueError, match="only has 0 examples"):
        theseus_mod._class_balanced_indices(dataset, shots_per_class=2, seed=0)


def test_balanced_indices_is_per_class_and_capped() -> None:
    labels = [0] * 10 + [1] * 3
    picked = balanced_indices(labels, per_class=5, seed=0)
    assert sum(1 for i in picked if labels[i] == 0) == 5
    assert sum(1 for i in picked if labels[i] == 1) == 3


def test_text_param_filter_drops_the_head_and_non_float_buffers() -> None:
    keep = text_param_filter(exclude_head=True)
    assert keep("encoder.block.0.layer.0.SelfAttention.q.weight", torch.zeros(4, 4))
    assert not keep("classification_head.out_proj.weight", torch.zeros(3, 4))
    assert not keep("score.weight", torch.zeros(3, 4))
    assert not keep("encoder.embed_positions.position_ids", torch.zeros(4))
    assert not keep("something.int_buffer", torch.zeros(4, dtype=torch.long))

    keep_all = text_param_filter(exclude_head=False)
    assert keep_all("score.weight", torch.zeros(3, 4))


# --------------------------------------------------------------------------
# Block map
# --------------------------------------------------------------------------


def test_block_map_t5_covers_encoder_then_decoder() -> None:
    model = _tiny_t5(num_layers=2)
    # T5ForSequenceClassification nests the stacks under `transformer.`, which is
    # exactly why the patterns are not anchored at the start of the name.
    assert block_index_of("transformer.encoder.block.1.layer.0.SelfAttention.q.weight") == ("encoder", 1)
    assert block_index_of("transformer.decoder.block.0.layer.1.EncDecAttention.q.weight") == ("decoder", 0)
    assert block_index_of("transformer.shared.weight") is None

    assert num_residual_blocks(model) == 4  # 2 encoder + 2 decoder
    modules = block_modules(model)
    assert len(modules) == 4
    assert modules[0] is model.transformer.encoder.block[0]
    assert modules[2] is model.transformer.decoder.block[0]

    block_ids, num_blocks = text_parameter_blocks(model)
    assert num_blocks == 5  # 4 residual + 1 output block
    assert len(block_ids) == len(list(model.named_parameters()))
    assert max(block_ids) == 4
    by_name = dict(zip([n for n, _ in model.named_parameters()], block_ids, strict=True))
    assert by_name["transformer.shared.weight"] == 0  # embeddings fold into block 0
    assert by_name["classification_head.out_proj.weight"] == 4  # head -> output block


def test_block_map_qwen_decoder_only() -> None:
    model = _tiny_qwen(num_layers=3)
    assert block_index_of("model.layers.2.self_attn.q_proj.weight") == ("layers", 2)
    assert num_residual_blocks(model) == 3
    assert block_modules(model)[1] is model.model.layers[1]

    block_ids, num_blocks = text_parameter_blocks(model)
    assert num_blocks == 4
    by_name = dict(zip([n for n, _ in model.named_parameters()], block_ids, strict=True))
    assert by_name["model.embed_tokens.weight"] == 0
    assert by_name["score.weight"] == 3


# --------------------------------------------------------------------------
# Pooled feature extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory,dim", [(_tiny_t5, 32), (_tiny_qwen, 32)])
def test_head_as_identity_yields_the_pooled_feature(factory, dim) -> None:
    model = factory()
    dataset = _DictDataset(4)
    batch = _collate([dataset[i] for i in range(4)])
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

    with torch.no_grad():
        before = model(**kwargs).logits
        with _head_as_identity(model) as head:
            pooled = model(**kwargs).logits
        after = model(**kwargs).logits

    assert before.shape == (4, NUM_LABELS)
    assert pooled.shape == (4, dim), "with the head swapped out, logits are the head's input"
    assert torch.allclose(after, before), "the head must be restored on exit"

    # The swap is exact: re-applying the head reproduces the real logits.
    expected = pooled @ head.weight.T + (head.bias if head.bias is not None else 0.0)
    assert torch.allclose(expected, before, atol=1e-4)


def test_head_intermediate_linears_flags_t5s_untrained_dense_layer() -> None:
    # T5ForSequenceClassification's head is dense -> tanh -> out_proj; `dense`
    # has no pretrained weights (always randomly initialized), so a nearest-mean
    # head built on its output is class means of a random rotation of the
    # model's real representation unless dense is neutralized first.
    model = _tiny_t5()
    intermediate = head_intermediate_linears(model)
    assert [name for name, _ in intermediate] == ["classification_head.dense"]
    dense = intermediate[0][1]
    assert dense.weight.shape == (dense.weight.shape[0], dense.weight.shape[0]), "must be square to neutralize to identity"


def test_head_intermediate_linears_empty_for_a_bare_decoder_only_head() -> None:
    # Qwen/Llama-style sequence classification heads are a single Linear
    # straight on the pooled feature -- nothing to neutralize.
    model = _tiny_qwen()
    assert head_intermediate_linears(model) == []


# --------------------------------------------------------------------------
# Methods end to end
# --------------------------------------------------------------------------


def _delta_between(model_a, model_b) -> dict[str, torch.Tensor]:
    keep = text_param_filter(exclude_head=True)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    return {
        k: (sd_b[k].float() - sd_a[k].float())
        for k in sd_a
        if k in sd_b and sd_a[k].shape == sd_b[k].shape and keep(k, sd_a[k])
    }


def test_gradfix_runs_unmodified_on_a_text_model() -> None:
    from merge_and_rebase.models.grad_recipes import seq_classification_recipe

    base, tuned = _tiny_t5(seed=0), _tiny_t5(seed=1)
    delta = _delta_between(base, tuned)
    assert delta, "expected a non-empty delta"

    method = get_method("gradfix")
    prepared = method.prepare(
        target_model=base,
        target_dataloader=_loader(_DictDataset(8)),
        recipe=seq_classification_recipe(device="cpu", mask_class=[0, 2]),
        device="cpu",
        vote="mean",
    )
    assert prepared, "gradient signs should not be empty"

    out = method.transport(
        source_base=base.state_dict(),
        target_base=base.state_dict(),
        delta=delta,
        prepared=prepared,
        mask_mode="normal",
    )
    assert set(out) == set(delta)
    for key, value in out.items():
        assert value.shape == delta[key].shape
        # "normal" masking only ever zeroes entries, never changes magnitude.
        assert torch.all((value == 0) | (value == delta[key]))


@pytest.mark.parametrize("method_name", ["theseus", "bico"])
def test_theseus_and_bico_transport_across_widths(method_name) -> None:
    from merge_and_rebase.models.grad_recipes import seq_classification_recipe

    source_base, source_tuned = _tiny_t5(d_model=32, seed=0), _tiny_t5(d_model=32, seed=1)
    target_base = _tiny_t5(d_model=48, seed=2)
    delta = _delta_between(source_base, source_tuned)
    target_sd = {k: v.float() for k, v in target_base.state_dict().items()}

    source_loader = alias_inputs_loader(_loader(_DictDataset(8, seed=0)))
    target_loader = alias_inputs_loader(_loader(_DictDataset(8, seed=0)))

    method = get_method(method_name)
    kwargs = {
        "source_model": TextEncoderShim(source_base, PAD),
        "target_model": TextEncoderShim(target_base, PAD),
        "source_dataloader": source_loader,
        "target_dataloader": target_loader,
        "target_base": target_sd,
        "delta": delta,
        "device": "cpu",
        "seq_align": "interpolate",
        "num_batches": 1,
        "patch_qkv": False,
        "verbose": False,
        "show_progress": False,
    }
    if method_name == "bico":
        recipe = seq_classification_recipe(device="cpu", mask_class=[0, 2])
        kwargs["source_recipe"] = recipe
        kwargs["target_recipe"] = recipe

    prepared = method.prepare(**kwargs)
    assert prepared["activation_registry"], "no activations were collected through the shim"
    assert prepared["transforms_by_key"], "no per-layer transforms were computed"

    out = method.transport(
        source_base={k: v.float() for k, v in source_base.state_dict().items()},
        target_base=target_sd,
        delta=delta,
        prepared=prepared,
        verbose=False,
        show_progress=False,
    )
    assert out, "transport produced no keys"
    for key, value in out.items():
        # Transported deltas must live in the *target*'s shapes, not the source's.
        assert value.shape == target_sd[key].shape
        assert torch.isfinite(value).all()


# --------------------------------------------------------------------------
# steer_text
# --------------------------------------------------------------------------


def _steer_prepare(tmp_path, **overrides):
    source_pre = _tiny_t5(d_model=32, seed=0)
    source_ft = _tiny_t5(d_model=32, seed=1)
    target = _tiny_t5(d_model=48, seed=2)

    params = {
        "feature_regime": "standard",
        "stage_2_strategy": "global_ridge",
        "few_shot": 4,
        "feature_cache_dir": str(tmp_path / "cache"),
        "seed": 0,
        "verbose": False,
    }
    params.update(overrides)

    prepared = get_method("steer_text").prepare(
        llm_source=SimpleNamespace(model=source_ft),
        llm_source_pretrained=SimpleNamespace(model=source_pre),
        llm_target=SimpleNamespace(model=target),
        source_loaders=_text_loaders(),
        target_loaders=_text_loaders(),
        task="rte",
        mask_class=[0, 2],
        device="cpu",
        source_tag="src",
        target_tag="tgt",
        **params,
    )
    return prepared, target


def test_steer_text_prepare_fits_and_caches(tmp_path) -> None:
    prepared, _ = _steer_prepare(tmp_path)
    assert callable(prepared["correction_fn"])
    for key in ("stage0_test_acc", "stage1_test_acc", "stage2_test_acc"):
        assert 0.0 <= prepared["diagnostics"][key] <= 1.0

    cache = tmp_path / "cache" / "src_to_tgt" / "rte" / "standard"
    assert (cache / "train" / "features_A.pt").exists()
    assert (cache / "test" / "features_B.pt").exists()

    # A second call must reuse the cache rather than recompute.
    again, _ = _steer_prepare(tmp_path)
    assert again["diagnostics"] == prepared["diagnostics"]


def test_steer_text_never_produces_a_weight_delta(tmp_path) -> None:
    prepared, _ = _steer_prepare(tmp_path)
    assert get_method("steer_text").transport(
        source_base={}, target_base={}, delta={"a": torch.zeros(2)}, prepared=prepared
    ) == {}


def test_steer_text_correction_shifts_logits_and_restores(tmp_path) -> None:
    prepared, target = _steer_prepare(tmp_path)
    _, head = text_rebase_head(target)

    dataset = _DictDataset(4)
    batch = _collate([dataset[i] for i in range(4)])
    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    llm = SimpleNamespace(model=target)

    with torch.no_grad():
        with _head_as_identity(target):
            pooled = target(**kwargs).logits
        baseline = target(**kwargs).logits
        with steer_text_correction_context(llm, prepared, alpha=2.0):
            corrected = target(**kwargs).logits
        restored = target(**kwargs).logits

    correction = 2.0 * prepared["correction_fn"]({"global": pooled})
    expected = baseline + correction @ head.weight.T
    assert torch.allclose(corrected, expected, atol=1e-4)
    assert torch.allclose(restored, baseline), "the hook must be removed on exit"

    with torch.no_grad(), steer_text_correction_context(llm, prepared, alpha=0.0):
        assert torch.allclose(target(**kwargs).logits, baseline, atol=1e-6)


def test_steer_text_rejects_contradictory_params(tmp_path) -> None:
    with pytest.raises(ValueError, match="block_ridge requires feature_regime='linear'"):
        _steer_prepare(tmp_path, stage_2_strategy="block_ridge")
    with pytest.raises(ValueError, match="exactly one of few_shot or total_support_examples"):
        _steer_prepare(tmp_path, few_shot=None)


def test_steer_text_detects_an_unloaded_checkpoint(tmp_path) -> None:
    same = _tiny_t5(seed=0)
    with pytest.raises(ValueError, match="identical to the pretrained"):
        get_method("steer_text").prepare(
            llm_source=SimpleNamespace(model=same),
            llm_source_pretrained=SimpleNamespace(model=same),
            llm_target=SimpleNamespace(model=_tiny_t5(d_model=48, seed=2)),
            source_loaders=_text_loaders(),
            target_loaders=_text_loaders(),
            task="rte",
            device="cpu",
            few_shot=4,
            feature_cache_dir=str(tmp_path / "cache"),
            verbose=False,
        )


def text_rebase_head(model):
    from merge_and_rebase.rebase.text import head_linear

    return head_linear(model)


def test_neutralize_intermediate_head_layers_makes_t5s_dense_an_identity() -> None:
    from merge_and_rebase.rebase.text import neutralize_intermediate_head_layers

    model = _tiny_t5(seed=0)
    dense = model.classification_head.dense
    assert not torch.equal(dense.weight, torch.eye(dense.weight.shape[0]))

    written = neutralize_intermediate_head_layers(model)

    assert torch.equal(dense.weight, torch.eye(dense.weight.shape[0]))
    assert torch.equal(dense.bias, torch.zeros_like(dense.bias))
    # The written tensors are what a task-head payload must carry to restore
    # this same space at injection time.
    assert set(written) == {"classification_head.dense.weight", "classification_head.dense.bias"}


def test_neutralize_intermediate_head_layers_is_a_noop_without_one() -> None:
    from merge_and_rebase.rebase.text import neutralize_intermediate_head_layers

    assert neutralize_intermediate_head_layers(_tiny_qwen(seed=0)) == {}


def test_train_linear_probe_head_starts_from_the_head_already_in_the_model() -> None:
    """It must not re-initialize the head: steer_text's correction is fit via
    pinv(w_b) against the head that is live at prepare() time, so a fresh draw
    here would discard the readout the correction only works through."""
    from merge_and_rebase.rebase.text import train_linear_probe_head

    model = _tiny_t5(seed=0)
    loaders = _text_loaders(n=8, seed=0)

    marker = torch.full_like(model.classification_head.out_proj.weight, 0.123)
    with torch.no_grad():
        model.classification_head.out_proj.weight.copy_(marker)

    # lr=0 -> Adam applies no update, so anything that changed came from a reset.
    train_linear_probe_head(model, loaders.train, device="cpu", mask_class=loaders.mask_class, lr=0.0, steps=1)

    assert torch.equal(model.classification_head.out_proj.weight, marker)


def test_train_linear_probe_head_fits_a_few_shot_support_set() -> None:
    from merge_and_rebase.rebase.text import train_linear_probe_head

    model = _tiny_t5(seed=0)
    loaders = _text_loaders(n=12, seed=0)  # classes=2 default -> head_class_ids [0, 2]

    original_backbone = {n: p.clone() for n, p in model.named_parameters() if "classification_head" not in n}
    original_dense = {n: p.clone() for n, p in model.named_parameters() if "classification_head.dense" in n}
    original_out_proj = {n: p.clone() for n, p in model.named_parameters() if "classification_head.out_proj" in n}

    trained = train_linear_probe_head(
        model,
        loaders.train,
        device="cpu",
        mask_class=loaders.mask_class,
        lr=0.05,
        steps=50,
        eval_loaders={"support": loaders.train},
    )

    # Backbone must stay untouched -- only the head's final linear is ever trained.
    for n, p in model.named_parameters():
        if "classification_head" not in n:
            assert torch.equal(p, original_backbone[n])
    # T5's intermediate layer (dense) must ALSO stay untouched: steer_text's cached
    # features/correction are fit against that exact draw, so resetting it here
    # would invalidate them (see the function's docstring).
    for n, p in model.named_parameters():
        if n in original_dense:
            assert torch.equal(p, original_dense[n])
    # out_proj (the actual linear probe) must have moved from its starting point.
    assert any(not torch.equal(p, original_out_proj[n]) for n, p in model.named_parameters() if n in original_out_proj)
    # requires_grad is restored to its pre-call state (a fresh model: all True).
    assert all(p.requires_grad for p in model.parameters())
    # Returned dict contains exactly out_proj's own qualified parameter names/shapes
    # -- NOT dense, which was never touched.
    assert set(trained) == set(original_out_proj)
    for n, p in original_out_proj.items():
        assert trained[n].shape == p.shape

    # It should have actually fit the tiny few-shot set (near-perfect train accuracy).
    model.eval()
    idx = torch.tensor(loaders.mask_class, dtype=torch.long)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loaders.train:
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            pred = idx[logits.index_select(dim=1, index=idx).argmax(dim=-1)]
            correct += int((pred == batch["labels"]).sum())
            total += int(batch["labels"].numel())
    assert correct / total >= 0.9
