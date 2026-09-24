from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from merge_and_rebase.rebase.text.adapters import head_intermediate_linears, head_linear, text_param_filter
from merge_and_rebase.rebase.text.encoder_classifier import (
    T5EncoderForSequenceClassification,
    masked_mean,
    segment_masks_from_eos,
    segment_pooled,
)
from merge_and_rebase.rebase.text.steer_text import (
    _head_as_identity,
    _masked_mean,
    block_modules,
    num_residual_blocks,
    text_parameter_blocks,
)

_LAYERS = 2
_D_MODEL = 16
_VOCAB = 64


def _config(num_labels: int = 3, num_layers: int = _LAYERS) -> T5Config:
    return T5Config(
        d_model=_D_MODEL,
        d_ff=32,
        d_kv=4,
        num_layers=num_layers,
        num_heads=2,
        vocab_size=_VOCAB,
        num_labels=num_labels,
    )


def _model(num_labels: int = 3, num_layers: int = _LAYERS, seed: int = 0) -> T5EncoderForSequenceClassification:
    torch.manual_seed(seed)
    model = T5EncoderForSequenceClassification(_config(num_labels, num_layers)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return model


def _batch(batch: int = 4, tokens: int = 7, pad_from: int = 5):
    ids = torch.randint(1, _VOCAB - 1, (batch, tokens))
    mask = torch.ones(batch, tokens, dtype=torch.long)
    mask[:, pad_from:] = 0
    return ids, mask


# ---------------------------------------------------------------------------
# The decoder is gone, and the block map sees only encoder blocks
# ---------------------------------------------------------------------------


def test_no_decoder_parameters() -> None:
    names = [n for n, _ in _model().named_parameters()]
    assert names, "model has no parameters"
    assert not [n for n in names if "decoder" in n]


def test_block_map_counts_only_encoder_blocks() -> None:
    """``text_parameter_blocks`` must report N residual blocks + 1 output block.

    Under ``T5ForSequenceClassification`` this same call returns 2N (encoder and
    decoder), which is what makes ``block_ridge`` group across a meaningless
    stack boundary. The point of the encoder-only wrapper is that it cannot.
    """
    model = _model()
    assert num_residual_blocks(model) == _LAYERS
    block_ids, num_blocks = text_parameter_blocks(model)
    assert num_blocks == _LAYERS + 1
    assert len(block_ids) == len(list(model.named_parameters()))
    assert len(block_modules(model)) == _LAYERS


def test_block_modules_capture_hidden_states() -> None:
    """Each block module's output[0] must be ``[B, T, D]`` for the capture hook."""
    model = _model()
    ids, mask = _batch()
    seen: list[tuple[int, ...]] = []

    def hook(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        seen.append(tuple(out.shape))

    handles = [m.register_forward_hook(hook) for m in block_modules(model)]
    try:
        model(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()

    assert len(seen) == _LAYERS
    assert all(shape == (ids.shape[0], ids.shape[1], _D_MODEL) for shape in seen)


# ---------------------------------------------------------------------------
# The head contract steer_text and llm_merge both depend on
# ---------------------------------------------------------------------------


def test_head_is_a_single_linear_named_for_t5() -> None:
    """The name is load-bearing, not cosmetic.

    ``llm_merge._task_head_tensor_for_param`` gates its ``head_class_ids``
    scatter on names ending in ``classification_head.out_proj.weight``/
    ``.bias``; a head named ``score`` would silently lose that path. And a
    single linear is what makes ``head_intermediate_linears`` empty, i.e. no
    untrained ``dense`` between the representation and the readout.
    """
    model = _model()
    name, module = head_linear(model)
    assert name == "classification_head.out_proj"
    assert isinstance(module, nn.Linear)
    assert head_intermediate_linears(model) == []


def test_head_as_identity_exposes_the_pooled_feature() -> None:
    """``_head_as_identity`` is how every feature consumer reads pre-head activations."""
    model = _model()
    ids, mask = _batch()
    logits = model(input_ids=ids, attention_mask=mask).logits
    _, head = head_linear(model)

    with _head_as_identity(model):
        pooled = model(input_ids=ids, attention_mask=mask).logits

    assert pooled.shape == (ids.shape[0], _D_MODEL)
    assert torch.allclose(head(pooled), logits, atol=1e-6)


def test_head_is_excluded_from_the_task_vector() -> None:
    keep = text_param_filter()
    names = [n for n, _ in _model().named_parameters()]
    assert not [n for n in names if "classification_head" in n and keep(n, torch.zeros(1))]
    assert [n for n in names if ".block." in n and keep(n, torch.zeros(1))]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_masked_mean_respects_the_mask() -> None:
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    expected = torch.stack([hidden[0, :2].mean(dim=0), hidden[1, :1].mean(dim=0)])
    assert torch.allclose(masked_mean(hidden, mask), expected)
    assert torch.allclose(masked_mean(hidden, None), hidden.mean(dim=1))


def test_steer_text_pools_with_the_same_function() -> None:
    """One definition, aliased -- so the pooled feature and the per-block
    activations cannot drift apart by someone editing one of two copies."""
    assert _masked_mean is masked_mean


def test_pooled_feature_responds_to_padding() -> None:
    """A pooled feature that ignores the mask would look fine but carry padding."""
    model = _model()
    ids, mask = _batch(pad_from=5)
    with _head_as_identity(model):
        masked = model(input_ids=ids, attention_mask=mask).logits
        unmasked = model(input_ids=ids, attention_mask=torch.ones_like(mask)).logits
    assert not torch.allclose(masked, unmasked, atol=1e-5)


def test_segment_masks_from_eos_splits_premise_and_hypothesis() -> None:
    """``premise <eos> hypothesis <eos>``, two different segment lengths plus trailing pad."""
    eos = 1
    input_ids = torch.tensor(
        [
            [5, 6, eos, 7, eos, 0],  # premise=2 tok, hyp=1 tok, no pad
            [5, eos, 6, 7, eos, 0],  # premise=1 tok, hyp=2 tok, no pad
            [5, eos, 6, eos, 0, 0],  # premise=1 tok, hyp=1 tok, 2 pad
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    seg1, seg2 = segment_masks_from_eos(input_ids, attention_mask, eos)
    assert seg1.tolist() == [
        [True, True, False, False, False, False],
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
    ]
    assert seg2.tolist() == [
        [False, False, False, True, False, False],
        [False, False, True, True, False, False],
        [False, False, True, False, False, False],
    ]
    # Padding never leaks into either segment, even where an all-real mask would allow it.
    assert not (seg1 & ~attention_mask.bool()).any()
    assert not (seg2 & ~attention_mask.bool()).any()


def test_segment_pooled_matches_masked_mean_on_each_slice() -> None:
    eos = 1
    input_ids = torch.tensor([[5, 6, eos, 7, eos, 0], [5, eos, 6, 7, eos, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]])
    hidden = torch.randn(2, 6, 4)

    seg1, seg2 = segment_masks_from_eos(input_ids, attention_mask, eos)
    pooled = segment_pooled(hidden, attention_mask, input_ids, eos)
    expected = torch.cat(
        [masked_mean(hidden, seg1), masked_mean(hidden, seg2), masked_mean(hidden, attention_mask)], dim=-1
    )
    assert pooled.shape == (2, 4 * 3)
    assert torch.allclose(pooled, expected)


def test_segment_masks_from_eos_rejects_a_row_with_a_single_eos() -> None:
    eos = 1
    input_ids = torch.tensor([[5, 6, eos, 7, 8]])  # only one EOS: no hypothesis boundary
    attention_mask = torch.ones_like(input_ids)
    with pytest.raises(ValueError, match="fewer than 2 EOS"):
        segment_masks_from_eos(input_ids, attention_mask, eos)


# ---------------------------------------------------------------------------
# from_pretrained / save_pretrained
# ---------------------------------------------------------------------------


def test_save_and_reload_round_trip_is_exact(tmp_path) -> None:
    """Includes the tied ``shared``/``embed_tokens`` weight.

    They alias one storage, and safetensors refuses aliased tensors, so this is
    the regression test for using ``save_pretrained`` (which drops the tied
    duplicate and re-ties on load) rather than a raw ``safetensors.save_file``.
    """
    model = _model()
    shared = model.transformer.shared.weight
    embed = model.transformer.encoder.embed_tokens.weight
    assert shared.data_ptr() == embed.data_ptr(), "precondition: the embedding is tied"

    out = tmp_path / "converted"
    model.save_pretrained(out)
    reloaded = T5EncoderForSequenceClassification.from_pretrained(out).eval()

    original = model.state_dict()
    for key, value in reloaded.state_dict().items():
        assert key in original, f"unexpected key after reload: {key}"
        assert torch.equal(value, original[key]), f"tensor changed across the round trip: {key}"

    assert (
        reloaded.transformer.shared.weight.data_ptr()
        == reloaded.transformer.encoder.embed_tokens.weight.data_ptr()
    ), "embedding was not re-tied on load"


def test_loads_a_base_t5_checkpoint_without_silently_reinitializing(tmp_path) -> None:
    """A base T5 checkpoint has unprefixed keys (``encoder.block.N...``) plus a
    decoder and an LM head. ``base_model_prefix`` is what maps them onto
    ``transformer.*`` here; if that failed the encoder would be *randomly
    initialized* with no error, which is the quiet failure this guards.
    """
    torch.manual_seed(0)
    base = T5ForConditionalGeneration(_config()).eval()
    with torch.no_grad():
        for p in base.parameters():
            p.normal_(0.0, 0.02)

    ckpt = tmp_path / "base"
    base.save_pretrained(ckpt)
    loaded = T5EncoderForSequenceClassification.from_pretrained(ckpt, config=_config(num_labels=3)).eval()

    got = loaded.state_dict()
    checked = 0
    for key, value in base.state_dict().items():
        if key.startswith("decoder.") or key.startswith("lm_head"):
            continue
        target = f"transformer.{key}"
        assert target in got, f"encoder tensor not loaded: {key}"
        assert torch.equal(got[target], value), f"encoder tensor differs after load: {key}"
        checked += 1

    assert checked > 0, "no encoder tensors were compared"
    assert not [k for k in got if "decoder" in k or "lm_head" in k]


def test_forward_reports_loss_when_labels_are_given() -> None:
    model = _model()
    ids, mask = _batch()
    labels = torch.randint(0, 3, (ids.shape[0],))
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert out.loss is not None and out.loss.ndim == 0
    assert out.logits.shape == (ids.shape[0], 3)
    assert model(input_ids=ids, attention_mask=mask).loss is None


# ---------------------------------------------------------------------------
# TextLM.build integration
#
# These need a real tokenizer, which no synthetic config can provide, so they
# skip unless a T5 checkpoint is already in the local HF cache. Every test above
# stays fully offline and synthetic, as the rest of this suite is.
# ---------------------------------------------------------------------------

_REAL_T5 = "google-t5/t5-small"


def _build_or_skip(model_kind: str = "encoder_classification"):
    import pytest

    from merge_and_rebase.models.text_lm import TextBuildConfig, TextLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = TextBuildConfig(
        model_name_or_path=_REAL_T5,
        model_arch="t5",
        device="cpu",
        dtype="fp32",
        model_kind=model_kind,
        num_labels=3,
    )
    try:
        return TextLM.build(cfg)
    except Exception as exc:  # noqa: BLE001 - any offline/cache failure is a skip
        pytest.skip(f"{_REAL_T5} is not available offline: {type(exc).__name__}")


def test_textlm_build_encoder_classification() -> None:
    llm = _build_or_skip()
    model = llm.model
    assert type(model).__name__ == "T5EncoderForSequenceClassification"
    assert not [n for n, _ in model.named_parameters() if "decoder" in n]
    # Prompt mode must refuse this kind: there is no decoder to score against.
    assert llm._is_encoder_decoder() is False
    assert head_linear(model)[0] == "classification_head.out_proj"
    assert head_intermediate_linears(model) == []


def test_textlm_build_loads_real_pretrained_encoder_weights() -> None:
    """The failure this guards is silent: a key-space mismatch would leave the
    encoder randomly initialized and still return well-shaped logits."""
    from transformers import T5EncoderModel

    llm = _build_or_skip()
    reference = T5EncoderModel.from_pretrained(_REAL_T5)
    got = dict(llm.model.transformer.encoder.named_parameters())
    want = dict(reference.encoder.named_parameters())
    assert set(want).issubset(got), "encoder parameters are missing from the wrapper"
    for name, value in want.items():
        assert torch.equal(got[name], value), f"encoder weight not loaded from the checkpoint: {name}"


def test_block_map_sees_no_decoder_on_a_real_checkpoint() -> None:
    from merge_and_rebase.rebase.text.adapters import count_transformer_blocks

    llm = _build_or_skip()
    counts = count_transformer_blocks(llm.model)
    assert "decoder" not in counts, f"decoder blocks leaked into the block map: {counts}"
    assert counts.get("encoder", 0) > 0
