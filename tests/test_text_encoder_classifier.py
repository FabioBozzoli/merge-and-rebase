from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForMaskedLM, T5Config, T5ForConditionalGeneration

from merge_and_rebase.rebase.text.adapters import head_intermediate_linears, head_linear, text_param_filter
from merge_and_rebase.rebase.text.encoder_classifier import (
    RobertaEncoderForSequenceClassification,
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


# ---------------------------------------------------------------------------
# RoBERTa: pair-encoding layout for the segments, and the encoder-only wrapper
# ---------------------------------------------------------------------------

_RB_BOS, _RB_PAD, _RB_EOS = 0, 1, 2


def _roberta_config(num_labels: int = 3, num_layers: int = _LAYERS) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=_VOCAB,
        hidden_size=_D_MODEL,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
        pad_token_id=_RB_PAD,
        bos_token_id=_RB_BOS,
        eos_token_id=_RB_EOS,
        num_labels=num_labels,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def _roberta(num_labels: int = 3, seed: int = 0) -> RobertaEncoderForSequenceClassification:
    torch.manual_seed(seed)
    model = RobertaEncoderForSequenceClassification(_roberta_config(num_labels)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return model


def test_roberta_segment_masks_use_two_separator_eos_and_drop_the_bos() -> None:
    """``<s> p p </s></s> h </s> pad``: with the T5 rule the hypothesis would be lost."""
    ids = torch.tensor(
        [
            [_RB_BOS, 5, 6, _RB_EOS, _RB_EOS, 7, _RB_EOS, _RB_PAD],
            [_RB_BOS, 5, _RB_EOS, _RB_EOS, 6, 7, _RB_EOS, _RB_PAD],
        ]
    )
    mask = (ids != _RB_PAD).long()
    seg1, seg2 = segment_masks_from_eos(ids, mask, _RB_EOS, separator_eos=2, bos_token_id=_RB_BOS)
    assert seg1.tolist() == [
        [False, True, True, False, False, False, False, False],
        [False, True, False, False, False, False, False, False],
    ]
    assert seg2.tolist() == [
        [False, False, False, False, False, True, False, False],
        [False, False, False, False, True, True, False, False],
    ]
    # The T5 rule (one separator EOS) would leave the RoBERTa hypothesis out of both segments.
    t5_seg1, t5_seg2 = segment_masks_from_eos(ids, mask, _RB_EOS)
    assert not t5_seg2.any()

    hidden = torch.randn(2, 8, 4)
    pooled = segment_pooled(hidden, mask, ids, _RB_EOS, separator_eos=2, bos_token_id=_RB_BOS)
    expected = torch.cat([masked_mean(hidden, seg1), masked_mean(hidden, seg2), masked_mean(hidden, mask)], dim=-1)
    assert pooled.shape == (2, 12) and torch.allclose(pooled, expected)


def test_roberta_segment_masks_need_three_eos() -> None:
    ids = torch.tensor([[_RB_BOS, 5, _RB_EOS, 6, _RB_EOS]])  # T5-style layout: only two EOS
    with pytest.raises(ValueError, match="fewer than 3 EOS"):
        segment_masks_from_eos(ids, torch.ones_like(ids), _RB_EOS, separator_eos=2, bos_token_id=_RB_BOS)


def test_roberta_wrapper_has_no_mlm_or_pooler_head_and_a_single_linear_head() -> None:
    model = _roberta()
    names = [n for n, _ in model.named_parameters()]
    assert not [n for n in names if "lm_head" in n or "pooler" in n]
    name, module = head_linear(model)
    assert name == "classification_head.out_proj" and isinstance(module, nn.Linear)
    assert head_intermediate_linears(model) == []
    # Blocks are the encoder layers, matched by steer_text's existing BERT/RoBERTa pattern.
    assert num_residual_blocks(model) == _LAYERS
    assert len(block_modules(model)) == _LAYERS
    _, num_blocks = text_parameter_blocks(model)
    assert num_blocks == _LAYERS + 1


def test_roberta_head_as_identity_exposes_the_masked_mean_feature() -> None:
    model = _roberta()
    ids = torch.tensor([[_RB_BOS, 5, 6, _RB_EOS, _RB_EOS, 7, _RB_EOS, _RB_PAD]] * 2)
    mask = (ids != _RB_PAD).long()
    logits = model(input_ids=ids, attention_mask=mask).logits
    _, head = head_linear(model)
    with _head_as_identity(model):
        pooled = model(input_ids=ids, attention_mask=mask).logits
        hidden = model.roberta(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(pooled, masked_mean(hidden, mask), atol=1e-6)
    assert torch.allclose(head(pooled), logits, atol=1e-6)
    assert not torch.allclose(pooled, hidden.mean(dim=1), atol=1e-6)  # padding is masked out


def test_roberta_wrapper_loads_a_base_checkpoint_without_reinitializing(tmp_path) -> None:
    """A base RoBERTa checkpoint is a masked-LM one (``roberta.*`` + ``lm_head.*``)."""
    torch.manual_seed(0)
    base = RobertaForMaskedLM(_roberta_config()).eval()
    with torch.no_grad():
        for p in base.parameters():
            p.normal_(0.0, 0.02)
    base.save_pretrained(tmp_path / "base")
    loaded = RobertaEncoderForSequenceClassification.from_pretrained(tmp_path / "base", num_labels=3).eval()

    got = loaded.state_dict()
    checked = 0
    for key, value in base.state_dict().items():
        if key.startswith("lm_head") or "position_ids" in key:
            continue
        assert key in got, f"encoder tensor not loaded: {key}"
        assert torch.equal(got[key], value), f"encoder tensor differs after load: {key}"
        checked += 1
    assert checked > 0
    assert not [k for k in got if "lm_head" in k or "pooler" in k]


def test_encoder_classification_registry_maps_model_types_to_wrappers() -> None:
    from merge_and_rebase.models import text_lm
    from merge_and_rebase.rebase.text import encoder_classifier

    registry = text_lm._ENCODER_CLASSIFIER_BY_MODEL_TYPE
    assert set(registry) == {"t5", "mt5", "umt5", "longt5", "roberta"}
    assert {registry[t] for t in ("t5", "mt5", "umt5", "longt5")} == {"T5EncoderForSequenceClassification"}
    assert registry["roberta"] == "RobertaEncoderForSequenceClassification"
    assert all(hasattr(encoder_classifier, name) for name in registry.values())
