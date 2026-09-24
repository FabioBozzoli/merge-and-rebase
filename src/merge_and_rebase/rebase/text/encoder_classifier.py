"""
A T5 *encoder* with a linear classification head, for ``steer_text``.

``TextLM.build(model_kind="sequence_classification")`` gives T5 through
``T5ForSequenceClassification``, which builds the decoder, pools at the
decoder's eos position, and puts ``T5ClassificationHead``
(``dropout -> dense -> tanh -> dropout -> out_proj``) between the model and the
feature. Three things follow, each of which the surrounding code currently works
around rather than avoids:

1. the pooled feature Stage 1/2 are fit on is a *decoder* state;
2. ``steer_text._BLOCK_PATTERNS`` matches ``encoder.block.N`` **and**
   ``decoder.block.N``, so ``text_parameter_blocks`` reports 24 blocks for
   t5-base and 48 for t5-large, and ``block_ridge`` groups 48 -> 24 across a
   boundary that has no meaning;
3. ``dense`` is in no base checkpoint -- ``AutoModelForSequenceClassification``
   redraws it per process -- so "the pooled feature" is really
   ``tanh(random_rotation(decoder_eos_state))``. That is the entire reason
   ``adapters.neutralize_intermediate_head_layers`` exists.

This module removes all three by never constructing the decoder and by putting
exactly one ``nn.Linear`` between the pooled feature and the logits.

Two naming decisions carry the compatibility, and neither is arbitrary:

``self.transformer``
    ``T5PreTrainedModel.base_model_prefix == "transformer"``, so naming the
    backbone this way is what lets ``from_pretrained`` load a *base* T5
    checkpoint (whose keys are ``shared.weight``, ``encoder.block.N...``, with
    no prefix) into this wrapper: HF detects that the model expects the prefix
    and the checkpoint lacks it, and adds it. ``T5ForSequenceClassification``
    relies on the same mechanism. It also means parameters are named
    ``transformer.encoder.block.N.*``, which is exactly the prefixed spelling
    ``steer_text._BLOCK_PATTERNS`` was written to tolerate (see its comment).

``classification_head.out_proj``
    Not ``score``. ``adapters._HEAD_ROOTS`` accepts either, but
    ``llm_merge._task_head_tensor_for_param`` gates its ``head_class_ids``
    scatter -- the path that places a task's classes into a shared head space --
    on names ending in ``classification_head.out_proj.weight``/``.bias``.
    Naming the head ``score`` would silently lose that path and then raise on
    any width mismatch. As a bonus the param names match what
    ``T5ForSequenceClassification`` produces, so ``heads.pt`` files built for
    the decoder-based path drop straight in.

Because the head container holds exactly one ``nn.Linear``,
``adapters.head_intermediate_linears`` returns ``[]`` and
``neutralize_intermediate_head_layers`` becomes a genuine no-op.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5PreTrainedModel


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool ``[B, T, D]`` over real tokens.

    This is the single definition of the pooling rule used anywhere in the text
    stack: ``steer_text._masked_mean`` is an alias to it, so the pooled "CLS"
    feature this model returns and the per-block activations
    ``steer_text._TextBlockCapture`` records are pooled identically by
    construction rather than by two functions happening to agree -- except for
    ``_TextBlockCapture``'s residual blocks under ``segment_pooling=True``,
    which call :func:`segment_pooled` instead (itself built from three calls
    to this function); the output block (always the plain pooled feature) is
    unaffected either way.

    Tensors that are not ``[B, T, D]`` pass through unchanged, which is what
    lets the block-capture hook feed it whatever a block returns without
    inspecting shapes first.
    """
    if hidden.ndim != 3:
        return hidden
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def segment_masks_from_eos(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two ``[B, T]`` boolean masks splitting a pair-encoded row at its EOS tokens.

    T5's pair encoding (``build_nli_tokenized_loader``) lays out
    ``premise <eos> hypothesis <eos>`` with no other separator. Segment 1 is
    every token strictly before the first EOS, segment 2 every token strictly
    between the first and second EOS; both exclude the EOS tokens themselves.

    Padding sits after the second EOS under right-padding and is excluded by
    construction (its cumulative EOS count is 2), but every row is also
    intersected with ``attention_mask`` as a defensive measure against any
    other padding convention.

    Raises ``ValueError`` if any row has fewer than two EOS tokens, rather
    than degrading to a whole-row mask: on this repo's NLI tasks that means
    truncation or an upstream tokenization change removed the
    premise/hypothesis boundary, and a silent fallback would produce
    plausible-looking but wrong numbers.
    """
    is_eos = input_ids == int(eos_token_id)
    counts = is_eos.sum(dim=1)
    if bool((counts < 2).any()):
        bad = int((counts < 2).sum())
        raise ValueError(
            f"segment_masks_from_eos: {bad} row(s) have fewer than 2 EOS tokens "
            f"(eos_token_id={eos_token_id}); cannot locate the premise/hypothesis boundary. "
            "This happens with a single-string premise_hypothesis_template (one EOS per row) "
            "or if truncation removed the second segment entirely."
        )
    cumcount = is_eos.cumsum(dim=1)
    real = attention_mask.bool()
    segment1 = (cumcount == 0) & real
    segment2 = (cumcount == 1) & ~is_eos & real
    return segment1, segment2


def segment_pooled(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """``[premise_mean ; hypothesis_mean ; global_mean]``, each via :func:`masked_mean`.

    A strict superset of the plain global mean: the global third is not
    linearly recoverable from the first two alone (their implicit weights are
    the two segments' lengths, which vary per example), so a ridge regressing
    on this vector can always zero the first two blocks' coefficients and
    recover today's global-only behaviour.
    """
    segment1, segment2 = segment_masks_from_eos(input_ids, attention_mask, eos_token_id)
    return torch.cat(
        [masked_mean(hidden, segment1), masked_mean(hidden, segment2), masked_mean(hidden, attention_mask)],
        dim=-1,
    )


class T5EncoderClassificationHead(nn.Module):
    """A single ``out_proj`` linear, wrapped so the qualified name matches T5's.

    The wrapper exists purely for the name: ``classification_head.out_proj`` is
    what ``llm_merge``'s head-injection scatter matches on. Keeping exactly one
    ``nn.Linear`` inside is what makes
    ``adapters.head_intermediate_linears`` return ``[]``, i.e. there is no
    untrained layer between the model's representation and the readout.
    """

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out_proj(features)


class T5EncoderForSequenceClassification(T5PreTrainedModel):
    """T5's encoder stack plus a linear head. The decoder is never constructed.

    Presents the same surface every text consumer in this repo already expects:
    ``model(input_ids=..., attention_mask=...).logits``, a
    ``head_linear``-discoverable final ``nn.Linear``, and ``encoder.block.N``
    parameter names for block discovery. ``steer_text`` therefore needs no
    changes at all -- in particular ``_head_as_identity`` still works, because
    ``forward`` computes logits *only* by calling the head, so replacing the
    head with ``nn.Identity`` makes ``logits`` be the pooled feature itself.
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    # A base T5 checkpoint carries the decoder and the LM head; neither has a
    # home here. T5EncoderModel declares the same for `decoder`, but that
    # declaration belongs to the submodule and is not consulted for *this*
    # class's load, so it is restated.
    _keys_to_ignore_on_load_unexpected = [r"decoder", r"lm_head"]
    # Deliberately not declaring _tied_weights_keys: `shared.weight` and
    # `encoder.embed_tokens.weight` alias one storage, but that tie is declared
    # on T5EncoderModel, and `_get_tied_weight_keys` walks named_children and
    # prefixes what it finds -- so it already resolves to
    # `transformer.encoder.embed_tokens.weight`. Restating it here would be
    # redundant, and stating it *wrongly* would make save_pretrained drop the
    # wrong tensor.

    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.classification_head = T5EncoderClassificationHead(config.d_model, config.num_labels)
        # A T5Config arrives with is_encoder_decoder=True because it describes a
        # seq2seq architecture -- but this model has no decoder, so the flag is
        # simply false of it. Correcting it here is what makes
        # ``TextLM._is_encoder_decoder`` report the truth, which in turn makes
        # prompt-mode evaluation refuse this kind instead of trying to score
        # decoder continuations that cannot exist. T5EncoderModel already
        # deep-copies the config and clears this flag for its own stack, so
        # nothing downstream of the encoder depends on it being True.
        self.config.is_encoder_decoder = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        self.transformer.set_input_embeddings(new_embeddings)

    def get_encoder(self) -> nn.Module:
        return self.transformer.encoder

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        del return_dict, kwargs  # accepted for call-site compatibility, unused
        encoder_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled = masked_mean(encoder_outputs.last_hidden_state, attention_mask)
        logits = self.classification_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.view(-1).long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
