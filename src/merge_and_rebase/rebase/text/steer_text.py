"""
``steer`` for HuggingFace text models, registered as ``"steer_text"``.

``rebase/methods/steer.py`` cannot be adapted the way ``theseus``/``bico`` can:
it takes three ``OpenClipClassifier`` objects, builds CLIP zero-shot text heads,
reads ``model.visual``, monkey-patches ``encode_image``, and partitions
parameters with hard-coded ViT/ResNet name tables that *raise* on anything they
do not recognise (``steer.py:394``). What is CLIP-specific there is only the
scaffolding, so this module rebuilds the scaffolding for text and **imports the
math unchanged** -- Stage 1, ridge, the MLP, block ridge, block grouping,
support-set sampling and the on-disk feature cache all come straight from
``steer.py``, so a text run and a vision run are fitting the very same
estimator.

Three deliberate differences from the vision version, each forced by the domain:

1. **No L2 normalization.** ``steer.py`` normalizes the pooled feature because
   CLIP zero-shot classification is defined on the unit sphere. A sequence
   classification head is an affine map on the *raw* pooled feature; normalizing
   would move the model off its own operating point. Features here stay raw.
2. **The head comes from the model, not from a prompt ensemble.** The pooled
   feature is obtained by temporarily replacing the final head ``nn.Linear``
   with ``nn.Identity`` (:func:`_head_as_identity`), so the model's own output
   *becomes* the feature the head consumes. That is exact for every
   ``AutoModelForSequenceClassification`` -- T5 pools at the eos position, Qwen
   and Llama at the last non-pad token, and neither rule has to be reimplemented
   here. ``w_a``/``w_b`` are then simply the two heads' weight matrices.
   Stage 1 uses the weights only, not the biases: a bias difference is a
   per-class constant that a linear map from ``delta_A`` cannot represent, so
   folding it into the residual would only inject noise into the pinv solve.
3. **The block map never raises.** An unrecognised parameter name is assigned to
   the trailing output block and reported once, so an HF architecture that is
   not in the pattern table degrades instead of crashing.

Limits, enforced at runtime:

- requires ``model_kind="sequence_classification"`` -- Stage 1 needs a linear
  head, and prompt-mode evaluation has none;
- ``stage_2_strategy="block_ridge"`` requires ``feature_regime="linear"`` (same
  rule as vision), and the linear regime runs one ``torch.func.jvp`` per block
  per batch through the whole LM. That is affordable for a small T5 and
  punishing for a multi-billion-parameter model; a warning is printed above a
  parameter threshold. ``joint_ridge`` has the same requirement: it needs no
  per-block deltas, but B's per-block activations are only collected there.

Stage-2 options over B's blocks (``block_ridge`` and ``joint_ridge`` only; any
other strategy rejects them rather than ignoring them):

- ``stage_2_strategy="joint_ridge"``: one ridge on all grouped blocks, each divided
  by the root of its support mean squared norm and concatenated, fit to the *total*
  Stage-1 target. ``ridge_lambda`` is then a trace-relative penalty per block. It
  contains ``global_ridge`` on f_B as the special case of zero coefficients on the
  residual blocks.
- ``block_pooling`` (``"mean"`` | ``"unitnorm"`` | ``"rmsnorm"``): how each block's
  tokens are pooled; see ``_TextBlockCapture``. Non-mean poolings are recollected
  from B alone and cached as ``features_B_blocks_pool-<name>.pt`` next to the
  split's other files; the live correction hook pools the same way.
- ``block_source`` (``"residual"`` | ``"attention"``): which tensor each block
  contributes -- its output (the residual stream) or the input of its self-attention
  output projection (the heads' attention outputs before W_O); see
  ``_TextBlockCapture``. The attention source is recollected like a non-mean pooling
  and cached as ``features_B_blocks_src-attention_pool-<name>.pt``.
- ``block_feature_preprocessing`` (``"none"`` | ``"zscore"``): standardize every
  block with support-set statistics and fit an unpenalized intercept; folded into
  the coefficients and a bias vector, so evaluation needs no extra state.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...utils.linearization import LinearizedModule
from ..base import TensorDict
from ..methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _cache_split_dir,
    _few_shot,
    _fit_block_ridge,
    _fit_global_mlp,
    _load_or_compute_split,
    _predict_block_ridge,
    _random_sample,
    _ridge,
    _stage1_projection,
)
from ..registry import register
from .adapters import head_linear
from .encoder_classifier import masked_mean, segment_pooled

# Matched with ``search`` against ``name + "."`` and anchored on a preceding dot
# or the string start, so a task-head wrapper's prefix does not hide the stack:
# T5ForSequenceClassification names its blocks ``transformer.encoder.block.N.*``,
# not ``encoder.block.N.*``. Each entry maps a name to (stack, block index).
_BLOCK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("encoder", re.compile(r"(?:^|\.)encoder\.block\.(\d+)\.")),  # T5 / mT5 / UL2
    ("decoder", re.compile(r"(?:^|\.)decoder\.block\.(\d+)\.")),  # T5 / mT5 / UL2
    ("encoder", re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")),  # BERT / RoBERTa
    ("layers", re.compile(r"(?:^|\.)transformer\.h\.(\d+)\.")),  # GPT-2 / GPT-J
    ("layers", re.compile(r"(?:^|\.)layers\.(\d+)\.")),  # Qwen2/2.5, Llama, Mistral
)

# Stable stack ordering when a model has more than one (T5: encoder then decoder).
_STACK_ORDER: dict[str, int] = {"encoder": 0, "decoder": 1, "layers": 0}

# Embedding-ish parameters, folded into block 0 the way steer folds the ViT stem.
_STEM_FRAGMENTS: tuple[str, ...] = ("shared.", "embed_tokens.", "wte.", "wpe.", "embeddings.")

# jvp-per-block through a model this large is where "linear" stops being sane.
_LINEAR_REGIME_PARAM_WARN = 500_000_000


def block_index_of(name: str) -> tuple[str, int] | None:
    """``(stack, index)`` for a transformer-block parameter, else ``None``."""
    padded = name + "."
    for stack, pattern in _BLOCK_PATTERNS:
        match = pattern.search(padded)
        if match is not None:
            return stack, int(match.group(1))
    return None


def _ordered_block_keys(model: nn.Module) -> list[tuple[str, int]]:
    """Every (stack, index) present, in the canonical flattened order."""
    keys: set[tuple[str, int]] = set()
    for name, _ in model.named_parameters():
        found = block_index_of(name)
        if found is not None:
            keys.add(found)
    return sorted(keys, key=lambda k: (_STACK_ORDER.get(k[0], 99), k[1]))


def text_parameter_blocks(model: nn.Module) -> tuple[tuple[int, ...], int]:
    """Assign each parameter to a block id. Twin of ``steer._parameter_blocks_for_visual``.

    Returns ``(block_ids, num_blocks)`` with ``block_ids`` aligned to
    ``model.named_parameters()`` order and ``num_blocks == num_residual + 1``:
    ids ``0..num_residual-1`` are the transformer blocks and ``num_residual`` is
    the trailing output block (final norms, pooler, head), matching steer's
    numbering so the imported block-ridge code needs no adjustment.
    """
    ordered = _ordered_block_keys(model)
    if not ordered:
        raise ValueError(
            "No transformer blocks found on this model. Recognised layouts: "
            "encoder/decoder.block.N (T5), model.layers.N (Qwen/Llama), "
            "transformer.h.N (GPT-2), encoder.layer.N (BERT)."
        )
    index_of = {key: i for i, key in enumerate(ordered)}
    output_block_id = len(ordered)

    block_ids: list[int] = []
    unknown: list[str] = []
    for name, _ in model.named_parameters():
        found = block_index_of(name)
        if found is not None:
            block_ids.append(index_of[found])
        elif any(fragment in name for fragment in _STEM_FRAGMENTS):
            block_ids.append(0)
        else:
            unknown.append(name)
            block_ids.append(output_block_id)
    if unknown:
        print(
            f"[steer_text] {len(unknown)} parameter(s) matched no block pattern and were assigned "
            f"to the output block (e.g. {unknown[:3]})."
        )
    return tuple(block_ids), output_block_id + 1


def _block_module_name(name: str) -> tuple[str, int] | None:
    """``(stack, index)`` when ``name`` is *exactly* a block module's path, else ``None``."""
    padded = name + "."
    for stack, pattern in _BLOCK_PATTERNS:
        match = pattern.search(padded)
        # Ending exactly at the appended dot means the block index is the *last*
        # path segment, i.e. this module is the block itself, not something inside it.
        if match is not None and match.end() == len(padded):
            return stack, int(match.group(1))
    return None


def block_modules(model: nn.Module) -> list[nn.Module]:
    """The transformer-block modules, in the same order as ``text_parameter_blocks``."""
    found: dict[tuple[str, int], nn.Module] = {}
    for name, module in model.named_modules():
        key = _block_module_name(name)
        if key is not None and key not in found:
            found[key] = module

    ordered = _ordered_block_keys(model)
    missing = [key for key in ordered if key not in found]
    if missing:
        raise ValueError(f"Could not resolve block modules for {missing}.")
    return [found[key] for key in ordered]


def num_residual_blocks(model: nn.Module) -> int:
    return len(_ordered_block_keys(model))


# Where each stack keeps the norm applied to its last block's output before anything
# reads it: T5Stack.final_layer_norm, Llama/Qwen's model.norm, GPT-2's transformer.ln_f.
_FINAL_NORM_ATTRS: tuple[str, ...] = ("final_layer_norm", "norm", "ln_f")


def block_final_norms(model: nn.Module) -> list[nn.Module]:
    """Each block's own stack's final norm, in the same order as ``block_modules``.

    Resolved per block rather than once per model because T5ForSequenceClassification
    has two stacks with two different final norms: an encoder block must be normalized
    by the encoder's, a decoder block by the decoder's.
    """
    found: dict[tuple[str, int], nn.Module] = {}
    for name, _module in model.named_modules():
        key = _block_module_name(name)
        if key is None or key in found:
            continue
        # "transformer.encoder.block.3" -> "transformer.encoder"; "model.layers.3" -> "model".
        parent_path = name.rsplit(".", 2)[0] if name.count(".") >= 2 else ""
        parent = model.get_submodule(parent_path) if parent_path else model
        norm = next((getattr(parent, a) for a in _FINAL_NORM_ATTRS if isinstance(getattr(parent, a, None), nn.Module)), None)
        if norm is None:
            raise ValueError(f"No final norm ({'/'.join(_FINAL_NORM_ATTRS)}) found on '{parent_path}' for block '{name}'.")
        found[key] = norm
    return [found[key] for key in _ordered_block_keys(model)]


# --------------------------------------------------------------------------
# Pooled-feature extraction
# --------------------------------------------------------------------------


@contextmanager
def _head_as_identity(model: nn.Module):
    """Swap the final head ``nn.Linear`` for ``nn.Identity`` for the duration of the block.

    The model's ``logits`` then *are* the pooled feature the head consumes, with
    the architecture's own pooling rule applied (T5's eos position, Qwen's last
    non-pad token). Unlike a forward hook this survives ``torch.func`` transforms,
    which is what the linear regime needs.
    """
    name, head = head_linear(model)
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, attr, nn.Identity())
    try:
        yield head
    finally:
        setattr(parent, attr, head)


def _pooled_features(model: nn.Module, batch: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Pooled head-input feature ``[B, D]`` for one batch. Assumes the head is already Identity."""
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device) if "attention_mask" in batch else None,
    )
    return out.logits


# Mean-pool ``[B, T, D]`` over real tokens; text twin of ``steer._pool_block_output``.
# Defined in encoder_classifier.py rather than here so that T5EncoderForSequenceClassification's
# pooled "CLS" feature and the per-block activations captured below are pooled by the *same*
# function, not by two copies that could drift apart.
_masked_mean = masked_mean


BLOCK_POOLINGS: tuple[str, ...] = ("mean", "unitnorm", "rmsnorm")
BLOCK_SOURCES: tuple[str, ...] = ("residual", "attention")

# Path, inside one block, of the self-attention output projection whose *input* is the
# attention output: T5Block, then Llama/Qwen decoder layers.
_ATTENTION_OUT_PATHS: tuple[str, ...] = ("layer.0.SelfAttention.o", "self_attn.o_proj")


def block_attention_out_projections(model: nn.Module) -> list[nn.Module]:
    """Each block's self-attention output projection, in the same order as ``block_modules``."""
    found = []
    for block in block_modules(model):
        for path in _ATTENTION_OUT_PATHS:
            try:
                found.append(block.get_submodule(path))
                break
            except AttributeError:
                continue
        else:
            raise ValueError(
                f"No self-attention output projection ({', '.join(_ATTENTION_OUT_PATHS)}) in block "
                f"{type(block).__name__}; block_source='attention' does not support this architecture yet."
            )
    return found


class _TextBlockCapture:
    """Capture pooled per-block activations during a forward pass.

    ``source`` decides which tensor each block contributes:

    - ``"residual"``: the block's output, i.e. the residual stream after the block;
    - ``"attention"``: the *input* of the block's self-attention output projection
      (T5 ``SelfAttention.o``), i.e. the heads' attention outputs before W_O. Each
      token there is a mix ``sum_j a_tj v_j`` of value vectors, so its mean over tokens
      weights the values by how much attention each token receives -- information a
      per-token linear map of the residual stream cannot supply, and unlike the
      cumulative residual stream it is not near-identical from block to block.

    ``pooling`` decides what each token becomes before the masked mean over real tokens:

    - ``"mean"``: nothing (for the residual source, what the feature cache holds);
    - ``"unitnorm"``: every token divided by its own L2 norm;
    - ``"rmsnorm"``: every token through its stack's final norm (``block_final_norms``),
      i.e. pooled the way the model pools its own output feature. Residual source only.

    T5's unnormalized residual stream gives a few tokens very large norms, and a
    plain mean is dominated by them; normalizing each token first gives every token
    the same weight in the average. Attention outputs have no such tokens.

    ``segment_pooling=True`` replaces the single masked mean with
    :func:`~merge_and_rebase.rebase.text.encoder_classifier.segment_pooled`
    (premise mean, hypothesis mean, global mean concatenated), for a pair-encoded
    row split at its two EOS tokens -- mutually exclusive with ``pooling != "mean"``
    (there is no unitnorm/rmsnorm-before-segment-split yet). Callers set
    ``capture.input_ids`` per batch, next to ``capture.attention_mask``.
    """

    def __init__(
        self, model: nn.Module, pooling: str = "mean", source: str = "residual", segment_pooling: bool = False
    ) -> None:
        if pooling not in BLOCK_POOLINGS:
            raise ValueError(f"block pooling must be one of {BLOCK_POOLINGS}, got {pooling!r}")
        if source not in BLOCK_SOURCES:
            raise ValueError(f"block source must be one of {BLOCK_SOURCES}, got {source!r}")
        if source == "attention" and pooling == "rmsnorm":
            raise ValueError("rmsnorm pooling applies the residual stream's final norm; it is undefined for attention outputs")
        if segment_pooling and pooling != "mean":
            raise ValueError("segment_pooling is defined for pooling='mean' only (no unitnorm/rmsnorm composition yet)")
        self.source = source
        self.modules = block_modules(model) if source == "residual" else block_attention_out_projections(model)
        self.pooling = pooling
        self.norms = block_final_norms(model) if pooling == "rmsnorm" else None
        self.segment_pooling = segment_pooling
        self.eos_token_id: int | None = None
        if segment_pooling:
            eos_token_id = getattr(model.config, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("segment_pooling requires model.config.eos_token_id to be set.")
            self.eos_token_id = int(eos_token_id)
        self.activations: dict[int, torch.Tensor] = {}
        self.attention_mask: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self._handles: list[Any] = []

    def _pool(self, block_id: int, out: torch.Tensor) -> torch.Tensor:
        if out.ndim == 3 and self.segment_pooling:
            return segment_pooled(out, self.attention_mask, self.input_ids, self.eos_token_id)
        if out.ndim == 3 and self.pooling == "unitnorm":
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        elif out.ndim == 3 and self.pooling == "rmsnorm":
            out = self.norms[block_id](out)
        return _masked_mean(out, self.attention_mask)

    def _make_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out):
                self.activations[block_id] = self._pool(block_id, out).detach()

        return hook

    def _make_pre_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                self.activations[block_id] = self._pool(block_id, inputs[0]).detach()

        return hook

    def __enter__(self) -> _TextBlockCapture:
        for block_id, module in enumerate(self.modules):
            if self.source == "attention":
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(block_id)))
            else:
                self._handles.append(module.register_forward_hook(self._make_hook(block_id)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _iter_paired(source_loader: Any, target_loader: Any) -> Iterable[tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]]:
    """Walk both loaders in lockstep.

    Both are built ``shuffle=False`` over datasets tokenized from the *same*
    example list, so batch ``i`` holds the same examples on both sides -- the
    per-example pairing Stage 1 depends on (``steer._aligned_loader_pair`` makes
    the same guarantee for images).
    """
    if len(source_loader.dataset) != len(target_loader.dataset):
        raise ValueError(
            "steer_text requires source and target datasets of equal length for pairing, "
            f"got {len(source_loader.dataset)} vs {len(target_loader.dataset)}."
        )
    return zip(source_loader, target_loader, strict=True)


@torch.no_grad()
def _collect_standard_split(
    *,
    source_finetuned: nn.Module,
    source_pretrained: nn.Module,
    target: nn.Module,
    source_loader: Any,
    target_loader: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Standard regime: plain forward passes, ``delta_A = f_A_ft(x) - f_A_pre(x)``."""
    features_a: list[torch.Tensor] = []
    delta_a: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with _head_as_identity(source_finetuned), _head_as_identity(source_pretrained), _head_as_identity(target):
        for batch_a, batch_b in _iter_paired(source_loader, target_loader):
            f_a_ft = _pooled_features(source_finetuned, batch_a, device)
            f_a_pre = _pooled_features(source_pretrained, batch_a, device)
            features_a.append(f_a_ft.cpu())
            delta_a.append((f_a_ft - f_a_pre).cpu())
            features_b.append(_pooled_features(target, batch_b, device).cpu())
            labels.append(batch_a["labels"].cpu())

    return {
        "features_A": torch.cat(features_a, dim=0),
        "delta_A": torch.cat(delta_a, dim=0),
        "features_B": torch.cat(features_b, dim=0),
        "y_A": torch.cat(labels, dim=0),
    }


def _collect_linear_split(
    *,
    source_pretrained: nn.Module,
    source_finetuned: nn.Module,
    target: nn.Module,
    source_loader: Any,
    target_loader: Any,
    device: torch.device,
    target_block_pooling: str = "global",
) -> dict[str, Any]:
    """Linear regime: Taylor-linearized ``delta_A`` via ``LinearizedModule``, split per block.

    One ``jvp`` per source block per batch, plus one for the full delta. The
    per-block residuals sum to the full linearized delta by construction; a
    mismatch is reported rather than silently accepted, exactly as in
    ``steer._collect_linear_split``.

    ``target_block_pooling="segments"`` pools B's *residual* blocks with
    :func:`_TextBlockCapture`'s segment-split pooling instead of a single
    global mean (see ``SteerTextRebase.prepare``'s validation for the
    restrictions this requires); the output block (``out_b`` below) is always
    the model's own pooled feature, unaffected either way.
    """
    num_target_residual = num_residual_blocks(target)

    features_a: list[torch.Tensor] = []
    delta_a: list[torch.Tensor] = []
    delta_a_blocks: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    features_b_blocks: dict[int, list[torch.Tensor]] = {b: [] for b in range(num_target_residual + 1)}
    labels: list[torch.Tensor] = []

    with _head_as_identity(source_pretrained), _head_as_identity(source_finetuned), _head_as_identity(target):
        # Computed *inside* the head swap: with the head replaced by Identity its
        # parameters leave named_parameters(), and block_ids must stay aligned
        # with param_names for the per-block masking zip below. The head is not
        # something we want to linearize through anyway -- eval injects a
        # per-task head of its own.
        block_ids, num_a_blocks = text_parameter_blocks(source_pretrained)
        param_names = [name for name, _ in source_pretrained.named_parameters()]
        linmod = LinearizedModule.from_module(
            source_pretrained, device=device, copy_module=False, param_names=param_names
        )
        theta0 = dict(zip(linmod.param_names, linmod.theta0, strict=True))
        finetuned_params = {name: p.detach().to(device) for name, p in source_finetuned.named_parameters()}

        for batch_a, batch_b in _iter_paired(source_loader, target_loader):
            call_kwargs = {
                "input_ids": batch_a["input_ids"].to(device),
                "attention_mask": batch_a["attention_mask"].to(device) if "attention_mask" in batch_a else None,
            }
            with torch.no_grad():
                f0 = _pooled_features(source_pretrained, batch_a, device)

            full_out = linmod.forward(
                current_params=finetuned_params, kwargs=call_kwargs, output_transform=lambda out: out.logits
            )
            delta_full = (full_out - f0).detach()

            block_residuals = torch.zeros(
                f0.shape[0], num_a_blocks, delta_full.shape[-1], dtype=delta_full.dtype
            )
            for block_id in range(num_a_blocks):
                masked_params = {
                    name: (finetuned_params[name] if bid == block_id else theta0[name])
                    for name, bid in zip(param_names, block_ids, strict=True)
                }
                block_out = linmod.forward(
                    current_params=masked_params, kwargs=call_kwargs, output_transform=lambda out: out.logits
                )
                block_residuals[:, block_id, :] = (block_out - f0).detach().cpu()

            reconstructed = block_residuals.sum(dim=1)
            if not torch.allclose(reconstructed, delta_full.cpu(), rtol=1e-3, atol=1e-4):
                print(
                    "[steer_text] warning: per-block delta decomposition does not sum to the full "
                    f"linearized delta (max abs diff={float((reconstructed - delta_full.cpu()).abs().max()):.6g})."
                )

            features_a.append(f0.cpu())
            delta_a.append(delta_full.cpu())
            delta_a_blocks.append(block_residuals)
            labels.append(batch_a["labels"].cpu())

            with _TextBlockCapture(target, segment_pooling=(target_block_pooling == "segments")) as capture:
                capture.attention_mask = batch_b.get("attention_mask")
                if target_block_pooling == "segments":
                    capture.input_ids = batch_b["input_ids"]
                with torch.no_grad():
                    out_b = _pooled_features(target, batch_b, device)
            for b in range(num_target_residual):
                features_b_blocks[b].append(capture.activations[b].cpu())
            features_b_blocks[num_target_residual].append(out_b.detach().cpu())
            features_b.append(out_b.detach().cpu())

    return {
        "features_A": torch.cat(features_a, dim=0),
        "delta_A": torch.cat(delta_a, dim=0),
        "delta_A_blocks": torch.cat(delta_a_blocks, dim=0),
        "features_B": torch.cat(features_b, dim=0),
        "features_B_blocks": {b: torch.cat(chunks, dim=0) for b, chunks in features_b_blocks.items()},
        "y_A": torch.cat(labels, dim=0),
    }


@torch.no_grad()
def _collect_target_blocks(
    *,
    target: nn.Module,
    target_loader: Any,
    device: torch.device,
    pooling: str,
    source: str = "residual",
    segment_pooling: bool = False,
) -> dict[int, torch.Tensor]:
    """B's per-block features for ``source``/``pooling``, from a plain forward of B alone.

    Only B's side changes with the source or pooling rule, so this never re-runs A's
    jvps: the rows line up with the cached split because ``target_loader`` is the same
    ``shuffle=False`` loader ``_collect_linear_split`` walked. ``segment_pooling=True``
    (only meaningful for ``source="attention"`` here -- the ``source="residual"`` case is
    handled directly in ``_collect_linear_split`` instead, see ``prepare()``) additionally
    sets ``capture.input_ids`` per batch.
    """
    with _head_as_identity(target), _TextBlockCapture(
        target, pooling=pooling, source=source, segment_pooling=segment_pooling
    ) as capture:
        num_blocks = len(capture.modules)
        chunks: dict[int, list[torch.Tensor]] = {b: [] for b in range(num_blocks)}
        for batch in target_loader:
            capture.attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            if segment_pooling:
                capture.input_ids = batch["input_ids"].to(device)
            capture.activations.clear()
            _pooled_features(target, batch, device)
            for b in range(num_blocks):
                chunks[b].append(capture.activations[b].cpu())
    return {b: torch.cat(v, dim=0) for b, v in chunks.items()}


def _block_cache_name(source: str, pooling: str, segment_pooling: bool = False) -> str:
    # The residual name predates block_source and is kept so existing caches stay valid.
    suffix = "_segpool" if segment_pooling else ""
    if source == "residual":
        return f"features_B_blocks_pool-{pooling}{suffix}.pt"
    return f"features_B_blocks_src-{source}_pool-{pooling}{suffix}.pt"


def _load_or_compute_pooled_blocks(
    *,
    cache_dir: Path,
    pooling: str,
    force_recompute: bool,
    compute_fn: Callable[[], dict[int, torch.Tensor]],
    verbose: bool,
    source: str = "residual",
    segment_pooling: bool = False,
) -> dict[int, torch.Tensor]:
    """B's block features under a non-default source/pooling, cached beside the split's other files.

    A separate file rather than a separate regime directory: the A-side tensors in the
    split do not depend on B's block features, and recomputing them costs one jvp per
    source block per batch. Jobs are expected to have their own ``feature_cache_dir``;
    the write is atomic only so that a killed job never leaves a partial file behind.
    """
    path = cache_dir / _block_cache_name(source, pooling, segment_pooling)
    label = f"'{source}/{pooling}'{' segment-pooled' if segment_pooling else ''} block features"
    if path.exists() and not force_recompute:
        if verbose:
            print(f"[steer_text] using cached {label} at {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    if verbose:
        print(f"[steer_text] computing {label} for {path}")
    blocks = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(blocks, tmp)
    os.replace(tmp, path)
    return blocks


def _standardize_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Support-set mean and std per channel; the std is floored at 1% of its median so a
    near-constant channel is not blown up into noise."""
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return mean, std.clamp_min(1e-2 * float(std.median()))


def _accuracy(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    mask_class: Sequence[int] | None = None,
) -> float:
    """Head accuracy on raw (un-normalized) pooled features.

    ``steer._accuracy`` L2-normalizes first because CLIP zero-shot lives on the
    unit sphere; a classification head does not, so normalizing here would score
    the model somewhere it never operates.

    ``mask_class`` restricts the argmax to the head columns this task actually
    uses, mirroring ``TextLM.sequence_classification_accuracy``. Without it a
    two-way task sharing a three-way head could "predict" the class it never has,
    and the diagnostic would not line up with the target_zeroshot baseline.
    """
    logits = features @ weight.T
    if bias is not None:
        logits = logits + bias
    if mask_class is not None:
        index = torch.tensor([int(c) for c in mask_class], dtype=torch.long)
        prediction = index[logits.index_select(dim=1, index=index).argmax(dim=1)]
    else:
        prediction = logits.argmax(dim=1)
    return float(prediction.eq(labels).double().mean())


def _head_tensors(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    _, head = head_linear(model)
    weight = head.weight.detach().to(device="cpu", dtype=torch.float64)
    bias = None if head.bias is None else head.bias.detach().to(device="cpu", dtype=torch.float64)
    return weight, bias


@dataclass(frozen=True)
class SteerTextRebase:
    """Few-shot feature-space steering for HF sequence-classification models.

    ``prepare()`` computes-or-loads cached A/B pooled features, fits Stage 1
    (head-aware logit correction) and the chosen Stage 2 predictor.
    ``transport()`` returns an empty delta -- like ``steer``, this method never
    touches target weights; ``text_rebase.py`` applies the correction at eval
    time through :func:`steer_text_correction_context`.
    """

    name: str = "steer_text"

    def prepare(
        self,
        *,
        llm_source: Any,
        llm_source_pretrained: Any,
        llm_target: Any,
        source_loaders: Any,
        target_loaders: Any,
        task: str,
        mask_class: Sequence[int] | None = None,
        device: str = "cuda",
        feature_regime: str = "standard",
        stage_2_strategy: str = "global_ridge",
        block_group_strategy: str = "concat",
        feature_cache_dir: str = "src/.cache/steer_text_features",
        force_recompute_features: bool = False,
        source_tag: str = "source",
        target_tag: str = "target",
        few_shot: int | None = None,
        total_support_examples: int | None = None,
        stage1_lambda: float = 1.0,
        ridge_lambda: float = 1.0,
        block_ridge_mode: str = "independent",
        rho: float = 0.9,
        block_ridge_lambda_scaling: str = "none",
        block_ridge_target_strategy: str = "reuse_logitmap",
        block_residuals_weighting_strategy: str = "identity",
        block_ridge_blockwise_stage1_lambda: float = 1.0,
        block_pooling: str = "mean",
        block_source: str = "residual",
        block_feature_preprocessing: str = "none",
        target_block_pooling: str = "global",
        mlp_hidden_dim: int = 1024,
        mlp_epochs: int = 100,
        seed: int = 42,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if feature_regime not in {"standard", "linear"}:
            raise ValueError("steer_text feature_regime must be 'standard' or 'linear'")
        if stage_2_strategy not in {"global_ridge", "global_mlp", "block_ridge", "joint_ridge"}:
            raise ValueError(
                "steer_text stage_2_strategy must be one of: global_ridge, global_mlp, block_ridge, joint_ridge"
            )
        if stage_2_strategy == "block_ridge" and feature_regime != "linear":
            raise ValueError("steer_text block_ridge requires feature_regime='linear' (per-block deltas are linear-only).")
        if stage_2_strategy == "joint_ridge" and feature_regime != "linear":
            # joint_ridge needs no per-block deltas, only B's per-block activations -- but
            # those are only collected by the linear-regime pass.
            raise ValueError("steer_text joint_ridge requires feature_regime='linear' (B's blocks are collected there).")
        uses_blocks = stage_2_strategy in {"block_ridge", "joint_ridge"}
        if block_pooling not in BLOCK_POOLINGS:
            raise ValueError(f"steer_text block_pooling must be one of: {', '.join(BLOCK_POOLINGS)}")
        if block_source not in BLOCK_SOURCES:
            raise ValueError(f"steer_text block_source must be one of: {', '.join(BLOCK_SOURCES)}")
        if block_source == "attention" and block_pooling == "rmsnorm":
            raise ValueError("steer_text block_pooling='rmsnorm' is defined for the residual source only")
        if block_feature_preprocessing not in {"none", "zscore"}:
            raise ValueError("steer_text block_feature_preprocessing must be 'none' or 'zscore'")
        if not uses_blocks and (
            block_pooling != "mean" or block_source != "residual" or block_feature_preprocessing != "none"
        ):
            raise ValueError(
                "steer_text block_pooling / block_source / block_feature_preprocessing only apply to block_ridge "
                "and joint_ridge; "
                f"stage_2_strategy={stage_2_strategy!r} would silently ignore them."
            )
        if block_group_strategy not in _BLOCK_GROUP_STRATEGIES:
            raise ValueError(f"steer_text block_group_strategy must be one of: {sorted(_BLOCK_GROUP_STRATEGIES)}")
        if block_ridge_lambda_scaling not in {"none", "trace"}:
            raise ValueError("steer_text block_ridge_lambda_scaling must be 'none' or 'trace'")
        if block_ridge_target_strategy not in {"reuse_logitmap", "blockwise_logitmap", "last_only"}:
            raise ValueError(
                "steer_text block_ridge_target_strategy must be one of: reuse_logitmap, blockwise_logitmap, last_only"
            )
        if block_residuals_weighting_strategy not in {"identity", "mean"}:
            raise ValueError("steer_text block_residuals_weighting_strategy must be 'identity' or 'mean'")
        if target_block_pooling not in {"global", "segments"}:
            raise ValueError("steer_text target_block_pooling must be 'global' or 'segments'")
        if target_block_pooling == "segments":
            if stage_2_strategy != "block_ridge":
                raise ValueError(
                    "steer_text target_block_pooling='segments' requires stage_2_strategy='block_ridge' "
                    f"(got {stage_2_strategy!r}); joint_ridge support is a separate, later change."
                )
            if block_pooling != "mean":
                raise ValueError(
                    "steer_text target_block_pooling='segments' requires block_pooling='mean' "
                    "(no unitnorm/rmsnorm composition with segment pooling yet)."
                )
        if (few_shot is None) == (total_support_examples is None):
            raise ValueError("steer_text requires exactly one of few_shot or total_support_examples")

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        log_prefix = "[steer_text]"

        source_model = llm_source.model
        source_pretrained_model = llm_source_pretrained.model
        target_model = llm_target.model

        if target_block_pooling == "segments":
            # A decoder block sees only the decoder's own (typically single-token) input,
            # never the encoder's premise/hypothesis sequence, so a segment split by the
            # encoder's EOS positions has no meaning on it. This rejects e.g.
            # T5ForSequenceClassification (encoder+decoder); only an encoder-only target
            # (T5EncoderForSequenceClassification) is valid here.
            target_stacks = {stack for stack, _ in _ordered_block_keys(target_model)}
            if "decoder" in target_stacks:
                raise ValueError(
                    "steer_text target_block_pooling='segments' requires an encoder-only target "
                    "(e.g. T5EncoderForSequenceClassification): the target has a 'decoder' stack "
                    f"({sorted(target_stacks)}), whose blocks never see the premise/hypothesis sequence."
                )

        w_a, _ = _head_tensors(source_model)
        w_b, b_b = _head_tensors(target_model)

        n_source_params = sum(p.numel() for p in source_model.parameters())
        if feature_regime == "linear" and n_source_params > _LINEAR_REGIME_PARAM_WARN:
            print(
                f"{log_prefix} WARNING: feature_regime='linear' runs one forward-mode jvp per source block "
                f"per batch through a {n_source_params / 1e9:.2f}B-parameter model. Expect very high memory "
                "and wall-clock cost; 'standard' with global_ridge/global_mlp is the affordable path."
            )

        # The whole method transports A's finetuning delta, so an A whose weights
        # equal the pretrained ones yields delta_A == 0 and silently degenerate
        # results. That happens whenever the finetuned checkpoint failed to load
        # under a non-strict load, so check rather than trust the caller.
        pretrained_params = dict(source_pretrained_model.named_parameters())
        if all(
            torch.equal(v.detach().cpu(), pretrained_params[k].detach().cpu())
            for k, v in source_model.named_parameters()
            if k in pretrained_params
        ):
            raise ValueError(
                f"steer_text: the finetuned source model for task '{task}' is identical to the pretrained "
                "source model, so delta_A is zero. The tuned checkpoint did not load."
            )

        need_blocks = uses_blocks

        def _compute(split: str) -> dict[str, Any]:
            source_loader = getattr(source_loaders, split)
            target_loader = getattr(target_loaders, split)
            if feature_regime == "standard":
                return _collect_standard_split(
                    source_finetuned=source_model,
                    source_pretrained=source_pretrained_model,
                    target=target_model,
                    source_loader=source_loader,
                    target_loader=target_loader,
                    device=dev,
                )
            return _collect_linear_split(
                source_pretrained=source_pretrained_model,
                source_finetuned=source_model,
                target=target_model,
                source_loader=source_loader,
                target_loader=target_loader,
                device=dev,
                # Segment-pool the primary split's *residual* blocks only when block_source
                # itself is "residual" -- otherwise those residual blocks are discarded
                # below in favor of the recomputed block_source="attention" ones anyway, so
                # segment-pooling them here would be pure waste (and would wrongly force
                # this expensive A-side jvp pass to recompute under a decorated cache key,
                # see cache_feature_regime below).
                target_block_pooling=(target_block_pooling if block_source == "residual" else "global"),
            )

        # features_B_blocks' residual entries change width (D -> 3D) under segment pooling
        # of the *residual* source, and the pooling scheme is not otherwise part of the
        # cache key -- decorate the regime segment of the cache path so a "segments" run
        # never reads or clobbers a "global" run's cache (_cache_split_dir only ever uses
        # this string as a literal path component; _compute() above closes over the real
        # feature_regime, so the branch logic is unaffected). block_source="attention"
        # needs no decoration here: its segment-pooled blocks live in their own cache file
        # below, and everything else in this split cache is identical either way.
        cache_feature_regime = (
            f"{feature_regime}__segpool" if target_block_pooling == "segments" and block_source == "residual" else feature_regime
        )
        cache_args = {
            "feature_cache_dir": feature_cache_dir,
            "source_tag": source_tag,
            "target_tag": target_tag,
            "task": task,
            "feature_regime": cache_feature_regime,
            "force_recompute_features": force_recompute_features,
            "need_blocks": need_blocks,
            "verbose": verbose,
        }
        train_data = _load_or_compute_split(split="train", compute_fn=lambda: _compute("train"), **cache_args)
        test_data = _load_or_compute_split(split="test", compute_fn=lambda: _compute("test"), **cache_args)

        recompute_segment_pooling = target_block_pooling == "segments" and block_source != "residual"
        if need_blocks and (block_pooling != "mean" or block_source != "residual"):
            # Swap B's per-block features for the recollected ones; f_B (the output block)
            # and everything on A's side stay as cached.
            for split, data in (("train", train_data), ("test", test_data)):
                pooled = _load_or_compute_pooled_blocks(
                    cache_dir=_cache_split_dir(feature_cache_dir, source_tag, target_tag, task, feature_regime, split),
                    pooling=block_pooling,
                    source=block_source,
                    segment_pooling=recompute_segment_pooling,
                    force_recompute=force_recompute_features,
                    compute_fn=lambda split=split: _collect_target_blocks(
                        target=target_model, target_loader=getattr(target_loaders, split), device=dev,
                        pooling=block_pooling, source=block_source, segment_pooling=recompute_segment_pooling,
                    ),
                    verbose=verbose,
                )
                blocks = dict(data["features_B_blocks"])
                keys = {int(k): k for k in blocks}
                if len(pooled) != len(blocks) - 1:
                    raise ValueError(
                        f"steer_text: {len(pooled)} '{block_source}' blocks but the cached split has "
                        f"{len(blocks) - 1} residual blocks."
                    )
                for b, value in pooled.items():
                    old = blocks[keys[int(b)]]
                    # Rows must line up with the cache; the width may differ for the attention
                    # source (heads * d_head need not equal d_model).
                    bad_rows = value.shape[0] != old.shape[0]
                    bad_width = block_source == "residual" and tuple(value.shape) != tuple(old.shape)
                    if bad_rows or bad_width:
                        raise ValueError(
                            f"steer_text: '{block_source}/{block_pooling}' block {b} has shape {tuple(value.shape)} "
                            f"but the cached split has {tuple(old.shape)}; rerun with force_recompute_features=true."
                        )
                    blocks[keys[int(b)]] = value
                data["features_B_blocks"] = blocks

        f_a = train_data["features_A"].double()
        delta_a = train_data["delta_A"].double()
        f_b = train_data["features_B"].double()
        delta_a_test = test_data["delta_A"].double()
        f_b_test = test_data["features_B"].double()
        test_labels = test_data["y_A"].long()

        # Support-set selection runs on *local* 0..K-1 labels, not the head-space
        # ids stored in y_A: _few_shot walks range(labels.max() + 1) and would
        # raise on the class a two-way task leaves empty in a shared 3-way head.
        local_labels = torch.as_tensor(
            [int(y) for y in source_loaders.local_labels["train"]], dtype=torch.long
        )
        if local_labels.numel() != f_a.shape[0]:
            raise ValueError(
                f"steer_text: {local_labels.numel()} local labels but {f_a.shape[0]} cached train features "
                "for this task. Rerun with force_recompute_features=true after changing max_samples_per_task."
            )
        if few_shot is not None:
            selected = _few_shot(local_labels, int(few_shot), int(seed))
        else:
            selected = _random_sample(local_labels, int(total_support_examples), int(seed))

        logit_map = _stage1_projection(
            f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=selected, regularization=stage1_lambda
        )
        p_b = torch.linalg.pinv(w_b)
        train_target = delta_a[selected] @ logit_map.T @ p_b.T
        test_target = delta_a_test @ logit_map.T @ p_b.T
        stage1_test_acc = _accuracy(f_b_test + test_target, w_b, b_b, test_labels, mask_class=mask_class)
        if verbose:
            print(
                f"{log_prefix} prepare: stage1 oracle test acc = {stage1_test_acc:.4f} "
                "(uses A's delta at test time; diagnostic only)"
            )

        num_source_blocks: int | None = None
        block_group_size: float | None = None

        if stage_2_strategy == "global_ridge":
            coefficient = _ridge(f_b[selected], train_target, ridge_lambda).to(dev)
            stage2_state = {"kind": "global_ridge", "coefficient": coefficient}

            def correction_fn(activations: Mapping[str, Any], *, _coef=coefficient) -> torch.Tensor:
                global_act = activations["global"]
                out = global_act.double().to(_coef.device) @ _coef
                return out.to(dtype=global_act.dtype, device=global_act.device)

        elif stage_2_strategy == "global_mlp":
            model = _fit_global_mlp(
                f_b[selected],
                train_target,
                seed=10_000 + int(seed) * 100 + int(selected.numel()),
                epochs=int(mlp_epochs),
                hidden_dim=int(mlp_hidden_dim),
            ).to(dev)
            stage2_state = {
                "kind": "global_mlp",
                "state_dict": model.state_dict(),
                "input_dim": int(f_b.shape[1]),
                "hidden_dim": int(mlp_hidden_dim),
                "output_dim": int(train_target.shape[1]),
                "epochs": int(mlp_epochs),
            }

            def correction_fn(activations: Mapping[str, Any], *, _model=model) -> torch.Tensor:
                global_act = activations["global"]
                model_device = next(_model.parameters()).device
                with torch.no_grad():
                    out = _model(global_act.double().to(model_device))
                return out.to(dtype=global_act.dtype, device=global_act.device)

        else:  # block_ridge / joint_ridge: both read B's grouped per-block activations
            delta_a_blocks_train = train_data["delta_A_blocks"].double()
            features_b_full_train = {int(b): v.double() for b, v in train_data["features_B_blocks"].items()}
            num_target_residual = len(features_b_full_train) - 1
            num_source_blocks = int(delta_a_blocks_train.shape[1])
            num_source_residual_blocks = num_source_blocks - 1
            if num_source_residual_blocks < 1 or num_target_residual < 1:
                raise ValueError(f"steer_text {stage_2_strategy}: source or target has no residual blocks to target.")

            residual_train = {b: features_b_full_train[b] for b in range(num_target_residual)}
            output_train = features_b_full_train[num_target_residual]

            if num_target_residual == num_source_residual_blocks:
                grouped_residual = residual_train
                block_group_size = 1.0
            elif num_target_residual > num_source_residual_blocks:
                grouped_residual = _BLOCK_GROUP_STRATEGIES[block_group_strategy](
                    residual_train, num_source_residual_blocks
                )
                block_group_size = num_target_residual / num_source_residual_blocks
            else:
                raise ValueError(
                    f"steer_text {stage_2_strategy}: target has fewer residual blocks ({num_target_residual}) "
                    f"than source ({num_source_residual_blocks}); cannot group."
                )

            grouped_train = dict(grouped_residual)
            grouped_train[num_source_residual_blocks] = output_train

            local_selected = torch.arange(int(selected.numel()))
            local_blocks_train = {b: v[selected] for b, v in grouped_train.items()}
            num_blocks = len(local_blocks_train)

            # Optional per-block standardization, with support-set statistics only. It is
            # folded into the coefficients and one bias vector below, so the live path
            # never needs the statistics: ((x - mu) / sd) @ C == x @ (C / sd) - (mu / sd) @ C.
            if block_feature_preprocessing == "zscore":
                block_stats = {b: _standardize_stats(x) for b, x in local_blocks_train.items()}
                fit_blocks = {b: (x - block_stats[b][0]) / block_stats[b][1] for b, x in local_blocks_train.items()}
            else:
                block_stats = None
                fit_blocks = local_blocks_train
            # Centered features get an unpenalized intercept: fit centered targets, add the mean back.
            fit_intercept = block_stats is not None

            if stage_2_strategy == "block_ridge":
                if block_ridge_target_strategy == "blockwise_logitmap":
                    # A separate Stage-1 map per block, fitted on that block's delta alone.
                    # Named apart from ``logit_map``: that one is the full-delta map the
                    # artifacts report, and must not end up holding the last block's.
                    block_targets_list = []
                    for delta_a_block in delta_a_blocks_train.unbind(dim=1):
                        block_logit_map = _stage1_projection(
                            f_a=f_a,
                            delta_a=delta_a_block,
                            w_a=w_a,
                            f_b=f_b,
                            w_b=w_b,
                            selected=selected,
                            regularization=block_ridge_blockwise_stage1_lambda,
                        )
                        block_targets_list.append(delta_a_block[selected] @ block_logit_map.T @ p_b.T)
                    block_targets = torch.stack(block_targets_list, dim=1)
                elif block_ridge_target_strategy == "reuse_logitmap":
                    # The full-delta Stage-1 map applied to each block's delta.
                    block_targets = delta_a_blocks_train[selected] @ logit_map.T @ p_b.T
                else:  # last_only
                    # Every block regresses the full Stage-1 train target.
                    block_targets = train_target.unsqueeze(1).expand(-1, delta_a_blocks_train.shape[1], -1)

                block_weight = 1.0 / num_blocks if block_residuals_weighting_strategy == "mean" else 1.0
                weights = [block_weight] * num_blocks
                # With centered features every block's support prediction has zero mean, so
                # the smoothed-residual carry stays zero-mean too and each block's intercept
                # is just the mean of its own target slice.
                intercepts = block_targets.mean(dim=0) if fit_intercept else None  # [num_blocks, d]
                raw = _fit_block_ridge(
                    fit_blocks,
                    block_targets - intercepts.unsqueeze(0) if fit_intercept else block_targets,
                    selected=local_selected,
                    regularization=ridge_lambda,
                    mode=block_ridge_mode,
                    rho=rho,
                    regularization_scaling=block_ridge_lambda_scaling,
                )
                joint_scales = None
            else:  # joint_ridge
                # One ridge on all blocks against the *total* Stage-1 target. Each block is
                # divided by the root of its mean squared row norm, tr(X_b X_b^T)/n on the
                # support -- so ridge_lambda is a per-block trace-relative penalty, the same
                # units as block_ridge_lambda_scaling="trace" -- then concatenated. It uses no
                # per-block targets, so block_ridge_target_strategy, the weighting strategy,
                # block_ridge_mode, rho and block_ridge_lambda_scaling do not apply.
                weights = [1.0] * num_blocks
                joint_scales = [float(fit_blocks[b].square().sum(dim=1).mean().sqrt()) for b in range(num_blocks)]
                z = torch.cat([fit_blocks[b] / joint_scales[b] for b in range(num_blocks)], dim=1)
                target_mean = train_target.mean(dim=0) if fit_intercept else None
                joint = _ridge(z, train_target - target_mean if fit_intercept else train_target, ridge_lambda)
                del z
                sizes = [int(fit_blocks[b].shape[1]) for b in range(num_blocks)]
                raw = [c / joint_scales[b] for b, c in enumerate(torch.split(joint, sizes, dim=0))]
                intercepts = None

            # Fold weights, standardization and intercepts into (coefficients, bias).
            coefficients = []
            bias: torch.Tensor | None = None
            if fit_intercept:
                bias = target_mean.clone() if stage_2_strategy == "joint_ridge" else torch.zeros_like(raw[0][0])
            for b, (c, w) in enumerate(zip(raw, weights, strict=True)):
                if block_stats is not None:
                    mu, sd = block_stats[b]
                    bias = bias - w * ((mu / sd) @ c)
                    c = c / sd.unsqueeze(1)
                    if intercepts is not None:
                        bias = bias + w * intercepts[b]
                coefficients.append((c * w).to(dev))
            if bias is not None:
                bias = bias.to(dev)

            stage2_state = {
                "kind": stage_2_strategy,
                "coefficients": coefficients,
                "bias": bias,
                "num_target_residual": int(num_target_residual),
                "num_source_residual_blocks": int(num_source_residual_blocks),
                "block_group_strategy": str(block_group_strategy),
                "block_pooling": str(block_pooling),
                "block_source": str(block_source),
                "block_feature_preprocessing": str(block_feature_preprocessing),
                "target_block_pooling": str(target_block_pooling),
                "weights": weights,
            }
            if stage_2_strategy == "block_ridge":
                stage2_state.update(
                    {
                        "block_ridge_mode": str(block_ridge_mode),
                        "rho": float(rho),
                        "lambda_scaling": str(block_ridge_lambda_scaling),
                        "target_strategy": str(block_ridge_target_strategy),
                        "blockwise_stage1_lambda": float(block_ridge_blockwise_stage1_lambda),
                        "weighting_strategy": str(block_residuals_weighting_strategy),
                    }
                )
            else:
                stage2_state["joint_block_scales"] = joint_scales

            def correction_fn(
                activations: Mapping[str, Any],
                *,
                _coefficients=coefficients,
                _bias=bias,
                _num_target_residual=num_target_residual,
                _num_source_residual_blocks=num_source_residual_blocks,
                _strategy=block_group_strategy,
            ) -> torch.Tensor:
                global_act = activations["global"]
                coef_device = _coefficients[0].device
                residual = {
                    b: activations["blocks"][b].double().to(coef_device) for b in range(_num_target_residual)
                }
                if _num_target_residual != _num_source_residual_blocks:
                    residual = _BLOCK_GROUP_STRATEGIES[_strategy](residual, _num_source_residual_blocks)
                blocks = dict(residual)
                blocks[_num_source_residual_blocks] = (
                    activations["blocks"][_num_target_residual].double().to(coef_device)
                )
                out = _predict_block_ridge(_coefficients, blocks)
                if _bias is not None:
                    out = out + _bias
                return out.to(dtype=global_act.dtype, device=global_act.device)

        # Stage 2 accuracy in cached-tensor space, the number directly comparable
        # to steer4rebase's run.py: predicted from the cached test features, no
        # alpha, no live forward pass. The "rebased" column text_rebase.py prints
        # goes through the live eval path instead, so a gap between the two
        # isolates a Stage 1/2 problem from a live-integration problem.
        cached_test_activations: dict[str, Any] = {"global": f_b_test}
        if need_blocks:
            cached_test_activations["blocks"] = {int(b): v.double() for b, v in test_data["features_B_blocks"].items()}
        stage2_test_acc = _accuracy(
            f_b_test + correction_fn(cached_test_activations), w_b, b_b, test_labels, mask_class=mask_class
        )
        stage0_test_acc = _accuracy(f_b_test, w_b, b_b, test_labels, mask_class=mask_class)
        if verbose:
            print(
                f"{log_prefix} prepare: cached-space B zero-shot test acc = {stage0_test_acc:.4f} "
                "(compare with the target_zeroshot baseline below -- they should match)"
            )
            print(
                f"{log_prefix} prepare: stage2 ({stage_2_strategy}) cached test acc = {stage2_test_acc:.4f} "
                "(predicted from B's cached features, no alpha)"
            )

        return {
            "correction_fn": correction_fn,
            "stage_2_strategy": stage_2_strategy,
            "feature_regime": feature_regime,
            "block_pooling": block_pooling,
            "block_source": block_source,
            "target_block_pooling": target_block_pooling,
            "num_source_blocks": num_source_blocks,
            "block_group_size": block_group_size,
            # The fitted transforms themselves, so a run can be inspected or
            # replayed without refitting. ``correction_fn`` closes over the same
            # Stage 2 object, so these are references, not copies -- consumers
            # must not mutate them. Everything here is small except
            # ``stage2_state``: global_ridge's coefficient is ``[d_b, d_b]``.
            "artifacts": {
                "stage1_logit_map": logit_map,
                "stage1_pinv_w_b": p_b,
                "stage1_lambda": float(stage1_lambda),
                "ridge_lambda": float(ridge_lambda),
                "w_a": w_a,
                "w_b": w_b,
                "b_b": b_b,
                "selected": selected,
                "stage2_state": stage2_state,
            },
            "diagnostics": {
                "stage0_test_acc": stage0_test_acc,
                "stage1_test_acc": stage1_test_acc,
                "stage2_test_acc": stage2_test_acc,
            },
        }

    def apply_correction(
        self, prepared: Mapping[str, Any], *, activations: Mapping[str, Any], alpha: float = 1.0
    ) -> torch.Tensor:
        return prepared["correction_fn"](activations) * float(alpha)

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        prepared: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        del source_base, target_base, delta, strict, prepared, kwargs
        return {}


@contextmanager
def steer_text_correction_context(llm: Any, prepared: Mapping[str, Any], *, alpha: float = 1.0):
    """Add ``alpha * correction`` to the pooled feature for the duration of the block.

    A forward pre-hook on the head ``nn.Linear`` rewrites its input, which is the
    exact analogue of ``steer_correction_context``'s ``encode_image`` patch --
    the correction lands on the pooled feature, just before classification, and
    the head's own weights and bias apply unchanged on top of it.
    """
    model = getattr(llm, "model", llm)
    _, head = head_linear(model)
    correction_fn = prepared["correction_fn"]
    need_blocks = prepared.get("stage_2_strategy") in {"block_ridge", "joint_ridge"}

    # Pool the live blocks exactly as the fit's features were pooled.
    segment_pooling = prepared.get("target_block_pooling", "global") == "segments"
    capture = (
        _TextBlockCapture(
            model,
            pooling=prepared.get("block_pooling", "mean"),
            source=prepared.get("block_source", "residual"),
            segment_pooling=segment_pooling,
        )
        if need_blocks
        else None
    )
    handles: list[Any] = []

    def _mask_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if capture is not None:
            capture.attention_mask = kwargs.get("attention_mask")
            capture.activations.clear()
            if capture.segment_pooling:
                input_ids = kwargs.get("input_ids")
                if input_ids is None:
                    raise RuntimeError(
                        "steer_text: target_block_pooling='segments' needs 'input_ids' as a forward "
                        "keyword argument to locate the premise/hypothesis boundary, but the model was "
                        "called without it (positional call, or a caller that omits it)."
                    )
                capture.input_ids = input_ids

    def _head_pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        feature = inputs[0]
        activations: dict[str, Any] = {"global": feature}
        if capture is not None:
            num_residual = len(capture.modules)
            missing = [b for b in range(num_residual) if b not in capture.activations]
            if missing:
                raise RuntimeError(
                    f"steer_text: block activations {missing} were not captured before the head ran."
                )
            blocks = dict(capture.activations)
            blocks[num_residual] = feature
            activations["blocks"] = blocks
        return (feature + float(alpha) * correction_fn(activations),) + tuple(inputs[1:])

    if capture is not None:
        capture.__enter__()
        handles.append(model.register_forward_pre_hook(_mask_hook, with_kwargs=True))
    handles.append(head.register_forward_pre_hook(_head_pre_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        if capture is not None:
            capture.__exit__(None, None, None)


register(SteerTextRebase())
