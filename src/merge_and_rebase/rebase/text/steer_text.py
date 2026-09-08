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
  parameter threshold.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ...utils.linearization import LinearizedModule
from ..base import TensorDict
from ..methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
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


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool ``[B, T, D]`` over real tokens. Text twin of ``steer._pool_block_output``."""
    if hidden.ndim != 3:
        return hidden
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class _TextBlockCapture:
    """Capture masked-mean-pooled per-block activations during a forward pass."""

    def __init__(self, model: nn.Module) -> None:
        self.modules = block_modules(model)
        self.activations: dict[int, torch.Tensor] = {}
        self.attention_mask: torch.Tensor | None = None
        self._handles: list[Any] = []

    def _make_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out):
                self.activations[block_id] = _masked_mean(out, self.attention_mask).detach()

        return hook

    def __enter__(self) -> _TextBlockCapture:
        for block_id, module in enumerate(self.modules):
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
) -> dict[str, Any]:
    """Linear regime: Taylor-linearized ``delta_A`` via ``LinearizedModule``, split per block.

    One ``jvp`` per source block per batch, plus one for the full delta. The
    per-block residuals sum to the full linearized delta by construction; a
    mismatch is reported rather than silently accepted, exactly as in
    ``steer._collect_linear_split``.
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

            with _TextBlockCapture(target) as capture:
                capture.attention_mask = batch_b.get("attention_mask")
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
        mlp_hidden_dim: int = 1024,
        mlp_epochs: int = 100,
        seed: int = 42,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if feature_regime not in {"standard", "linear"}:
            raise ValueError("steer_text feature_regime must be 'standard' or 'linear'")
        if stage_2_strategy not in {"global_ridge", "global_mlp", "block_ridge"}:
            raise ValueError("steer_text stage_2_strategy must be one of: global_ridge, global_mlp, block_ridge")
        if stage_2_strategy == "block_ridge" and feature_regime != "linear":
            raise ValueError("steer_text block_ridge requires feature_regime='linear' (per-block deltas are linear-only).")
        if block_group_strategy not in _BLOCK_GROUP_STRATEGIES:
            raise ValueError(f"steer_text block_group_strategy must be one of: {sorted(_BLOCK_GROUP_STRATEGIES)}")
        if (few_shot is None) == (total_support_examples is None):
            raise ValueError("steer_text requires exactly one of few_shot or total_support_examples")

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        log_prefix = "[steer_text]"

        source_model = llm_source.model
        source_pretrained_model = llm_source_pretrained.model
        target_model = llm_target.model

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

        need_blocks = stage_2_strategy == "block_ridge"

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
            )

        cache_args = {
            "feature_cache_dir": feature_cache_dir,
            "source_tag": source_tag,
            "target_tag": target_tag,
            "task": task,
            "feature_regime": feature_regime,
            "force_recompute_features": force_recompute_features,
            "need_blocks": need_blocks,
            "verbose": verbose,
        }
        train_data = _load_or_compute_split(split="train", compute_fn=lambda: _compute("train"), **cache_args)
        test_data = _load_or_compute_split(split="test", compute_fn=lambda: _compute("test"), **cache_args)

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

            def correction_fn(activations: Mapping[str, Any], *, _model=model) -> torch.Tensor:
                global_act = activations["global"]
                model_device = next(_model.parameters()).device
                with torch.no_grad():
                    out = _model(global_act.double().to(model_device))
                return out.to(dtype=global_act.dtype, device=global_act.device)

        else:  # block_ridge
            delta_a_blocks_train = train_data["delta_A_blocks"].double()
            block_targets = delta_a_blocks_train[selected] @ logit_map.T @ p_b.T

            features_b_full_train = {int(b): v.double() for b, v in train_data["features_B_blocks"].items()}
            num_target_residual = len(features_b_full_train) - 1
            num_source_blocks = int(block_targets.shape[1])
            num_source_residual_blocks = num_source_blocks - 1
            if num_source_residual_blocks < 1 or num_target_residual < 1:
                raise ValueError("steer_text block_ridge: source or target has no residual blocks to target.")

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
                    f"steer_text block_ridge: target has fewer residual blocks ({num_target_residual}) "
                    f"than source ({num_source_residual_blocks}); cannot group."
                )

            grouped_train = dict(grouped_residual)
            grouped_train[num_source_residual_blocks] = output_train

            local_selected = torch.arange(int(selected.numel()))
            local_blocks_train = {b: v[selected] for b, v in grouped_train.items()}
            coefficients = [
                c.to(dev)
                for c in _fit_block_ridge(
                    local_blocks_train,
                    block_targets,
                    selected=local_selected,
                    regularization=ridge_lambda,
                    mode=block_ridge_mode,
                    rho=rho,
                )
            ]

            def correction_fn(
                activations: Mapping[str, Any],
                *,
                _coefficients=coefficients,
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
            "num_source_blocks": num_source_blocks,
            "block_group_size": block_group_size,
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
    need_blocks = prepared.get("stage_2_strategy") == "block_ridge"

    capture = _TextBlockCapture(model) if need_blocks else None
    handles: list[Any] = []

    def _mask_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if capture is not None:
            capture.attention_mask = kwargs.get("attention_mask")
            capture.activations.clear()

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
