"""
Steer rebase method: few-shot feature-space steering (ported from steer4rebase).

Unlike the other rebase methods in this package, ``steer`` does not transport a
weight-space task-vector delta. It learns, from a tiny few-shot support set, a
correction to the target model B's pooled visual feature that makes B behave
like a fine-tuned source model A, without ever touching B's weights. Only
``global_ridge`` is a pure linear map on B's final feature (foldable into a
weight delta); ``global_mlp`` (nonlinear) and ``block_ridge`` (built from B's
*intermediate* per-block activations) cannot be expressed as a weight delta
under any approximation. So this method does not implement the ``transport()``
weight-delta contract meaningfully -- ``vision_rebase.py`` special-cases
``method_name == "steer"`` (mirroring its existing transfusion_mode/
theseus_like_method special-casing) and applies the correction by wrapping the
target classifier's forward pass instead of adding a state-dict delta.

Two phases, mirroring stage1.py/stage2.py/data.py/linearize.py from
steer4rebase (https://github.com internal repo at rebasin_linear/steer4rebase):

  - ``prepare()``: compute-or-load-from-disk-cache A's and B's features (Stage 0,
    replaces steer4rebase's external bea_utils.py feature collection), fit
    Stage 1 (head-aware logit-space correction) and the selected Stage 2
    predictor (global_ridge / global_mlp / block_ridge).
  - ``apply_correction()`` / the eval-time hook wrapper in vision_rebase.py:
    apply the fitted Stage 2 predictor to B's real forward-pass activations.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...utils.linearization import LinearizedModule
from ..base import TensorDict
from ..registry import register


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize the final pooled visual feature. Mirrors steer4rebase's
    LinearizedModelV2.forward_base_with_activations, which explicitly
    L2-normalizes the model's output before it is ever used in Stage 1/2 --
    the whole method is fit and evaluated in unit-norm CLIP embedding space,
    matching the standard zero-shot classification convention (normalize,
    then dot with the text head). Only the final global feature is
    normalized this way; intermediate per-block activations (used only as
    block_ridge regression inputs) are left raw, matching the original.
    """
    return F.normalize(x, dim=-1)

# --------------------------------------------------------------------------
# A. Stage 1 / Stage 2 math, ported near-verbatim from stage1.py / stage2.py
# --------------------------------------------------------------------------


def _stage1_projection(
    *,
    f_a: torch.Tensor,
    delta_a: torch.Tensor,
    w_a: torch.Tensor,
    f_b: torch.Tensor,
    w_b: torch.Tensor,
    selected: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Fit the logit correction map with a penalty on the transported residual."""
    if regularization <= 0:
        raise ValueError("steer stage1 regularization must be positive")
    residual = (f_a[selected] + delta_a[selected]) @ w_a.T - f_b[selected] @ w_b.T
    logit_map = torch.linalg.pinv(delta_a[selected]) @ residual
    return (logit_map / (1.0 + regularization)).T


def _stage1_target_corrections(
    *,
    f_a: torch.Tensor,
    delta_a: torch.Tensor,
    w_a: torch.Tensor,
    f_b: torch.Tensor,
    w_b: torch.Tensor,
    delta_a_test: torch.Tensor,
    selected: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return train and test target corrections in B's feature space."""
    logit_map = _stage1_projection(
        f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=selected, regularization=regularization
    )
    p_b = torch.linalg.pinv(w_b)
    train_target = delta_a[selected] @ logit_map.T @ p_b.T
    test_target = delta_a_test @ logit_map.T @ p_b.T
    return train_target, test_target


def _ridge(features: torch.Tensor, targets: torch.Tensor, regularization: float) -> torch.Tensor:
    if regularization <= 0:
        raise ValueError("steer ridge regularization must be positive")
    n, dimension = features.shape
    eye = torch.eye(n if n <= dimension else dimension, dtype=features.dtype)
    if n <= dimension:
        return features.T @ torch.linalg.solve(features @ features.T + regularization * eye, targets)
    return torch.linalg.solve(features.T @ features + regularization * eye, features.T @ targets)


class _ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_global_mlp(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    *,
    seed: int,
    epochs: int = 100,
    hidden_dim: int = 1024,
) -> _ResidualMLP:
    torch.manual_seed(seed)
    model = _ResidualMLP(train_features.shape[1], hidden_dim, train_target.shape[1]).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(train_features) - train_target).square().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _fit_block_ridge(
    blocks_train: Mapping[int, torch.Tensor],
    train_targets: torch.Tensor,
    *,
    selected: torch.Tensor,
    regularization: float,
    mode: str,
    rho: float = 0.9,
) -> list[torch.Tensor]:
    """
    Fit per-block ridge coefficients, chained with a smoothed-residual carry.

    ``train_targets`` has shape [n_selected, num_blocks, d_B]. Fitting is
    self-contained (depends only on train-side data), so the resulting
    coefficients can be applied to *any* new block activations at predict
    time -- this is the split-fit/predict factoring of steer4rebase's
    stage2.block_ridge (which fits and predicts in a single call).
    """
    if mode not in {"independent", "smoothed_residual"}:
        raise ValueError(f"Unknown steer block_ridge mode: {mode}")
    if mode == "smoothed_residual" and not 0.0 <= rho <= 1.0:
        raise ValueError("steer block_ridge rho must be in [0, 1]")

    num_blocks = train_targets.shape[1]
    state = torch.zeros_like(train_targets[:, 0])
    coefficients: list[torch.Tensor] = []
    for block_id in range(num_blocks):
        x_train = blocks_train[block_id][selected]
        local_target = train_targets[:, block_id]
        compensation = rho * state if mode == "smoothed_residual" else torch.zeros_like(state)
        fitted_target = local_target + compensation
        coefficient = _ridge(x_train, fitted_target, regularization)
        coefficients.append(coefficient)
        if mode == "smoothed_residual":
            block_train_prediction = x_train @ coefficient
            state = compensation + local_target - block_train_prediction
    return coefficients


def _predict_block_ridge(coefficients: Sequence[torch.Tensor], block_activations: Mapping[int, torch.Tensor]) -> torch.Tensor:
    prediction = None
    for block_id, coefficient in enumerate(coefficients):
        contribution = block_activations[block_id] @ coefficient
        prediction = contribution if prediction is None else prediction + contribution
    if prediction is None:
        raise ValueError("steer block_ridge has no fitted blocks.")
    return prediction


def _group_blocks_concat(blocks: Mapping[int, torch.Tensor], num_groups: int) -> dict[int, torch.Tensor]:
    """Group B blocks into ``num_groups`` consecutive groups by concatenation (ported from data.py)."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    block_ids = sorted(blocks)
    if not block_ids:
        raise ValueError("blocks dict is empty")
    if len(block_ids) < num_groups:
        raise ValueError(f"Cannot group {len(block_ids)} blocks into {num_groups} groups")
    if len(block_ids) == num_groups:
        return dict(blocks)
    boundaries = [round(i * len(block_ids) / num_groups) for i in range(num_groups + 1)]
    grouped: dict[int, torch.Tensor] = {}
    for g in range(num_groups):
        members = block_ids[boundaries[g] : boundaries[g + 1]]
        grouped[g] = torch.cat([blocks[b] for b in members], dim=1)
    return grouped


def _group_blocks_sum_avg(blocks: Mapping[int, torch.Tensor], num_groups: int) -> dict[int, torch.Tensor]:
    """
    Group B blocks by averaging (dimension-preserving alternative to concat).

    ponytail: group_grid.py's real "sum_avg" fits one ridge per member block and
    averages *predictions*; we average *features* before a single ridge instead
    (cheaper, same block_ridge fit/predict split as concat). Swap for the
    per-block-ridge-then-average version if exact reproduction of that specific
    sweep variant is needed.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    block_ids = sorted(blocks)
    if not block_ids:
        raise ValueError("blocks dict is empty")
    if len(block_ids) < num_groups:
        raise ValueError(f"Cannot group {len(block_ids)} blocks into {num_groups} groups")
    if len(block_ids) == num_groups:
        return dict(blocks)
    boundaries = [round(i * len(block_ids) / num_groups) for i in range(num_groups + 1)]
    grouped: dict[int, torch.Tensor] = {}
    for g in range(num_groups):
        members = block_ids[boundaries[g] : boundaries[g + 1]]
        stacked = torch.stack([blocks[b] for b in members], dim=0)
        grouped[g] = stacked.mean(dim=0)
    return grouped


_BLOCK_GROUP_STRATEGIES: dict[str, Callable[[Mapping[int, torch.Tensor], int], dict[int, torch.Tensor]]] = {
    "concat": _group_blocks_concat,
    "sum_avg": _group_blocks_sum_avg,
}


# --------------------------------------------------------------------------
# Support-set sampling (ported from data.py)
# --------------------------------------------------------------------------


def _few_shot(labels: torch.Tensor, shots: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    classes = int(labels.max().item()) + 1
    selected = []
    for class_id in range(classes):
        indices = torch.where(labels == class_id)[0]
        if len(indices) < shots:
            raise ValueError(f"Class {class_id} has {len(indices)} examples, need {shots}")
        selected.append(indices[torch.randperm(len(indices), generator=generator)[:shots]])
    return torch.cat(selected)


def _random_sample(labels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    total = labels.shape[0]
    return torch.randperm(total, generator=generator)[:n]


# --------------------------------------------------------------------------
# C. ViT / ResNet block-partition helpers (ported from linearize.py)
# --------------------------------------------------------------------------

# Keys here are relative to a CLIP model's `.visual` submodule (no "visual."
# prefix), matching this repo's own convention (see theseus.py's
# `_visual_module`/`_visual_state_dict`), rather than steer4rebase's
# `model.visual.*` (steer4rebase wraps the whole CLIP model as `self.model`).
_VIT_STEM_PARAMETERS = {"class_embedding", "positional_embedding", "conv1.weight", "ln_pre.weight", "ln_pre.bias"}
_VIT_OUTPUT_PARAMETERS = {"proj", "ln_post.weight", "ln_post.bias"}
_VIT_RESBLOCK_PATTERN = re.compile(r"^transformer\.resblocks\.(\d+)\.")

_RESNET_STEM_PARAMETERS = {
    "conv1.weight",
    "conv2.weight",
    "conv3.weight",
    "bn1.weight",
    "bn1.bias",
    "bn2.weight",
    "bn2.bias",
    "bn3.weight",
    "bn3.bias",
}
_RESNET_OUTPUT_PARAMETERS = {"proj"}
_RESNET_ATTNPOOL_PREFIX = "attnpool."
_RESNET_RESBLOCK_PATTERN = re.compile(r"^layer(\d+)\.(\d+)\.")
_RESNET_STAGE_BLOCKS = {1: 3, 2: 4, 3: 6, 4: 3}


def _is_resnet_visual(visual: nn.Module) -> bool:
    return hasattr(visual, "layer1") and not hasattr(visual, "transformer")


def _clip_vit_parameter_blocks(parameter_names: Iterable[str]) -> tuple[tuple[int | None, ...], int]:
    """Assign each visual parameter to a transformer-block id. Returns (block_ids, num_blocks)."""
    parameter_names = list(parameter_names)
    residual_block_ids = {
        int(m.group(1)) for name in parameter_names if (m := _VIT_RESBLOCK_PATTERN.match(name)) is not None
    }
    if not residual_block_ids:
        raise ValueError("No CLIP visual transformer residual blocks were found.")
    last_residual_block = max(residual_block_ids)
    expected = set(range(last_residual_block + 1))
    if residual_block_ids != expected:
        missing = sorted(expected - residual_block_ids)
        raise ValueError(f"CLIP visual transformer block IDs are not contiguous; missing {missing}.")

    output_block_id = last_residual_block + 1
    block_ids: list[int | None] = []
    for name in parameter_names:
        if name in _VIT_STEM_PARAMETERS:
            block_ids.append(0)
        elif name in _VIT_OUTPUT_PARAMETERS:
            block_ids.append(output_block_id)
        elif (m := _VIT_RESBLOCK_PATTERN.match(name)) is not None:
            block_ids.append(int(m.group(1)))
        else:
            raise ValueError(f"Cannot assign CLIP ViT parameter {name!r} to a block.")
    return tuple(block_ids), output_block_id + 1


def _resnet_block_id_for(stage: int, index: int) -> int:
    return sum(_RESNET_STAGE_BLOCKS[s] for s in range(1, stage)) + index


def _resnet_block_id_to_stage_index(block_id: int) -> tuple[int, int]:
    cumulative = 0
    for stage in range(1, 5):
        if block_id < cumulative + _RESNET_STAGE_BLOCKS[stage]:
            return stage, block_id - cumulative
        cumulative += _RESNET_STAGE_BLOCKS[stage]
    raise ValueError(f"ResNet block id {block_id} out of range.")


def _clip_resnet_parameter_blocks(parameter_names: Iterable[str]) -> tuple[tuple[int | None, ...], int]:
    """Assign each visual parameter to a bottleneck-block id for OpenCLIP's ModifiedResNet."""
    parameter_names = list(parameter_names)
    residual_blocks = {
        (int(m.group(1)), int(m.group(2)))
        for name in parameter_names
        if (m := _RESNET_RESBLOCK_PATTERN.match(name)) is not None
    }
    if not residual_blocks:
        raise ValueError("No CLIP visual ResNet bottleneck blocks were found.")
    max_stage = max(stage for stage, _ in residual_blocks)
    last_residual_id = sum(_RESNET_STAGE_BLOCKS[stage] for stage in range(1, max_stage + 1)) - 1
    output_block_id = last_residual_id + 1

    block_ids: list[int | None] = []
    for name in parameter_names:
        if name in _RESNET_STEM_PARAMETERS:
            block_ids.append(0)
        elif name in _RESNET_OUTPUT_PARAMETERS or name.startswith(_RESNET_ATTNPOOL_PREFIX):
            block_ids.append(output_block_id)
        elif (m := _RESNET_RESBLOCK_PATTERN.match(name)) is not None:
            block_ids.append(_resnet_block_id_for(int(m.group(1)), int(m.group(2))))
        else:
            raise ValueError(f"Cannot assign CLIP ResNet parameter {name!r} to a block.")
    return tuple(block_ids), output_block_id + 1


def _parameter_blocks_for_visual(visual: nn.Module) -> tuple[tuple[int | None, ...], int]:
    names = [name for name, _ in visual.named_parameters()]
    if _is_resnet_visual(visual):
        return _clip_resnet_parameter_blocks(names)
    return _clip_vit_parameter_blocks(names)


def _num_residual_blocks(visual: nn.Module) -> int:
    """Number of resblocks/bottlenecks (excludes the trailing output/projection block)."""
    if _is_resnet_visual(visual):
        return sum(len(getattr(visual, f"layer{stage}")) for stage in range(1, 5))
    return len(visual.transformer.resblocks)


def _get_block_module(visual: nn.Module, block_id: int) -> nn.Module:
    if _is_resnet_visual(visual):
        stage, index = _resnet_block_id_to_stage_index(block_id)
        return getattr(visual, f"layer{stage}")[index]
    return visual.transformer.resblocks[block_id]


def _pool_block_output(output: Any) -> torch.Tensor:
    out = output[0] if isinstance(output, (tuple, list)) else output
    if out.ndim == 3:
        return out.mean(dim=1)
    if out.ndim == 4:
        return out.mean(dim=(2, 3))
    return out


class _BlockActivationCapture:
    """Forward-hook manager capturing pooled per-block activations of a visual encoder.

    Mirrors LinearizedModelV2.create_activation_hooks' pooling rules (mean-pool
    3D ViT token output, GAP 4D ResNet conv maps). The final pooled/projected
    output is *not* captured here -- callers already have it as the plain
    return value of ``encode_image``.
    """

    def __init__(self, visual: nn.Module, block_ids: Sequence[int]) -> None:
        self.visual = visual
        self.block_ids = list(block_ids)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            self.activations[block_id] = _pool_block_output(output).detach()

        return hook

    def __enter__(self) -> "_BlockActivationCapture":
        for block_id in self.block_ids:
            module = _get_block_module(self.visual, block_id)
            self._handles.append(module.register_forward_hook(self._make_hook(block_id)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------
# D. Feature computation + disk cache
# --------------------------------------------------------------------------


@torch.no_grad()
def _iter_batches(loader: Any, *, device: torch.device) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for x, y in loader:
        yield x.to(device), y.to(device)


def _aligned_loader_pair(source_loader: Any, target_loader: Any) -> tuple[Any, Any]:
    """
    Rebuild A's and B's split loaders with ``shuffle=False`` over their
    underlying ``.dataset``, so batch ``i`` of A and batch ``i`` of B are
    guaranteed to be the same example.

    steer's Stage 1/2 fit requires per-example correspondence between A and
    B (delta_A[i] must be paired with B's activations for the *same* image
    i) -- exactly what steer4rebase's own feature collection mirrors with
    "single joint traversal ... to keep pairing". The train-split loaders
    handed in by vision_rebase.py are built with shuffle=True and no fixed
    generator (see data/vision_loaders.py), so two independently-constructed
    DataLoaders over the same dataset shuffle into *different, unrelated*
    orders -- silently pairing unrelated images and making the whole fit
    fit noise. Rebuilding both loaders here with shuffle=False bypasses that
    regardless of how the original loaders were configured.
    """
    source_ds = source_loader.dataset
    target_ds = target_loader.dataset
    if len(source_ds) != len(target_ds):
        raise ValueError(
            f"steer requires source and target datasets to have the same length for pairing, "
            f"got {len(source_ds)} vs {len(target_ds)}."
        )
    batch_size = source_loader.batch_size
    aligned_source = DataLoader(source_ds, batch_size=batch_size, shuffle=False)
    aligned_target = DataLoader(target_ds, batch_size=batch_size, shuffle=False)
    return aligned_source, aligned_target


@torch.no_grad()
def _collect_standard_split(
    *,
    clf_source_finetuned_visual: nn.Module,
    clf_source_pretrained_visual: nn.Module,
    target_visual: nn.Module,
    source_loader: Any,
    target_loader: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Standard/nonlinear regime: plain forward passes, delta_A = f_A_ft(x) - f_A_pre(x)."""
    features_a: list[torch.Tensor] = []
    delta_a: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for (x_a, y_a), (x_b, _y_b) in zip(_iter_batches(source_loader, device=device), _iter_batches(target_loader, device=device), strict=True):
        f_a_ft = _l2_normalize(clf_source_finetuned_visual(x_a))
        f_a_pre = _l2_normalize(clf_source_pretrained_visual(x_a))
        features_a.append(f_a_ft.cpu())
        delta_a.append((f_a_ft - f_a_pre).cpu())
        features_b.append(_l2_normalize(target_visual(x_b)).cpu())
        labels.append(y_a.cpu())
    return {
        "features_A": torch.cat(features_a, dim=0),
        "delta_A": torch.cat(delta_a, dim=0),
        "features_B": torch.cat(features_b, dim=0),
        "y_A": torch.cat(labels, dim=0),
    }


def _collect_linear_split(
    *,
    source_pretrained_visual: nn.Module,
    source_finetuned_params: Mapping[str, torch.Tensor],
    target_visual: nn.Module,
    source_loader: Any,
    target_loader: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Linear regime: Taylor-linearized delta_A via LinearizedModule (torch.func.jvp),
    decomposed per visual block. Reuses this repo's existing linearization
    primitive (utils/linearization.py) instead of porting steer4rebase's
    deprecated-functorch LinearizedModelV2.
    """
    block_ids, num_a_blocks = _parameter_blocks_for_visual(source_pretrained_visual)
    param_names = [name for name, _ in source_pretrained_visual.named_parameters()]
    linmod = LinearizedModule.from_module(source_pretrained_visual, device=device, copy_module=False, param_names=param_names)
    theta0 = dict(zip(linmod.param_names, linmod.theta0, strict=True))

    num_target_residual = _num_residual_blocks(target_visual)

    features_a: list[torch.Tensor] = []
    delta_a: list[torch.Tensor] = []
    delta_a_blocks: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    # Same numbering as the source side (_parameter_blocks_for_visual): keys
    # 0..num_target_residual-1 are the residual blocks, key num_target_residual
    # is the final pooled/output feature -- matching steer4rebase's own
    # create_activation_hooks convention (a hook on model.visual itself at
    # index num_blocks - 1) and its saved features_B_blocks.pt layout, so a
    # cache produced by the original pipeline is directly interchangeable.
    features_b_blocks: dict[int, list[torch.Tensor]] = {b: [] for b in range(num_target_residual + 1)}
    labels: list[torch.Tensor] = []

    for (x_a, y_a), (x_b, _y_b) in zip(_iter_batches(source_loader, device=device), _iter_batches(target_loader, device=device), strict=True):
        with torch.no_grad():
            f0 = _l2_normalize(source_pretrained_visual(x_a))

        full_out = linmod.forward(current_params=source_finetuned_params, args=(x_a,), output_transform=_l2_normalize)
        delta_full = (full_out - f0).detach()

        block_residuals = torch.zeros(x_a.shape[0], num_a_blocks, delta_full.shape[-1], dtype=delta_full.dtype)
        for block_id in range(num_a_blocks):
            masked_params = {
                name: (source_finetuned_params[name] if bid == block_id else theta0[name])
                for name, bid in zip(param_names, block_ids, strict=True)
            }
            block_out = linmod.forward(current_params=masked_params, args=(x_a,), output_transform=_l2_normalize)
            block_residuals[:, block_id, :] = (block_out - f0).detach()

        reconstructed = block_residuals.sum(dim=1)
        if not torch.allclose(reconstructed, delta_full.cpu(), rtol=1e-3, atol=1e-4):
            import logging

            logging.getLogger(__name__).warning(
                "steer: per-block delta decomposition does not sum to the full linearized delta "
                "(max abs diff=%.6g). Continuing with per-block values as computed.",
                (reconstructed - delta_full.cpu()).abs().max().item(),
            )

        features_a.append(f0.cpu())
        delta_a.append(delta_full.cpu())
        delta_a_blocks.append(block_residuals)
        labels.append(y_a.cpu())

        with _BlockActivationCapture(target_visual, list(range(num_target_residual))) as capture:
            with torch.no_grad():
                out_b = _l2_normalize(target_visual(x_b))
        for block_id in range(num_target_residual):
            features_b_blocks[block_id].append(capture.activations[block_id].cpu())
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


def _cache_split_dir(feature_cache_dir: str, source_tag: str, target_tag: str, task: str, regime: str, split: str) -> Path:
    return Path(feature_cache_dir) / f"{source_tag}_to_{target_tag}" / task / regime / split


def _load_cached_head(path: Path) -> torch.Tensor | None:
    """Load a head_A.pt/head_B.pt as steer4rebase's data.py::load_head does.

    steer4rebase saves classification heads as either a raw tensor or a dict
    with a "weight" key (``bea_utils._head_state``); unwrap the same way.
    """
    if not path.exists():
        return None
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if isinstance(value, Mapping):
        value = value.get("weight")
    if not isinstance(value, torch.Tensor):
        return None
    return value.double()


def _load_cached_split(cache_dir: Path, *, need_blocks: bool) -> dict[str, torch.Tensor] | None:
    required = ["features_A", "delta_A", "features_B", "y_A"]
    if need_blocks:
        required += ["delta_A_blocks", "features_B_blocks"]
    if not all((cache_dir / f"{name}.pt").exists() for name in required):
        return None
    try:
        return {name: torch.load(cache_dir / f"{name}.pt", map_location="cpu", weights_only=True) for name in required}
    except Exception:
        return None


def _save_split_cache(cache_dir: Path, data: Mapping[str, torch.Tensor]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, value in data.items():
        torch.save(value, cache_dir / f"{name}.pt")


def _load_or_compute_split(
    *,
    feature_cache_dir: str,
    source_tag: str,
    target_tag: str,
    task: str,
    feature_regime: str,
    split: str,
    force_recompute_features: bool,
    need_blocks: bool,
    compute_fn: Callable[[], dict[str, Any]],
    verbose: bool,
) -> dict[str, Any]:
    cache_dir = _cache_split_dir(feature_cache_dir, source_tag, target_tag, task, feature_regime, split)
    if not force_recompute_features:
        cached = _load_cached_split(cache_dir, need_blocks=need_blocks)
        if cached is not None:
            if verbose:
                print(f"[steer] using cached features at {cache_dir}")
            return cached
    if verbose:
        print(f"[steer] computing features for {cache_dir}")
    data = compute_fn()
    _save_split_cache(cache_dir, data)
    if verbose:
        print(f"[steer] computed and cached features to {cache_dir}")
    return data


# --------------------------------------------------------------------------
# E / SteerRebase
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SteerRebase:
    """
    Few-shot feature-space steering (ported from steer4rebase).

    ``prepare()`` computes-or-loads cached A/B features, fits Stage 1 (head-aware
    logit correction) and the selected Stage 2 predictor. ``apply_correction()``
    applies the fitted predictor to new activations at eval time. ``transport()``
    exists only for protocol-shape consistency with other rebase methods -- it
    returns an empty delta, since steer never touches target weights; see
    vision_rebase.py's ``steer_mode`` dispatch for how evaluation is wired.
    """

    name: str = "steer"

    def prepare(
        self,
        *,
        clf_source: Any,
        clf_source_pretrained: Any,
        clf_target: Any,
        source_loaders: Any,
        target_loaders: Any,
        classnames: list[str],
        task: str,
        source_build_cfg_task: Any,
        build_cfg_task: Any,
        device: str = "cuda",
        feature_regime: str = "standard",
        stage_2_strategy: str = "global_ridge",
        block_group_strategy: str = "concat",
        feature_cache_dir: str = "src/.cache/steer_features",
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
            raise ValueError("steer feature_regime must be 'standard' or 'linear'")
        if stage_2_strategy not in {"global_ridge", "global_mlp", "block_ridge"}:
            raise ValueError("steer stage_2_strategy must be one of: global_ridge, global_mlp, block_ridge")
        if stage_2_strategy == "block_ridge" and feature_regime != "linear":
            raise ValueError("steer block_ridge requires feature_regime='linear' (per-block deltas are linear-only).")
        if block_group_strategy not in _BLOCK_GROUP_STRATEGIES:
            raise ValueError(f"steer block_group_strategy must be one of: {sorted(_BLOCK_GROUP_STRATEGIES)}")
        if (few_shot is None) == (total_support_examples is None):
            raise ValueError("steer requires exactly one of few_shot or total_support_examples")

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        log_prefix = "[steer]"

        # steer4rebase's data.py loads head_A.pt/head_B.pt from the train split
        # directory (a dict with a "weight" key, or a raw tensor) rather than
        # recomputing them -- honor that cache convention here too, so heads
        # placed on disk (e.g. from the original steering.py pipeline) are
        # actually used instead of being silently replaced by a live
        # zero-shot recomputation that may use a different prompt ensemble.
        head_cache_dir = _cache_split_dir(feature_cache_dir, source_tag, target_tag, task, feature_regime, "train")
        cached_w_a = None if force_recompute_features else _load_cached_head(head_cache_dir / "head_A.pt")
        cached_w_b = None if force_recompute_features else _load_cached_head(head_cache_dir / "head_B.pt")
        if cached_w_a is not None and cached_w_b is not None:
            w_a, w_b = cached_w_a, cached_w_b
            if verbose:
                print(f"{log_prefix} using cached heads at {head_cache_dir}")
        else:
            clf_source.build_zeroshot_text_features(classnames, source_build_cfg_task, cache_dir="src/.cache/zs_cache")
            clf_target.build_zeroshot_text_features(classnames, build_cfg_task, cache_dir="src/.cache/zs_cache")
            w_a = clf_source._zs_text_features.detach().to(device="cpu", dtype=torch.float64)
            w_b = clf_target._zs_text_features.detach().to(device="cpu", dtype=torch.float64)
            head_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(w_a, head_cache_dir / "head_A.pt")
            torch.save(w_b, head_cache_dir / "head_B.pt")

        source_finetuned_visual = clf_source.model.visual.to(dev).eval()
        source_pretrained_visual = clf_source_pretrained.model.visual.to(dev).eval()
        target_visual = clf_target.model.visual.to(dev).eval()
        need_blocks = stage_2_strategy == "block_ridge"

        def _compute(split: str) -> dict[str, Any]:
            source_loader, target_loader = _aligned_loader_pair(getattr(source_loaders, split), getattr(target_loaders, split))
            if feature_regime == "standard":
                return _collect_standard_split(
                    clf_source_finetuned_visual=source_finetuned_visual,
                    clf_source_pretrained_visual=source_pretrained_visual,
                    target_visual=target_visual,
                    source_loader=source_loader,
                    target_loader=target_loader,
                    device=dev,
                )
            source_finetuned_params = {name: p.detach().to(dev) for name, p in source_finetuned_visual.named_parameters()}
            return _collect_linear_split(
                source_pretrained_visual=source_pretrained_visual,
                source_finetuned_params=source_finetuned_params,
                target_visual=target_visual,
                source_loader=source_loader,
                target_loader=target_loader,
                device=dev,
            )

        train_data = _load_or_compute_split(
            feature_cache_dir=feature_cache_dir,
            source_tag=source_tag,
            target_tag=target_tag,
            task=task,
            feature_regime=feature_regime,
            split="train",
            force_recompute_features=force_recompute_features,
            need_blocks=need_blocks,
            compute_fn=lambda: _compute("train"),
            verbose=verbose,
        )
        test_data = _load_or_compute_split(
            feature_cache_dir=feature_cache_dir,
            source_tag=source_tag,
            target_tag=target_tag,
            task=task,
            feature_regime=feature_regime,
            split="test",
            force_recompute_features=force_recompute_features,
            need_blocks=need_blocks,
            compute_fn=lambda: _compute("test"),
            verbose=verbose,
        )

        f_a = train_data["features_A"].double()
        delta_a = train_data["delta_A"].double()
        f_b = train_data["features_B"].double()
        train_labels = train_data["y_A"].long()
        delta_a_test = test_data["delta_A"].double()
        f_b_test = test_data["features_B"].double()
        test_labels = test_data["y_A"].long()

        if few_shot is not None:
            selected = _few_shot(train_labels, few_shot, seed)
        else:
            selected = _random_sample(train_labels, int(total_support_examples), seed)

        logit_map = _stage1_projection(f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=selected, regularization=stage1_lambda)
        p_b = torch.linalg.pinv(w_b)
        train_target = delta_a[selected] @ logit_map.T @ p_b.T
        test_target = delta_a_test @ logit_map.T @ p_b.T
        stage1_test_acc = _accuracy(f_b_test + test_target, w_b, test_labels)
        if verbose:
            print(f"{log_prefix} prepare: stage1 oracle test acc = {stage1_test_acc:.4f} (uses A's delta at test time; diagnostic only)")

        num_source_blocks = None
        block_group_size = None
        if stage_2_strategy == "global_ridge":
            coefficient = _ridge(f_b[selected], train_target, ridge_lambda).to(dev)

            def correction_fn(activations: Mapping[str, torch.Tensor], *, _coef=coefficient) -> torch.Tensor:
                global_act = activations["global"].double().to(_coef.device)
                return (global_act @ _coef).to(dtype=activations["global"].dtype, device=activations["global"].device)

        elif stage_2_strategy == "global_mlp":
            model = _fit_global_mlp(
                f_b[selected], train_target, seed=10_000 + seed * 100 + int(selected.numel()), epochs=mlp_epochs, hidden_dim=mlp_hidden_dim
            ).to(dev)

            def correction_fn(activations: Mapping[str, torch.Tensor], *, _model=model) -> torch.Tensor:
                model_device = next(_model.parameters()).device
                with torch.no_grad():
                    out = _model(activations["global"].double().to(model_device))
                return out.to(dtype=activations["global"].dtype, device=activations["global"].device)

        else:  # block_ridge
            delta_a_blocks_train = train_data["delta_A_blocks"].double()
            block_targets = delta_a_blocks_train[selected] @ logit_map.T @ p_b.T  # [n_sel, num_A_blocks, D_B]

            # Same numbering on both sides: key num_blocks-1 is the final
            # pooled/output feature, which has a *different* width than the
            # uniform residual blocks (e.g. transformer hidden dim vs. joint
            # CLIP embed dim) -- so it must never be concatenated/averaged
            # together with the residual blocks. It always stands alone as
            # its own group, mirroring group_grid.py's make_groups (which
            # keeps B's final layer alone for exactly this reason). This
            # dict is either our own _collect_linear_split output or an
            # externally-cached features_B_blocks.pt from steer4rebase's own
            # pipeline -- both use this exact layout, so either is accepted.
            features_b_full_train = {b: v.double() for b, v in train_data["features_B_blocks"].items()}
            num_target_blocks_total = len(features_b_full_train)
            num_target_residual = num_target_blocks_total - 1
            num_source_blocks = block_targets.shape[1]
            num_source_residual_blocks = num_source_blocks - 1
            if num_source_residual_blocks < 1:
                raise ValueError("steer block_ridge: source has no residual blocks to target.")
            if num_target_residual < 1:
                raise ValueError("steer block_ridge: target has no residual blocks to target.")

            residual_train = {b: features_b_full_train[b] for b in range(num_target_residual)}
            output_train = features_b_full_train[num_target_residual]

            if num_target_residual == num_source_residual_blocks:
                grouped_residual = residual_train
                block_group_size = 1
            elif num_target_residual > num_source_residual_blocks:
                grouped_residual = _BLOCK_GROUP_STRATEGIES[block_group_strategy](residual_train, num_source_residual_blocks)
                block_group_size = num_target_residual / num_source_residual_blocks
            else:
                raise ValueError(
                    f"steer block_ridge: target has fewer residual blocks ({num_target_residual}) "
                    f"than source ({num_source_residual_blocks}); cannot group."
                )

            grouped_train = dict(grouped_residual)
            grouped_train[num_source_residual_blocks] = output_train

            local_selected = torch.arange(int(selected.numel()))
            local_blocks_train = {b: v[selected] for b, v in grouped_train.items()}
            coefficients = [
                c.to(dev)
                for c in _fit_block_ridge(
                    local_blocks_train, block_targets, selected=local_selected, regularization=ridge_lambda, mode=block_ridge_mode, rho=rho
                )
            ]

            def correction_fn(
                activations: Mapping[str, torch.Tensor],
                *,
                _coefficients=coefficients,
                _num_target_residual=num_target_residual,
                _num_source_residual_blocks=num_source_residual_blocks,
                _strategy=block_group_strategy,
            ) -> torch.Tensor:
                coef_device = _coefficients[0].device
                residual = {b: activations["blocks"][b].double().to(coef_device) for b in range(_num_target_residual)}
                if _num_target_residual != _num_source_residual_blocks:
                    residual = _BLOCK_GROUP_STRATEGIES[_strategy](residual, _num_source_residual_blocks)
                blocks = dict(residual)
                blocks[_num_source_residual_blocks] = activations["blocks"][_num_target_residual].double().to(coef_device)
                out = _predict_block_ridge(_coefficients, blocks)
                return out.to(dtype=activations["global"].dtype, device=activations["global"].device)

        return {
            "correction_fn": correction_fn,
            "stage_2_strategy": stage_2_strategy,
            "feature_regime": feature_regime,
            "num_source_blocks": num_source_blocks,
            "block_group_size": block_group_size,
            "diagnostics": {"stage1_test_acc": stage1_test_acc},
        }

    def apply_correction(self, prepared: Mapping[str, Any], *, activations: Mapping[str, torch.Tensor], alpha: float = 1.0) -> torch.Tensor:
        correction = prepared["correction_fn"](activations)
        return correction * float(alpha)

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


def _accuracy(features: torch.Tensor, head: torch.Tensor, labels: torch.Tensor) -> float:
    normed = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    logits = normed @ head.T
    return float(logits.argmax(dim=1).eq(labels).double().mean())


@contextmanager
def steer_correction_context(clf_target: Any, prepared: Mapping[str, Any], *, alpha: float = 1.0):
    """
    Wrap ``clf_target.model.encode_image`` so calls during the ``with`` block
    return the L2-normalized visual feature plus ``alpha * correction``.
    Stage 1/2 are fit entirely in normalized-feature space (see
    ``_l2_normalize``), so the correction must be added there too -- not to
    the raw pooled feature -- to match the scale it was calibrated against.
    Used by vision_rebase.py's steer_mode eval branch in place of
    axpy_state_dict + load_into_model.
    """
    visual = clf_target.model.visual
    original_encode_image = clf_target.model.encode_image
    needs_blocks = prepared["stage_2_strategy"] == "block_ridge"
    num_target_residual = _num_residual_blocks(visual) if needs_blocks else 0

    def patched_encode_image(images: torch.Tensor) -> torch.Tensor:
        if needs_blocks:
            with _BlockActivationCapture(visual, list(range(num_target_residual))) as capture:
                out = original_encode_image(images)
            out_norm = _l2_normalize(out)
            # Same numbering as training (see _collect_linear_split): key
            # num_target_residual is the final pooled/output feature.
            blocks = dict(capture.activations)
            blocks[num_target_residual] = out_norm
            activations = {"global": out_norm, "blocks": blocks}
        else:
            out = original_encode_image(images)
            out_norm = _l2_normalize(out)
            activations = {"global": out_norm}
        method = _STEER_SINGLETON
        correction = method.apply_correction(prepared, activations=activations, alpha=alpha)
        return out_norm + correction.to(dtype=out_norm.dtype, device=out_norm.device)

    clf_target.model.encode_image = patched_encode_image
    try:
        yield
    finally:
        clf_target.model.encode_image = original_encode_image


_STEER_SINGLETON = SteerRebase()
register(_STEER_SINGLETON)
