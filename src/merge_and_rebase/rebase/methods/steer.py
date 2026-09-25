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
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ...utils.linearization import LinearizedModule
from ..base import TensorDict
from ..registry import register

_LINEARIZED_PARAM_PATTERN = re.compile(r"(?:^|.*\.)(params0|delta)\.(\d+)$")


def is_linearized_checkpoint(state_dict: Mapping[str, Any]) -> bool:
    """True for a steer4rebase LinearizedModelV2 checkpoint.

    Such a checkpoint stores weights as two positionally-indexed
    ``nn.ParameterList``s -- ``params0`` (frozen pretrained weights) and
    ``delta`` (the trained tangent directions) -- instead of a normal CLIP
    state dict, so key-based alignment cannot load it.
    """
    return any(_LINEARIZED_PARAM_PATTERN.match(str(k)) for k in state_dict)


def reconstruct_linearized_checkpoint(
    state_dict: Mapping[str, Any], pretrained_model: nn.Module
) -> dict[str, torch.Tensor]:
    """
    Rebuild absolute finetuned weights from a LinearizedModelV2 checkpoint.

    ``LinearizedModelV2`` keeps ``parameter_names`` as a plain tuple attribute,
    so it is absent from the saved state dict and the index -> parameter-name
    mapping has to be recovered. ``params0`` holds the pretrained weights, so
    each index is resolved by matching it exactly against the pretrained
    model's parameters (unique tensors, hence unambiguous), and the finetuned
    weight is ``params0[i] + delta[i]``. Indices that match nothing (e.g. text
    tower parameters absent from this model) are skipped.
    """
    params0: dict[int, torch.Tensor] = {}
    deltas: dict[int, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = _LINEARIZED_PARAM_PATTERN.match(str(key))
        if match is None or not isinstance(value, torch.Tensor):
            continue
        (params0 if match.group(1) == "params0" else deltas)[int(match.group(2))] = value

    if not params0:
        raise ValueError("steer: no 'params0' entries found in the linearized checkpoint.")
    if set(params0) != set(deltas):
        raise ValueError(
            f"steer: linearized checkpoint has mismatched params0/delta indices "
            f"({len(params0)} vs {len(deltas)})."
        )

    named = list(pretrained_model.named_parameters())
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for position, (_, param) in enumerate(named):
        by_shape.setdefault(tuple(param.shape), []).append(position)

    out: dict[str, torch.Tensor] = {}
    used: set[int] = set()
    for index in sorted(params0):
        base = params0[index]
        for position in by_shape.get(tuple(base.shape), []):
            if position in used:
                continue
            name, param = named[position]
            if torch.equal(param.detach().cpu().to(base.dtype), base.cpu()):
                used.add(position)
                out[name] = (base + deltas[index]).to(param.dtype)
                break

    if not out:
        raise ValueError(
            "steer: could not match any linearized checkpoint parameter against the pretrained model. "
            "The checkpoint's params0 do not correspond to this source model's pretrained weights."
        )
    return out


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
    regularization_scaling: str = "none",
) -> list[torch.Tensor]:
    """
    Fit per-block ridge coefficients, chained with a smoothed-residual carry.

    ``train_targets`` has shape [n_selected, num_blocks, d_B]. Fitting is
    self-contained (depends only on train-side data), so the resulting
    coefficients can be applied to *any* new block activations at predict
    time -- this is the split-fit/predict factoring of steer4rebase's
    stage2.block_ridge (which fits and predicts in a single call).

    ``regularization_scaling="trace"`` fits block ``b`` with
    ``regularization * tr(X_b X_bᵀ) / n``, i.e. relative to the mean
    eigenvalue of that block's Gram matrix. This makes the scalar
    ``regularization`` a dimensionless sweep parameter with comparable
    effective shrinkage across blocks whose activation scales differ.
    """
    if mode not in {"independent", "smoothed_residual"}:
        raise ValueError(f"Unknown steer block_ridge mode: {mode}")
    if mode == "smoothed_residual" and not 0.0 <= rho <= 1.0:
        raise ValueError("steer block_ridge rho must be in [0, 1]")
    if regularization_scaling not in {"none", "trace"}:
        raise ValueError(f"Unknown steer block_ridge regularization_scaling: {regularization_scaling}")

    num_blocks = train_targets.shape[1]
    state = torch.zeros_like(train_targets[:, 0])
    coefficients: list[torch.Tensor] = []
    for block_id in range(num_blocks):
        x_train = blocks_train[block_id][selected]
        local_target = train_targets[:, block_id]
        compensation = rho * state if mode == "smoothed_residual" else torch.zeros_like(state)
        fitted_target = local_target + compensation
        block_regularization = regularization
        if regularization_scaling == "trace":
            block_regularization = regularization * float(x_train.square().sum()) / x_train.shape[0]
        coefficient = _ridge(x_train, fitted_target, block_regularization)
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

_BLOCK_GRANULARITIES = {"residual", "attention", "linear", "model"}


@dataclass(frozen=True)
class _LinearActivationSite:
    """A linear operation and the activation used to predict its correction.

    ``module_name`` is ``None`` only for raw projection parameters such as a
    CLIP ViT's final ``visual.proj`` matrix.  Those sites use the encoder's
    final output, supplied by the caller after the forward pass.
    """

    name: str
    module_name: str | None
    hook: str  # "pre", "post", or "final"


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


def _linear_activation_sites(visual: nn.Module) -> tuple[_LinearActivationSite, ...]:
    """Return ordered linear-operation sites for a CLIP visual encoder.

    Besides explicit ``Linear`` and ``Conv2d`` modules, PyTorch's
    ``MultiheadAttention`` owns a packed QKV projection as raw parameters and
    calls its ``out_proj`` functionally (so a hook on the child Linear does not
    fire).  Represent those two operations with pre/post hooks on the parent
    attention module.  A raw top-level ``proj`` matrix is represented by a
    final-output site.
    """
    modules = dict(visual.named_modules())
    mha_out_proj_names = {
        f"{name}.out_proj" for name, module in modules.items() if name and isinstance(module, nn.MultiheadAttention)
    }
    sites: list[_LinearActivationSite] = []
    for name, module in modules.items():
        if not name:
            continue
        if isinstance(module, nn.MultiheadAttention):
            sites.append(_LinearActivationSite(f"{name}.in_proj", name, "pre"))
            sites.append(_LinearActivationSite(f"{name}.out_proj", name, "post"))
        elif isinstance(module, (nn.Linear, nn.Conv2d)) and name not in mha_out_proj_names:
            sites.append(_LinearActivationSite(name, name, "post"))

    direct_parameters = dict(visual.named_parameters(recurse=False))
    if "proj" in direct_parameters:
        sites.append(_LinearActivationSite("proj", None, "final"))
    if not sites:
        raise ValueError("No linear operations were found in the CLIP visual encoder.")
    return tuple(sites)


_VIT_LINEAR_SITE_PATTERN = re.compile(r"^transformer\.resblocks\.(\d+)\.(.+)$")


def _linear_site_groups(source_visual: nn.Module, target_visual: nn.Module) -> tuple[tuple[int, ...], ...]:
    """Map target linear sites to source sites, preserving operation roles.

    Equal-depth models map one-to-one. For a deeper target ViT, consecutive
    target transformer layers are assigned to each source layer, but QKV,
    attention output, MLP expansion, and MLP projection sites are never mixed
    with one another. Stem and final projection remain singleton groups.
    """
    source_sites = _linear_activation_sites(source_visual)
    target_sites = _linear_activation_sites(target_visual)
    target_by_name = {site.name: i for i, site in enumerate(target_sites)}
    if len(target_by_name) != len(target_sites):
        raise ValueError("Target visual encoder exposes duplicate linear site names.")

    source_residual = [
        (i, int(match.group(1)), match.group(2))
        for i, site in enumerate(source_sites)
        if (match := _VIT_LINEAR_SITE_PATTERN.match(site.name)) is not None
    ]
    target_residual = [
        (i, int(match.group(1)), match.group(2))
        for i, site in enumerate(target_sites)
        if (match := _VIT_LINEAR_SITE_PATTERN.match(site.name)) is not None
    ]

    if not source_residual and len(source_sites) != len(target_sites):
        raise ValueError(
            "Linear granularity can group unequal depths only for ViT transformer.resblocks encoders."
        )
    source_depth = max((layer for _, layer, _ in source_residual), default=-1) + 1
    target_depth = max((layer for _, layer, _ in target_residual), default=-1) + 1
    if target_depth < source_depth:
        raise ValueError(
            f"steer linear granularity cannot map shallower target depth {target_depth} to source depth {source_depth}."
        )
    boundaries = [round(i * target_depth / source_depth) for i in range(source_depth + 1)] if source_depth else []

    groups: list[tuple[int, ...]] = []
    for source_site in source_sites:
        match = _VIT_LINEAR_SITE_PATTERN.match(source_site.name)
        if match is None:
            if source_site.name not in target_by_name:
                raise ValueError(f"Target visual encoder lacks source linear site {source_site.name!r}.")
            groups.append((target_by_name[source_site.name],))
            continue
        source_layer = int(match.group(1))
        role = match.group(2)
        members = tuple(
            target_id
            for target_id, target_layer, target_role in target_residual
            if boundaries[source_layer] <= target_layer < boundaries[source_layer + 1] and target_role == role
        )
        if not members:
            raise ValueError(
                f"Target visual encoder has no linear sites matching source layer {source_layer} role {role!r}."
            )
        groups.append(members)
    return tuple(groups)


def _group_linear_site_activations(
    blocks: Mapping[int, torch.Tensor],
    groups: Sequence[Sequence[int]],
    strategy: str,
) -> dict[int, torch.Tensor]:
    grouped: dict[int, torch.Tensor] = {}
    for source_id, members in enumerate(groups):
        tensors = [blocks[target_id] for target_id in members]
        if strategy == "concat":
            grouped[source_id] = torch.cat(tensors, dim=1) if len(tensors) > 1 else tensors[0]
        elif strategy == "sum_avg":
            grouped[source_id] = torch.stack(tensors, dim=0).mean(dim=0) if len(tensors) > 1 else tensors[0]
        else:
            raise ValueError(f"Unknown steer block grouping strategy: {strategy}")
    return grouped


def _linear_parameter_blocks(visual: nn.Module) -> tuple[tuple[int, ...], int]:
    """Assign every visual parameter to its nearest linear-operation site.

    Weight and bias of one operation share a block. Affine normalizations and
    embeddings, which do not have their own linear activation site, are folded
    into the next linear operation in parameter order (or the preceding one at
    the tail). This preserves an exact, exhaustive JVP decomposition while
    keeping the requested one-block-per-linear-layer semantics.
    """
    parameter_names = [name for name, _ in visual.named_parameters()]
    sites = _linear_activation_sites(visual)
    owner_by_name: dict[str, int] = {}
    for block_id, site in enumerate(sites):
        if site.hook == "pre":
            attention_name = site.name.removesuffix(".in_proj")
            prefixes = (
                f"{attention_name}.in_proj_",
                f"{attention_name}.bias_k",
                f"{attention_name}.bias_v",
            )
            for name in parameter_names:
                if name.startswith(prefixes):
                    owner_by_name[name] = block_id
        elif site.hook == "final":
            owner_by_name[site.name] = block_id
        else:
            prefix = f"{site.name}."
            for name in parameter_names:
                if name.startswith(prefix):
                    owner_by_name[name] = block_id

    if not owner_by_name:
        raise ValueError("No visual parameters could be associated with linear-operation sites.")

    def _common_prefix_parts(left: str, right: str) -> int:
        count = 0
        for a, b in zip(left.split("."), right.split("."), strict=False):
            if a != b:
                break
            count += 1
        return count

    block_ids: list[int] = []
    for name in parameter_names:
        owner = owner_by_name.get(name)
        if owner is None:
            # LayerNorm 1 belongs with attention; LayerNorm 2 belongs with the
            # MLP. This is both more meaningful and independent of PyTorch's
            # parameter traversal order (top-level parameters are yielded
            # before child-module parameters, even when they are used last).
            residual_prefix = name.split(".ln_", maxsplit=1)[0]
            if ".ln_1." in name:
                owner = next(
                    (i for i, site in enumerate(sites) if site.name.startswith(f"{residual_prefix}.attn")),
                    None,
                )
            elif ".ln_2." in name:
                owner = next(
                    (i for i, site in enumerate(sites) if site.name.startswith(f"{residual_prefix}.mlp")),
                    None,
                )
            if owner is None and name in _VIT_OUTPUT_PARAMETERS:
                owner = len(sites) - 1
            if owner is None:
                scores = [_common_prefix_parts(name, site.name) for site in sites]
                best = max(scores)
                owner = scores.index(best) if best > 0 else 0
        block_ids.append(owner)

    used = set(block_ids)
    expected = set(range(len(sites)))
    if used != expected:
        missing = sorted(expected - used)
        raise ValueError(f"Linear-operation partition produced empty blocks: {missing}")
    return tuple(block_ids), len(sites)


def _parameter_blocks_for_visual(
    visual: nn.Module, block_granularity: str = "residual"
) -> tuple[tuple[int | None, ...], int]:
    if block_granularity == "model":
        return tuple(0 for _ in visual.parameters()), 1
    if block_granularity == "linear":
        return _linear_parameter_blocks(visual)
    if block_granularity not in {"residual", "attention"}:
        raise ValueError(f"Unknown steer block granularity: {block_granularity}")
    # ``attention`` changes B's predictor features, not A's JVP partition:
    # each source transformer block still contributes one delta target, plus
    # the trailing output block. This mirrors steer_text's attention source.
    names = [name for name, _ in visual.named_parameters()]
    if _is_resnet_visual(visual):
        if block_granularity == "attention":
            raise ValueError("steer attention granularity supports CLIP ViT visual encoders only.")
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

    def __enter__(self) -> _BlockActivationCapture:
        for block_id in self.block_ids:
            module = _get_block_module(self.visual, block_id)
            self._handles.append(module.register_forward_hook(self._make_hook(block_id)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _AttentionActivationCapture:
    """Capture each ViT block's attention result immediately before ``W_O``.

    Recent OpenCLIP attention modules call an explicit ``out_proj`` child, so
    a normal forward pre-hook gives exactly the desired tensor. PyTorch
    ``nn.MultiheadAttention`` instead applies that child's parameters through
    a functional call and never invokes the child module. For that legacy
    layout we temporarily run the attention with an identity output projection,
    save its raw result, and apply the original projection before returning it.
    The surrounding transformer therefore receives the unchanged value.
    """

    def __init__(self, visual: nn.Module, block_ids: Sequence[int]) -> None:
        if _is_resnet_visual(visual):
            raise ValueError("steer attention granularity supports CLIP ViT visual encoders only.")
        self.visual = visual
        self.block_ids = list(block_ids)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._patched_forwards: list[tuple[nn.MultiheadAttention, Callable[..., Any]]] = []

    def _make_pre_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: Any) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                self.activations[block_id] = _pool_block_output(inputs[0]).detach()

        return hook

    def _patch_multihead_attention(self, module: nn.MultiheadAttention, block_id: int) -> None:
        original_forward = module.forward
        original_out_proj = module.out_proj
        identity_out_proj = nn.Linear(
            original_out_proj.in_features,
            original_out_proj.out_features,
            bias=original_out_proj.bias is not None,
            device=original_out_proj.weight.device,
            dtype=original_out_proj.weight.dtype,
        )
        with torch.no_grad():
            identity_out_proj.weight.copy_(
                torch.eye(
                    original_out_proj.out_features,
                    original_out_proj.in_features,
                    device=original_out_proj.weight.device,
                    dtype=original_out_proj.weight.dtype,
                )
            )
            if identity_out_proj.bias is not None:
                identity_out_proj.bias.zero_()

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            module.out_proj = identity_out_proj
            try:
                output = original_forward(*args, **kwargs)
            finally:
                module.out_proj = original_out_proj

            raw = output[0] if isinstance(output, tuple) else output
            self.activations[block_id] = _pool_block_output(raw).detach()
            projected = F.linear(raw, original_out_proj.weight, original_out_proj.bias)
            if isinstance(output, tuple):
                return (projected,) + output[1:]
            return projected

        self._patched_forwards.append((module, original_forward))
        module.forward = wrapped_forward  # type: ignore[method-assign]

    def __enter__(self) -> _AttentionActivationCapture:
        for block_id in self.block_ids:
            block = _get_block_module(self.visual, block_id)
            attention = getattr(block, "attn", None)
            out_proj = getattr(attention, "out_proj", None)
            if not isinstance(attention, nn.Module) or not isinstance(out_proj, nn.Module):
                raise ValueError(
                    f"steer attention granularity could not find an attention out_proj in "
                    f"visual transformer block {block_id} ({type(block).__name__})."
                )
            if isinstance(attention, nn.MultiheadAttention):
                self._patch_multihead_attention(attention, block_id)
            else:
                self._handles.append(out_proj.register_forward_pre_hook(self._make_pre_hook(block_id)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, original_forward in self._patched_forwards:
            module.forward = original_forward  # type: ignore[method-assign]
        self._patched_forwards.clear()


class _LinearActivationCapture:
    """Capture inputs/outputs at the sites returned by `_linear_activation_sites`."""

    def __init__(self, visual: nn.Module, sites: Sequence[_LinearActivationSite]) -> None:
        self.visual = visual
        self.sites = list(sites)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_pre_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, inputs: Any) -> None:
            self.activations[block_id] = _pool_block_output(inputs).detach()

        return hook

    def _make_post_hook(self, block_id: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            self.activations[block_id] = _pool_block_output(output).detach()

        return hook

    def __enter__(self) -> _LinearActivationCapture:
        modules = dict(self.visual.named_modules())
        for block_id, site in enumerate(self.sites):
            if site.hook == "final":
                continue
            if site.module_name is None or site.module_name not in modules:
                raise ValueError(f"Cannot find target linear module {site.module_name!r} for site {site.name!r}.")
            module = modules[site.module_name]
            if site.hook == "pre":
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(block_id)))
            else:
                self._handles.append(module.register_forward_hook(self._make_post_hook(block_id)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _NoopActivationCapture:
    def __init__(self) -> None:
        self.activations: dict[int, torch.Tensor] = {}

    def __enter__(self) -> _NoopActivationCapture:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _activation_capture_for_visual(
    visual: nn.Module, block_granularity: str
) -> tuple[Any, tuple[int, ...], int]:
    """Build a target-side capture and identify sites filled by final output."""
    if block_granularity == "model":
        return _NoopActivationCapture(), (0,), 1
    if block_granularity == "linear":
        sites = _linear_activation_sites(visual)
        final_ids = tuple(i for i, site in enumerate(sites) if site.hook == "final")
        return _LinearActivationCapture(visual, sites), final_ids, len(sites)
    if block_granularity in {"residual", "attention"}:
        num_residual = _num_residual_blocks(visual)
        capture = (
            _BlockActivationCapture(visual, list(range(num_residual)))
            if block_granularity == "residual"
            else _AttentionActivationCapture(visual, list(range(num_residual)))
        )
        return capture, (num_residual,), num_residual + 1
    raise ValueError(f"Unknown steer block granularity: {block_granularity}")


def _complete_captured_activations(
    capture: Any,
    *,
    final_output: torch.Tensor,
    final_ids: Sequence[int],
    expected_blocks: int,
) -> dict[int, torch.Tensor]:
    blocks = dict(capture.activations)
    for block_id in final_ids:
        blocks[block_id] = final_output
    missing = sorted(set(range(expected_blocks)) - set(blocks))
    if missing:
        raise RuntimeError(f"steer failed to capture target activations for blocks {missing}")
    return blocks


# --------------------------------------------------------------------------
# D. Feature computation + disk cache
# --------------------------------------------------------------------------


@torch.no_grad()
def _iter_batches(loader: Any, *, device: torch.device) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for x, y in loader:
        yield x.to(device), y.to(device)


# Train images kept per class when extracting features for these tasks. Their
# full train splits (SVHN ~73k, MNIST 60k) make per-layer caches too large,
# while the support set never exceeds ~20 shots per class. Test stays full.
_TRAIN_FEATURES_PER_CLASS = {"SVHN": 50, "MNIST": 50}


def _dataset_labels(dataset: Any) -> torch.Tensor:
    """Labels of an HFVisionDataset read from its label column, without decoding images."""
    return torch.tensor([dataset._map_label(y) for y in dataset.split[dataset.label_key]])


def _aligned_loader_pair(
    source_loader: Any, target_loader: Any, *, per_class: int | None = None
) -> tuple[Any, Any]:
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
    if per_class is not None:
        # Fixed seed, independent of the run seed, so every run shares one cache.
        indices = _few_shot(_dataset_labels(source_ds), per_class, seed=0).tolist()
        source_ds, target_ds = Subset(source_ds, indices), Subset(target_ds, indices)
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
        features_a.append(f_a_pre.cpu())
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
    block_granularity: str = "residual",
) -> dict[str, torch.Tensor]:
    """
    Linear regime: Taylor-linearized delta_A via LinearizedModule (torch.func.jvp),
    decomposed per visual block. Reuses this repo's existing linearization
    primitive (utils/linearization.py) instead of porting steer4rebase's
    deprecated-functorch LinearizedModelV2.
    """
    block_ids, num_a_blocks = _parameter_blocks_for_visual(source_pretrained_visual, block_granularity)
    param_names = [name for name, _ in source_pretrained_visual.named_parameters()]
    linmod = LinearizedModule.from_module(source_pretrained_visual, device=device, copy_module=False, param_names=param_names)
    theta0 = dict(zip(linmod.param_names, linmod.theta0, strict=True))

    target_capture, target_final_ids, num_target_blocks = _activation_capture_for_visual(
        target_visual, block_granularity
    )

    features_a: list[torch.Tensor] = []
    delta_a: list[torch.Tensor] = []
    delta_a_blocks: list[torch.Tensor] = []
    features_b: list[torch.Tensor] = []
    # For residual granularity this preserves steer4rebase's cache layout:
    # residual blocks first, then the final pooled/output feature. The new
    # linear/model granularities use isolated cache directories and store their
    # own contiguous 0..N-1 site layout.
    features_b_blocks: dict[int, list[torch.Tensor]] = {b: [] for b in range(num_target_blocks)}
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

        # Capture object instances can be reused across batches: their hook
        # handles are removed on exit and each fired site overwrites its last
        # activation.
        with target_capture as capture:
            with torch.no_grad():
                out_b = _l2_normalize(target_visual(x_b))
        captured_blocks = _complete_captured_activations(
            capture,
            final_output=out_b.detach(),
            final_ids=target_final_ids,
            expected_blocks=num_target_blocks,
        )
        for block_id in range(num_target_blocks):
            features_b_blocks[block_id].append(captured_blocks[block_id].cpu())
        features_b.append(out_b.detach().cpu())

    return {
        "features_A": torch.cat(features_a, dim=0),
        "delta_A": torch.cat(delta_a, dim=0),
        "delta_A_blocks": torch.cat(delta_a_blocks, dim=0),
        "features_B": torch.cat(features_b, dim=0),
        "features_B_blocks": {b: torch.cat(chunks, dim=0) for b, chunks in features_b_blocks.items()},
        "y_A": torch.cat(labels, dim=0),
    }


def _cache_split_dir(
    feature_cache_dir: str,
    source_tag: str,
    target_tag: str,
    task: str,
    regime: str,
    split: str,
    block_granularity: str = "residual",
) -> Path:
    cache_regime = regime if block_granularity == "residual" else f"{regime}_{block_granularity}granularity"
    return Path(feature_cache_dir) / f"{source_tag}_to_{target_tag}" / task / cache_regime / split


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
    block_granularity: str = "residual",
) -> dict[str, Any]:
    cache_dir = _cache_split_dir(
        feature_cache_dir,
        source_tag,
        target_tag,
        task,
        feature_regime,
        split,
        block_granularity,
    )
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
        block_granularity: str = "residual",
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
        block_ridge_lambda_scaling: str = "none",
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
        if block_granularity not in _BLOCK_GRANULARITIES:
            raise ValueError(f"steer block_granularity must be one of: {sorted(_BLOCK_GRANULARITIES)}")
        if block_group_strategy not in _BLOCK_GROUP_STRATEGIES:
            raise ValueError(f"steer block_group_strategy must be one of: {sorted(_BLOCK_GROUP_STRATEGIES)}")
        if block_ridge_lambda_scaling not in {"none", "trace"}:
            raise ValueError("steer block_ridge_lambda_scaling must be 'none' or 'trace'")
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
        eval_basis_mismatch = False
        head_cache_dir = _cache_split_dir(
            feature_cache_dir,
            source_tag,
            target_tag,
            task,
            feature_regime,
            "train",
            block_granularity,
        )
        cached_w_a = None if force_recompute_features else _load_cached_head(head_cache_dir / "head_A.pt")
        cached_w_b = None if force_recompute_features else _load_cached_head(head_cache_dir / "head_B.pt")

        clf_source.build_zeroshot_text_features(classnames, source_build_cfg_task, cache_dir="src/.cache/zs_cache")
        clf_target.build_zeroshot_text_features(classnames, build_cfg_task, cache_dir="src/.cache/zs_cache")
        live_w_a = clf_source._zs_text_features.detach().to(device="cpu", dtype=torch.float64)
        live_w_b = clf_target._zs_text_features.detach().to(device="cpu", dtype=torch.float64)

        if cached_w_a is not None and cached_w_b is not None:
            w_a, w_b = cached_w_a, cached_w_b
            if verbose:
                print(f"{log_prefix} using cached heads at {head_cache_dir}")
            # Unlike steer4rebase's run.py -- which both fits *and* evaluates
            # against data.w_b -- vision_rebase.py evaluates through a live
            # forward pass classified with clf_target's own zero-shot head
            # (eval/utils.py::eval_task_top1 rebuilds it). If the cached head
            # differs, Stage 1/2 are fit in a different basis than the one the
            # correction is scored in, and the live "rebased" numbers are not
            # comparable to the cached-space diagnostics below.
            if w_b.shape != live_w_b.shape or not torch.allclose(w_b, live_w_b, rtol=1e-4, atol=1e-6):
                if w_b.shape != live_w_b.shape:
                    print(
                        f"{log_prefix} WARNING: cached head_B has shape {tuple(w_b.shape)} but the live "
                        f"zero-shot head used at eval has shape {tuple(live_w_b.shape)}."
                    )
                else:
                    # A pure scale difference (e.g. task-arithmetic heads carry a
                    # logit_scale factor while ours are L2-normalized) leaves argmax
                    # unchanged and cancels out of Stage 1 (logit_map scales with the
                    # head, pinv(w_b) inversely). A *direction* difference means the
                    # two heads are genuinely different classifiers, and the live
                    # 'rebased' score is then measured in the wrong basis.
                    cos = F.cosine_similarity(w_b, live_w_b, dim=-1)
                    scale = w_b.norm(dim=-1) / live_w_b.norm(dim=-1).clamp_min(1e-12)
                    print(
                        f"{log_prefix} cached head_B differs from the live eval head: "
                        f"max abs diff={float((w_b - live_w_b).abs().max()):.6g}, "
                        f"per-class cosine min={float(cos.min()):.6f} mean={float(cos.mean()):.6f}, "
                        f"norm ratio mean={float(scale.mean()):.4f}"
                    )
                    if float(cos.min()) > 0.999:
                        print(
                            f"{log_prefix} -> directions match; the heads differ only by scale, which does "
                            f"not affect argmax or the fitted correction."
                        )
                    else:
                        eval_basis_mismatch = True
                        print(
                            f"{log_prefix} WARNING: cached head_B does not match the zero-shot head this run "
                            f"evaluates with (per-class cosine mean={float(cos.mean()):.4f}; near-zero means "
                            f"the two heads live in different embedding spaces, i.e. the cache was built for a "
                            f"different model than '{target_tag}'). The cached-space diagnostics below remain "
                            f"valid and are the numbers comparable to steer4rebase's run.py, but the live "
                            f"'rebased' column is measured against a different classifier and is meaningless. "
                            f"To get valid live numbers, point feature_cache_dir at a fresh directory with "
                            f"force_recompute_features=true, or align source/target in the config with the "
                            f"models the cache was generated from."
                        )
        else:
            w_a, w_b = live_w_a, live_w_b
            head_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(w_a, head_cache_dir / "head_A.pt")
            torch.save(w_b, head_cache_dir / "head_B.pt")

        source_finetuned_visual = clf_source.model.visual.to(dev).eval()
        source_pretrained_visual = clf_source_pretrained.model.visual.to(dev).eval()
        target_visual = clf_target.model.visual.to(dev).eval()

        # The whole method transports A's fine-tuning delta, so an A whose visual
        # weights equal the pretrained ones yields delta_A == 0 and silently
        # degenerate results. This happens whenever the finetuned checkpoint fails
        # to load (mismatched keyspace under a non-strict load), so check rather
        # than trust the caller.
        finetuned_params = dict(source_finetuned_visual.named_parameters())
        pretrained_params = dict(source_pretrained_visual.named_parameters())
        if all(
            torch.equal(v.detach(), pretrained_params[k].detach())
            for k, v in finetuned_params.items()
            if k in pretrained_params
        ):
            raise ValueError(
                "steer: the finetuned source visual encoder is identical to the pretrained one, so "
                "delta_A would be zero. The tuned checkpoint most likely failed to load into the source "
                "model (mismatched keyspace)."
            )
        need_blocks = stage_2_strategy == "block_ridge"

        def _compute(split: str) -> dict[str, Any]:
            source_loader, target_loader = _aligned_loader_pair(
                getattr(source_loaders, split),
                getattr(target_loaders, split),
                per_class=train_per_class if split == "train" else None,
            )
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
                block_granularity=block_granularity,
            )

        train_per_class = _TRAIN_FEATURES_PER_CLASS.get(task)
        train_data = _load_or_compute_split(
            feature_cache_dir=feature_cache_dir,
            source_tag=source_tag,
            target_tag=target_tag,
            task=task,
            feature_regime=feature_regime,
            # Separate cache dir so a subsampled cache never masquerades as a full one.
            split="train" if train_per_class is None else f"train_{train_per_class}perclass",
            force_recompute_features=force_recompute_features,
            need_blocks=need_blocks,
            compute_fn=lambda: _compute("train"),
            verbose=verbose,
            block_granularity=block_granularity,
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
            block_granularity=block_granularity,
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

            features_b_full_train = {b: v.double() for b, v in train_data["features_B_blocks"].items()}
            num_target_blocks_total = len(features_b_full_train)
            num_source_blocks = block_targets.shape[1]
            num_target_residual = None
            num_source_residual_blocks = None
            linear_site_groups = None

            if block_granularity in {"residual", "attention"}:
                # Residual and attention layouts both expose one predictor per
                # transformer block plus the final projected feature. The latter
                # has a different width and must remain outside block grouping.
                num_target_residual = num_target_blocks_total - 1
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
                    grouped_residual = _BLOCK_GROUP_STRATEGIES[block_group_strategy](
                        residual_train, num_source_residual_blocks
                    )
                    block_group_size = num_target_residual / num_source_residual_blocks
                else:
                    raise ValueError(
                        f"steer block_ridge: target has fewer residual blocks ({num_target_residual}) "
                        f"than source ({num_source_residual_blocks}); cannot group."
                    )
                grouped_train = dict(grouped_residual)
                grouped_train[num_source_residual_blocks] = output_train
            elif block_granularity == "linear":
                linear_site_groups = _linear_site_groups(source_pretrained_visual, target_visual)
                if len(linear_site_groups) != num_source_blocks:
                    raise ValueError(
                        f"steer linear granularity produced {len(linear_site_groups)} target groups for "
                        f"{num_source_blocks} source blocks."
                    )
                grouped_train = _group_linear_site_activations(
                    features_b_full_train, linear_site_groups, block_group_strategy
                )
                block_group_size = num_target_blocks_total / num_source_blocks
            else:  # model: exactly one source and one target block
                if num_source_blocks != 1 or num_target_blocks_total != 1:
                    raise ValueError(
                        f"steer model granularity expected one source/target block, got "
                        f"{num_source_blocks}/{num_target_blocks_total}."
                    )
                grouped_train = features_b_full_train
                block_group_size = 1

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
                    regularization_scaling=block_ridge_lambda_scaling,
                )
            ]

            def correction_fn(
                activations: Mapping[str, torch.Tensor],
                *,
                _coefficients=coefficients,
                _num_target_residual=num_target_residual,
                _num_source_residual_blocks=num_source_residual_blocks,
                _strategy=block_group_strategy,
                _granularity=block_granularity,
                _num_target_blocks_total=num_target_blocks_total,
                _linear_site_groups=linear_site_groups,
            ) -> torch.Tensor:
                coef_device = _coefficients[0].device
                if _granularity in {"residual", "attention"}:
                    residual = {
                        b: activations["blocks"][b].double().to(coef_device)
                        for b in range(_num_target_residual)
                    }
                    if _num_target_residual != _num_source_residual_blocks:
                        residual = _BLOCK_GROUP_STRATEGIES[_strategy](residual, _num_source_residual_blocks)
                    blocks = dict(residual)
                    blocks[_num_source_residual_blocks] = (
                        activations["blocks"][_num_target_residual].double().to(coef_device)
                    )
                elif _granularity == "linear":
                    raw_blocks = {
                        b: activations["blocks"][b].double().to(coef_device)
                        for b in range(_num_target_blocks_total)
                    }
                    blocks = _group_linear_site_activations(raw_blocks, _linear_site_groups, _strategy)
                else:
                    blocks = {
                        b: activations["blocks"][b].double().to(coef_device)
                        for b in range(_num_target_blocks_total)
                    }
                out = _predict_block_ridge(_coefficients, blocks)
                return out.to(dtype=activations["global"].dtype, device=activations["global"].device)

        # Stage 2 accuracy in *cached-tensor space*, computed exactly the way
        # steer4rebase's run.py reports it:
        #     accuracy(f_b_test + prediction, w_b, test_labels)
        # i.e. predicted from the cached test features/blocks, with no alpha
        # and no live forward pass. This is the number directly comparable to
        # the original repo's reported results; the "rebased" column printed
        # by vision_rebase.py instead goes through the live eval path
        # (activation hooks + alpha sweep), so comparing the two isolates a
        # Stage 1/2 problem from a live-integration problem.
        cached_test_activations: dict[str, Any] = {"global": f_b_test}
        if need_blocks:
            cached_test_activations["blocks"] = {b: v.double() for b, v in test_data["features_B_blocks"].items()}
        stage2_test_acc = _accuracy(f_b_test + correction_fn(cached_test_activations), w_b, test_labels)
        # Uncorrected zero-shot accuracy of B in cached-tensor space. This must
        # match the target_zeroshot baseline vision_rebase.py measures with a
        # live forward pass; a material gap means the cached features/head do
        # not come from the model being evaluated (e.g. a cache built with the
        # source/target assignment swapped), which invalidates any comparison
        # between the cached-space diagnostics and the live "rebased" column.
        stage0_test_acc = _accuracy(f_b_test, w_b, test_labels)
        if verbose:
            print(
                f"{log_prefix} prepare: cached-space B zero-shot test acc = {stage0_test_acc:.4f} "
                f"(compare with the target_zeroshot baseline below -- they should match)"
            )
            print(
                f"{log_prefix} prepare: stage2 ({stage_2_strategy}) cached test acc = {stage2_test_acc:.4f} "
                f"(predicted from B's cached features, no alpha -- matches run.py's reported metric)"
            )

        return {
            "correction_fn": correction_fn,
            "stage_2_strategy": stage_2_strategy,
            "feature_regime": feature_regime,
            "block_granularity": block_granularity,
            "block_ridge_lambda_scaling": block_ridge_lambda_scaling,
            "num_source_blocks": num_source_blocks,
            "block_group_size": block_group_size,
            "eval_basis_mismatch": eval_basis_mismatch,
            "diagnostics": {
                "stage0_test_acc": stage0_test_acc,
                "stage1_test_acc": stage1_test_acc,
                "stage2_test_acc": stage2_test_acc,
            },
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
    block_granularity = str(prepared.get("block_granularity", "residual"))

    def patched_encode_image(images: torch.Tensor) -> torch.Tensor:
        if needs_blocks:
            capture_context, final_ids, expected_blocks = _activation_capture_for_visual(
                visual, block_granularity
            )
            with capture_context as capture:
                out = original_encode_image(images)
            out_norm = _l2_normalize(out)
            blocks = _complete_captured_activations(
                capture,
                final_output=out_norm,
                final_ids=final_ids,
                expected_blocks=expected_blocks,
            )
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
