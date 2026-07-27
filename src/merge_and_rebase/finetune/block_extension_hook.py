from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from merge_and_rebase.eval.block_extension import BlockExtender, BlockExtensionConfig


def apply_block_extension_to_model(
    model: nn.Module,
    block_ext_cfg: dict[str, Any],
    calibration_loader: Any,
    device: str,
) -> int:
    """Extend or shrink a single model's visual transformer depth in-place.

    Args:
        model: An open_clip model (with ``visual.transformer.resblocks``).
        block_ext_cfg: Configuration dict matching ``BlockExtensionConfig`` fields.
        calibration_loader: Data loader for activation capture.
        device: Device string.

    Returns:
        Final depth after extension/shrink.
    """
    cfg = BlockExtensionConfig(
        blocks_to_add=block_ext_cfg.get("blocks_to_add", None),
        target_layers_total=block_ext_cfg.get("target_layers_total", None),
        insertion_order=str(block_ext_cfg.get("insertion_order", "bottom-top")),
        extension_density=str(block_ext_cfg.get("extension_density", "spread")),
        extension_strategy=str(block_ext_cfg.get("extension_strategy", "interpolate_per_weight")),
        dampening_factor=float(block_ext_cfg.get("dampening_factor", 1.0)),
        n_batches_act=max(1, int(block_ext_cfg.get("n_batches_act", 5))),
        calibration_split=str(block_ext_cfg.get("calibration_split", "train")),
        skip_correction=bool(block_ext_cfg.get("skip_correction", False)),
        skip_final_ln=bool(block_ext_cfg.get("skip_final_ln", False)),
        ridge_identity=float(block_ext_cfg.get("ridge_identity", 100.0)),
        n_cascade_iters=max(1, int(block_ext_cfg.get("n_cascade_iters", 1))),
        share_ft_refs=bool(block_ext_cfg.get("share_ft_refs", False)),
        component_ridge=block_ext_cfg.get("component_ridge", None),
        verbose=bool(block_ext_cfg.get("verbose", True)),
        show_progress=bool(block_ext_cfg.get("show_progress", True)),
    )

    model_copy = deepcopy(model)

    extender = BlockExtender(
        model_base=model,
        model_ft=model_copy,
        device=device,
        verbose=bool(cfg.verbose),
        show_progress=bool(cfg.show_progress),
    )

    final_depth = extender.extend_and_calibrate(
        loader=calibration_loader,
        n_batches=cfg.n_batches_act,
        strategy=cfg.extension_strategy,
        dampening_factor=float(cfg.dampening_factor),
        blocks_to_add=cfg.blocks_to_add,
        target_layers_total=cfg.target_layers_total,
        insertion_order=cfg.insertion_order,
        extension_density=cfg.extension_density,
        skip_correction=bool(cfg.skip_correction),
        skip_final_ln=bool(cfg.skip_final_ln),
        ridge_identity=float(cfg.ridge_identity),
        n_cascade_iters=int(cfg.n_cascade_iters),
        share_ft_refs=bool(cfg.share_ft_refs),
        component_ridge=cfg.component_ridge,
    )

    del model_copy
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()

    return final_depth
