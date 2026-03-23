from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from merge_and_rebase.finetune.strategies.full import cosine_lr
from merge_and_rebase.utils.linearization import LinearizedModule

from .registry import register


@dataclass(frozen=True)
class NtkVision:
    """
    First-order linearized fine-tuning around initialization using JVP:

      f_lin(x; theta) = f(x; theta0) + J_theta0(x) @ (theta - theta0)

    Implemented with torch.func.jvp over the *whole ImageEncoder* parameter set.
    """

    name: str = "ntk"

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_length: int,
        optimizer: str = "adamw",
        steps: int,
        device: torch.device,
        strategy_cfg: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        del strategy_cfg  # currently unused for ntk-jvp mode

        # Train all parameters; the model forward is replaced with a linearized one.
        for p in model.parameters():
            p.requires_grad = True

        model.to(device)
        linearized = LinearizedModule.from_module(model, device=device, copy_module=True)
        if not linearized.param_names:
            raise RuntimeError("No parameters found for NTK strategy.")

        def _linearized_forward(images: torch.Tensor) -> torch.Tensor:
            return linearized.forward(current_module=model, images=images)

        model.forward = _linearized_forward  # type: ignore[method-assign]
        model._ntk_linearized = True  # type: ignore[attr-defined]

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = self.get_optimizer(trainable_params, optimizer, lr, weight_decay)
        scheduler = cosine_lr(opt, lr, warmup_length, steps)

        info: dict[str, int] = {
            "trainable_params": sum(p.numel() for p in trainable_params),
            "linearized_params": len(linearized.param_names),
            "linearized_buffers": len(linearized.buffer_names),
        }
        return opt, scheduler, info

    def get_optimizer(self, params, opt: str, lr: float, weight_decay: float) -> optim.Optimizer:
        if opt.lower() == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        if opt.lower() == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if opt.lower() == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {opt}")


register(NtkVision())
