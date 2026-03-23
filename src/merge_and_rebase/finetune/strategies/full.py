from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .registry import register


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            if step < warmup_length:
                lr = base_lr * (step + 1) / warmup_length
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            # assign_learning_rate(param_group, lr)
            param_group["lr"] = lr

    return _lr_adjuster


@dataclass(frozen=True)
class FullFinetune:
    name: str = "full"

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
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:

        params = (p for p in model.parameters() if p.requires_grad)
        opt = self.get_optimizer(params, optimizer, lr, weight_decay)
        scheduler = cosine_lr(opt, lr, warmup_length, steps)
        info = {"trainable_params": sum(p.numel() for p in params)}

        return opt, scheduler, info

    def get_optimizer(self, params, opt, lr, weight_decay):
        if opt.lower() == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt.lower() == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt.lower() == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")


register(FullFinetune())
