from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from .registry import register


@dataclass(frozen=True)
class LinearProbe:
    name: str = "linear_probe"

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
        raise NotImplementedError("LinearProbe strategy is not yet implemented.")


register(LinearProbe())
