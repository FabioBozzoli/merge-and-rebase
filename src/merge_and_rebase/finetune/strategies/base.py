from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.optim as optim


@runtime_checkable
class Strategy(Protocol):
    name: str

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        device: torch.device,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        """
        Must:
          - set requires_grad appropriately
          - return optimizer, scheduler(step), and info dict
        """
