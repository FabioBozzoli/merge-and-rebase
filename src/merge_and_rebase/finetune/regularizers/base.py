from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Regularizer(Protocol):
    name: str

    def prepare_model(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> None: ...

    def configure(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[callable, dict[str, int]]: ...

