from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy

import torch
import torch.nn as nn
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def forward_ad_safe_attention_context(device: torch.device):
    old_mha_fastpath: bool | None = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        old_mha_fastpath = bool(torch.backends.mha.get_fastpath_enabled())
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        if device.type == "cuda":
            with sdpa_kernel([SDPBackend.MATH], set_priority=True):
                yield
        else:
            with nullcontext():
                yield
    finally:
        if old_mha_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(old_mha_fastpath)


def _snapshot_named_tensors(named: list[tuple[str, torch.Tensor]]) -> tuple[list[str], tuple[torch.Tensor, ...]]:
    names = [n for n, _ in named]
    values = tuple(t.detach().clone() for _, t in named)
    return names, values


class LinearizedModule:
    """
    First-order linearization helper around a frozen reference module.
    """

    def __init__(self, ref_module: nn.Module) -> None:
        self.ref_module = ref_module
        self.param_names, self.theta0 = _snapshot_named_tensors(list(ref_module.named_parameters()))
        self.buffer_names, self.buffer_values = _snapshot_named_tensors(list(ref_module.named_buffers()))

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        device: torch.device | None = None,
        copy_module: bool = True,
    ) -> LinearizedModule:
        ref_module = deepcopy(module) if copy_module else module
        if device is not None:
            ref_module = ref_module.to(device)
        ref_module.eval()
        for p in ref_module.parameters():
            p.requires_grad = False
        return cls(ref_module)

    def forward(
        self,
        *,
        current_module: nn.Module,
        images: torch.Tensor,
        output_transform: callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.ref_module.train(current_module.training)
        params_now = dict(current_module.named_parameters())
        tangents = tuple(params_now[n] - p0 for n, p0 in zip(self.param_names, self.theta0, strict=True))

        def _f(*primals: torch.Tensor) -> torch.Tensor:
            param_map = {n: p for n, p in zip(self.param_names, primals, strict=True)}
            buffer_map = {n: b for n, b in zip(self.buffer_names, self.buffer_values, strict=True)}
            out = functional_call(self.ref_module, (param_map, buffer_map), (images,), strict=False)
            return output_transform(out) if output_transform is not None else out

        with forward_ad_safe_attention_context(images.device):
            f0, f_jvp = jvp(_f, self.theta0, tangents)
        return f0 + f_jvp
