from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights


@dataclass(frozen=True)
class IsoCMerge:
    name: str = "isoc_merge"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        w = default_weights(len(tuned), weights)

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for k in tqdm(keys, desc=f"Computing ISO-C directions - SVD Precision {torch.float64}"):
            b = base[k]
            if b.ndim == 2 and "text_projection" not in k:
                direction[k] = self._isoc_delta([d[k] for d in deltas], w=w).to(dtype=b.dtype, device=b.device)
            else:
                direction[k] = torch.zeros_like(b)
        return base, direction

    def apply(self, prepared: tuple[TensorDict, TensorDict], *, alpha: float, **kwargs) -> TensorDict:
        base, direction = prepared
        return axpy_state_dict(base, direction, alpha=float(alpha))

    def merge(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        prepared = self.prepare(
            base=base,
            tuned=tuned,
            weights=weights,
            strict=strict,
        )
        return self.apply(prepared, alpha=float(alpha))

    @staticmethod
    def _isoc_delta(mats: list[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
        weighted = [float(wi) * m for wi, m in zip(w, mats, strict=False)]
        return IsoCMerge._isotropize(sum(weighted))

    @staticmethod
    def _isotropize(mat: torch.Tensor) -> torch.Tensor:
        u, s, v = torch.linalg.svd(mat.to(torch.float64), full_matrices=False)
        s_iso = torch.ones_like(s) * s.mean()
        return (u @ torch.diag(s_iso) @ v).type_as(mat)


register(IsoCMerge())
