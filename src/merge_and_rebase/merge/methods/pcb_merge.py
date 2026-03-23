from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class PCBMerge:
    """
    Simplified PCB merge on flattened task vectors [N, D].

    Defaults follow the reference implementation, while keeping the code compact and robust:
      - clamp absolute deltas by per-task rank ratios
      - build a balancing mask from intra/inter signals
      - aggregate with per-task lambda scaling
    """

    name: str = "pcb"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        if len(tuned) == 0:
            raise ValueError("tuned must be non-empty")

        method_params = get_method_params(kwargs)
        clamp_min_ratio = float(method_params.get("clamp_min_ratio", 0.01))
        clamp_max_ratio = float(method_params.get("clamp_max_ratio", 0.01))
        att_ratio = float(method_params.get("att_ratio", 0.05))
        lam = float(method_params.get("lam", 1.2))

        self._validate_ratios(
            clamp_min_ratio=clamp_min_ratio,
            clamp_max_ratio=clamp_max_ratio,
            att_ratio=att_ratio,
        )

        w = default_weights(len(tuned), weights)
        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        flat = TaskVector.stack_flattened(deltas, keys, dtype=torch.float32)
        merged_flat = self._pcb_delta(
            flat,
            w=w,
            clamp_min_ratio=clamp_min_ratio,
            clamp_max_ratio=clamp_max_ratio,
            att_ratio=att_ratio,
            lam=lam,
        )
        direction = TaskVector.unflatten_like(merged_flat, like=base, keys=keys)
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
            **kwargs,
        )
        return self.apply(prepared, alpha=float(alpha))

    @staticmethod
    def _validate_ratios(*, clamp_min_ratio: float, clamp_max_ratio: float, att_ratio: float) -> None:
        if not (0.0 <= clamp_min_ratio < 1.0):
            raise ValueError("clamp_min_ratio must be in [0, 1).")
        if not (0.0 <= clamp_max_ratio < 1.0):
            raise ValueError("clamp_max_ratio must be in [0, 1).")
        if clamp_min_ratio + clamp_max_ratio >= 1.0:
            raise ValueError("clamp_min_ratio + clamp_max_ratio must be < 1.")
        if not (0.0 < att_ratio <= 1.0):
            raise ValueError("att_ratio must be in (0, 1].")

    @staticmethod
    def _normalize_minmax(x: torch.Tensor, *, dim: int, eps: float = 1e-12) -> torch.Tensor:
        min_values = x.amin(dim=dim, keepdim=True)
        max_values = x.amax(dim=dim, keepdim=True)
        denom = (max_values - min_values).clamp_min(eps)
        return (x - min_values) / denom

    @staticmethod
    def _clamp_by_ratio(x: torch.Tensor, *, min_ratio: float, max_ratio: float) -> torch.Tensor:
        if x.ndim == 1:
            d = x.shape[0]
            sorted_x, _ = torch.sort(x)
            lo_idx = int(d * min_ratio)
            hi_idx = int(d * (1.0 - max_ratio) - 1)
            hi_idx = max(lo_idx, hi_idx)
            min_v = sorted_x[lo_idx]
            max_v = sorted_x[hi_idx]
            return torch.clamp(x, min=min_v, max=max_v)

        if x.ndim == 2:
            d = x.shape[1]
            sorted_x, _ = torch.sort(x, dim=1)
            lo_idx = int(d * min_ratio)
            hi_idx = int(d * (1.0 - max_ratio) - 1)
            hi_idx = max(lo_idx, hi_idx)
            min_v = sorted_x[:, lo_idx].unsqueeze(1)
            max_v = sorted_x[:, hi_idx].unsqueeze(1)
            return torch.clamp(x, min=min_v, max=max_v)

        raise ValueError(f"Expected x to be 1D or 2D, got shape {tuple(x.shape)}")

    @classmethod
    def _pcb_delta(
        cls,
        M: torch.Tensor,
        *,
        w: torch.Tensor,
        clamp_min_ratio: float,
        clamp_max_ratio: float,
        att_ratio: float,
        lam: float,
    ) -> torch.Tensor:
        if M.ndim != 2:
            raise ValueError(f"Expected M to have shape [N, D], got {tuple(M.shape)}")
        if M.shape[0] == 0:
            return torch.empty((0,), dtype=M.dtype, device=M.device)
        if w.numel() != M.shape[0]:
            raise ValueError(f"weights length must match row count in M. got {w.numel()} vs {M.shape[0]}")

        abs_M = M.abs()
        abs_clamped = cls._clamp_by_ratio(abs_M, min_ratio=clamp_min_ratio, max_ratio=clamp_max_ratio)
        clamped_M = M.sign() * abs_clamped

        norm_abs = cls._normalize_minmax(abs_clamped, dim=1)
        intra = torch.exp(float(M.shape[0]) * norm_abs.square())
        signed_norm = M.sign() * norm_abs
        inter = torch.tanh(M * signed_norm.sum(dim=0))
        balancing = intra * inter

        # Keep strongest attention fraction per task before normalizing to [0, 1].
        scale_seed = cls._clamp_by_ratio(balancing, min_ratio=1.0 - att_ratio, max_ratio=0.0)
        scale = cls._normalize_minmax(scale_seed, dim=1)

        lams = (float(lam) * w.to(device=M.device, dtype=M.dtype)).view(-1, 1)
        num = (clamped_M * lams * scale).sum(dim=0)
        den = scale.sum(dim=0).clamp_min(1e-12)
        return num / den


register(PCBMerge())
register(PCBMerge(name="pcb_merge"))
