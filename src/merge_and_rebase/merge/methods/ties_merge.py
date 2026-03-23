from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class TIESMerge:
    """
    TIES-style merge:
      - prune each task vector to topK magnitude per-row (here: global vector topK fraction)
      - resolve sign per-dimension (majority)
      - disjoint merge by sign (mean by default)
    """

    name: str = "ties_merge"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        method_params = get_method_params(kwargs)
        merging_type = str(method_params.get("merging_type", "mean"))  # mean | sum | max
        topk = float(method_params.get("topk", 1.0))  # topK fraction per row for pruning (in [0,1] or [0,100])

        w = default_weights(len(tuned), weights)

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        flat = TaskVector.stack_flattened(deltas, keys, dtype=torch.float32)

        pruned, _mask = self._topk_mask(flat, topk=topk)
        sign = self._resolve_sign(pruned)
        merged_flat = self._disjoint_merge(pruned, sign, w=w, merge=merging_type)

        direction: TensorDict = TaskVector.unflatten_like(merged_flat, like=base, keys=keys)
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
        keep_ratio: float | None = None,
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        prepared = self.prepare(
            base=base,
            tuned=tuned,
            weights=weights,
            keep_ratio=keep_ratio,
            strict=strict,
            **kwargs,
        )
        return self.apply(prepared, alpha=float(alpha))

    @staticmethod
    def _topk_mask(M: torch.Tensor, topk: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Keep top |x| fraction per row (topk in [0,1] or [0,100]).
        Returns pruned M and boolean mask of kept entries.
        """
        if topk > 1.0:
            topk = topk / 100.0
        topk = float(topk)

        if topk >= 1.0:
            mask = torch.ones_like(M, dtype=torch.bool)
            return M, mask

        _, d = M.shape
        k = max(1, int(d * topk))
        vals, _ = torch.topk(M.abs(), k=k, dim=1, largest=True, sorted=False)
        thr = vals.min(dim=1, keepdim=True).values
        mask = M.abs() >= thr
        return M * mask, mask

    @staticmethod
    def _resolve_sign(M: torch.Tensor) -> torch.Tensor:
        """
        Majority sign per dimension after pruning.
        Zeros get filled with global majority.
        """
        if torch.all(M == 0):
            return torch.ones(M.shape[1], device=M.device, dtype=torch.float32)
        s = torch.sign(M.sum(dim=0))
        global_majority = torch.sign(s.sum())
        global_majority = global_majority if global_majority != 0 else torch.tensor(1.0, device=s.device)
        s[s == 0] = global_majority
        return s

    @staticmethod
    def _disjoint_merge(M: torch.Tensor, ref_sign: torch.Tensor, *, w: torch.Tensor, merge: str) -> torch.Tensor:
        """
        Select entries agreeing with ref_sign and aggregate across rows with weights.
        """
        keep = torch.where(ref_sign.unsqueeze(0) > 0, M > 0, M < 0)
        selected = M * keep

        w_row = w.to(selected.device, selected.dtype).view(-1, 1)
        selected = selected * w_row

        if merge == "mean":
            denom = (keep.to(selected.dtype) * w_row).sum(dim=0).clamp_min(1e-12)
            return selected.sum(dim=0) / denom
        if merge == "sum":
            return selected.sum(dim=0)
        if merge == "max":
            vals, _ = selected.abs().max(dim=0)
            return vals * ref_sign.to(vals.dtype)
        raise ValueError(f"Unknown TIES merge type '{merge}'")


register(TIESMerge())
