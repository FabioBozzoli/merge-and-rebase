from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class DAREMerge:
    name: str = "dare_merge"

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
        method_params = get_method_params(kwargs)

        if "drop_rate" in method_params:
            drop_rate = float(method_params["drop_rate"])
        elif "p" in method_params:
            drop_rate = float(method_params["p"])
        elif "keep_ratio" in method_params:
            drop_rate = 1.0 - float(method_params["keep_ratio"])
        else:
            drop_rate = 0.9

        seed_val = method_params.get("seed", None)
        seed = None if seed_val is None else int(seed_val)
        rescale = bool(method_params.get("rescale", True))

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)
        flat = TaskVector.stack_flattened(deltas, keys, dtype=torch.float32)
        merged_flat = self._dare_delta(flat, w=w, drop_rate=drop_rate, rescale=rescale, seed=seed)
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
    def _dare_delta(
        M: torch.Tensor, *, w: torch.Tensor, drop_rate: float, rescale: bool, seed: int | None
    ) -> torch.Tensor:
        """
        DARE on flattened task vectors [N, D]:
          1) random unstructured sparsification per task
          2) optional scaling by 1/(1-p) to preserve expectation
          3) weighted aggregation across tasks
        """
        if M.ndim != 2:
            raise ValueError(f"Expected M to have shape [N, D], got {tuple(M.shape)}")

        if M.shape[0] == 0:
            return torch.empty((0,), dtype=M.dtype, device=M.device)

        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must satisfy 0 <= drop_rate < 1.")

        if w.numel() != M.shape[0]:
            raise ValueError(f"weights length must match row count in M. got {w.numel()} vs {M.shape[0]}")

        keep_prob = 1.0 - float(drop_rate)
        if keep_prob == 1.0:
            sparse = M
        else:
            gen = None
            if seed is not None:
                gen = torch.Generator(device=M.device)
                gen.manual_seed(seed)
            mask = (torch.rand(M.shape, device=M.device, generator=gen) < keep_prob).to(M.dtype)
            sparse = M * mask
            if rescale:
                sparse = sparse / keep_prob

        return (sparse * w.to(device=M.device, dtype=M.dtype).view(-1, 1)).sum(dim=0)


register(DAREMerge())
