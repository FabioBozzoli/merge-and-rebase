"""AdamW with an fp32 master copy and optimizer state sharded across data-parallel ranks (ZeRO-2 style).

For full-rank linearized training: the trainable dW tensors are bf16 (they are the jvp tangents), but per-step
updates of size lr ~ 1e-5 would round away if added to a bf16 tensor, and fp32 master + Adam m, v for 3B
parameters (36 GiB) do not fit on one 45 GB GPU next to the model. So:

  * every trainable param becomes a view of one flat bf16 buffer, and its ``.grad`` a view of one flat bf16
    gradient buffer that autograd accumulates into in place;
  * at the step, the gradient is summed across ranks in fp32 with a chunked reduce-scatter, so each rank holds
    only its 1/world_size shard;
  * each rank runs ``torch.optim.AdamW`` on its fp32 master shard, then the bf16 shards are all-gathered back
    into the flat parameter buffer, so every rank again holds the same full dW.

``full_master`` / ``load_full_`` move the fp32 dW in and out (the best and per-epoch dW files, and warm starts
from them). ``state_dict`` / ``load_state_dict`` hold the whole unsharded fp32 master and Adam moments, so a
resume may use a different number of ranks.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from merge_and_rebase.utils.distributed import DistInfo


class ShardedMasterAdamW:
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        info: DistInfo,
        *,
        lr: float,
        weight_decay: float = 0.0,
        chunk_numel: int = 1 << 26,
    ) -> None:
        if not params:
            raise ValueError("ShardedMasterAdamW needs at least one parameter.")
        self.info = info
        self.params = list(params)
        self.chunk_numel = int(chunk_numel)
        world = max(1, info.world_size)
        total = sum(p.numel() for p in self.params)
        self.shard_numel = math.ceil(total / world)
        ref = self.params[0]
        self.flat = torch.zeros(self.shard_numel * world, dtype=ref.dtype, device=ref.device)
        self.flat_grad = torch.zeros_like(self.flat)
        offset = 0
        for p in self.params:
            n = p.numel()
            self.flat[offset : offset + n].copy_(p.detach().reshape(-1))
            p.data = self.flat[offset : offset + n].view_as(p)
            p.grad = self.flat_grad[offset : offset + n].view_as(p)
            offset += n

        lo = info.rank * self.shard_numel
        self._shard = slice(lo, lo + self.shard_numel)
        self.master = torch.nn.Parameter(self.flat[self._shard].float())
        self.master.grad = torch.zeros_like(self.master)
        self.inner = torch.optim.AdamW(
            [self.master], lr=lr, weight_decay=weight_decay, fused=self.master.is_cuda or None
        )

    @property
    def param_groups(self) -> list[dict]:
        return self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: ARG002 - grads are views, never dropped
        self.flat_grad.zero_()

    def reduce_and_clip_(self, *, scale: float, max_norm: float) -> float:
        """Sum the local grads across ranks into this rank's fp32 shard, scale, clip by global norm.

        Same semantics as ``reduce_gradients_`` followed by ``clip_grad_norm_``. Returns the pre-clip norm.
        """
        grad = self.master.grad
        if self.info.enabled and dist.is_initialized():
            world, s = self.info.world_size, self.shard_numel
            for a in range(0, s, self.chunk_numel):
                b = min(a + self.chunk_numel, s)
                # piece j of the input is shard j's [a, b): reduce-scatter hands rank r the sum of piece r.
                pieces = torch.stack([self.flat_grad[j * s + a : j * s + b] for j in range(world)]).float()
                dist.reduce_scatter_tensor(grad[a:b], pieces.reshape(-1), op=dist.ReduceOp.SUM)
        else:
            grad.copy_(self.flat_grad)
        grad.mul_(scale)
        sq = grad.pow(2).sum()
        if self.info.enabled and dist.is_initialized():
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        norm = float(sq.sqrt())
        if max_norm > 0:
            grad.mul_(min(1.0, max_norm / (norm + 1e-6)))  # clip_grad_norm_'s coefficient
        return norm

    def step(self) -> None:
        self.inner.step()
        self.flat[self._shard].copy_(self.master.detach())
        if self.info.enabled and dist.is_initialized():
            dist.all_gather_into_tensor(self.flat, self.flat[self._shard])

    def _gather(self, shard: torch.Tensor) -> torch.Tensor | None:
        """The concatenation of every rank's ``shard`` on CPU, trimmed to the params' numel, on rank 0. Collective."""
        s = self.shard_numel
        if self.info.enabled and dist.is_initialized():
            world = self.info.world_size
            flat = torch.empty(world, s, dtype=shard.dtype) if self.info.is_main else None
            for a in range(0, s, self.chunk_numel):
                b = min(a + self.chunk_numel, s)
                buf = torch.empty(world * (b - a), dtype=shard.dtype, device=shard.device)
                dist.all_gather_into_tensor(buf, shard.detach()[a:b].contiguous())
                if flat is not None:
                    flat[:, a:b] = buf.view(world, b - a).cpu()
            if flat is None:
                return None
            flat = flat.view(-1)
        else:
            flat = shard.detach().to("cpu", copy=True)
        return flat[: sum(p.numel() for p in self.params)]

    def full_master(self) -> list[torch.Tensor] | None:
        """Every param's fp32 master value on CPU, in ``params`` order, on rank 0 (None elsewhere). Collective."""
        flat = self._gather(self.master)
        if flat is None:
            return None
        out, off = [], 0
        for p in self.params:
            out.append(flat[off : off + p.numel()].view_as(p))
            off += p.numel()
        return out

    def load_full_(self, values: list[torch.Tensor]) -> None:
        """Set every param (rounded to its dtype) and this rank's fp32 master shard from ``values`` (``params`` order)."""
        lo, hi = self._shard.start, self._shard.stop
        off = 0
        with torch.no_grad():
            for p, v in zip(self.params, values, strict=True):
                n = p.numel()
                v = v.reshape(-1).to(device=self.master.device, dtype=self.master.dtype)
                p.view(-1).copy_(v)
                a, b = max(off, lo), min(off + n, hi)
                if a < b:
                    self.master[a - lo : b - lo] = v[a - off : b - off]
                off += n

    def state_dict(self) -> dict | None:
        """Unsharded fp32 master, Adam moments and step on rank 0 (None elsewhere). Collective."""
        st = self.inner.state.get(self.master, {})
        out = {k: self._gather(st[k]) if k in st else None for k in ("exp_avg", "exp_avg_sq")}
        out["master"] = self._gather(self.master)
        if out["master"] is None:
            return None
        out["step"] = float(st["step"]) if "step" in st else 0.0
        return out

    def load_state_dict(self, state: dict) -> None:
        """Inverse of ``state_dict``: this rank's shard of each tensor; the bf16 params from the master."""
        lo, hi = self._shard.start, self._shard.stop

        def shard(flat: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(self.master)  # the padding past the last param stays zero
            n = max(0, min(hi, flat.numel()) - lo)
            out[:n] = flat[lo : lo + n].to(out.device, out.dtype)
            return out

        sizes = [p.numel() for p in self.params]
        self.load_full_(list(state["master"].split(sizes)))
        if state["exp_avg"] is not None:
            # AdamW keeps step as a float32 tensor: on the param's device when fused, else on CPU
            dev = self.master.device if self.inner.defaults.get("fused") else "cpu"
            step = torch.tensor(state["step"], dtype=torch.float32, device=dev)
            self.inner.state[self.master] = {
                "step": step,
                "exp_avg": shard(state["exp_avg"]),
                "exp_avg_sq": shard(state["exp_avg_sq"]),
            }
