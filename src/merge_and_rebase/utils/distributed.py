"""Single-node (or multi-node) data parallelism for the finetune entrypoints.

Deliberately not DDP: `finetune.forward_mode` replaces ``model.forward`` with a
linearized closure, and DDP's forward hooks/bucketing assume the module's own
forward and reduce *all* grads. Here only the LoRA factors are trainable (a few
tens of MB), so one all-reduce per optimizer step after the local accumulation
window is both simpler and cheaper.

Launched by srun (one task per GPU); torchrun's env vars are honoured too.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.distributed as dist


class DistInfo:
    """rank/world/local_rank, with world_size == 1 when not distributed."""

    def __init__(self, rank: int = 0, world_size: int = 1, local_rank: int = 0) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DistInfo(rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank})"


def _env_int(*names: str, default: int = 0) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            return int(raw)
    return int(default)


def init_distributed(*, backend: str | None = None, device: str = "cuda") -> DistInfo:
    """Join the process group described by the environment, if there is one.

    Returns a world_size == 1 DistInfo (and initializes nothing) for a plain
    single-process run, so callers need no branching at the call site.
    """
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    if world_size <= 1:
        return DistInfo()

    rank = _env_int("RANK", "SLURM_PROCID", default=0)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", str(20000 + _env_int("SLURM_JOB_ID", default=0) % 20000))
        if backend is None:
            backend = "nccl" if (device.startswith("cuda") and torch.cuda.is_available()) else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def shutdown_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor, info: DistInfo) -> torch.Tensor:
    """In-place SUM across ranks (no-op when not distributed)."""
    if info.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_gradients_(params: Iterable[torch.nn.Parameter], info: DistInfo, *, scale: float = 1.0) -> None:
    """Sum the grads of `params` across ranks (one flat all-reduce), then scale.

    Call it once per optimizer step, after the local accumulation window and
    before clipping. The window holds *unnormalized* summed-loss gradients, so
    SUM across ranks followed by `scale = loss_scale / total_tokens` is the exact
    token-weighted mean, whatever the split of rows across ranks and batches.
    Every rank then holds the same gradient, so the parameters stay identical and
    only rank 0 has to write checkpoints.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    if info.enabled and dist.is_initialized():
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for grad, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True):
            grad.copy_(reduced)
    if scale != 1.0:
        torch._foreach_mul_(grads, scale)


def broadcast_flag(flag: bool, info: DistInfo, *, device: torch.device | str = "cpu") -> bool:
    """True if `flag` is set on ANY rank, so every rank takes the same branch.

    Used for the pre-walltime SIGUSR1: Slurm signals each task independently and
    a checkpoint write must not be a collective that only some ranks enter.
    """
    if not info.enabled or not dist.is_initialized():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())
