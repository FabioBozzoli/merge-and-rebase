from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

from merge_and_rebase.finetune.sharded_adamw import ShardedMasterAdamW
from merge_and_rebase.utils.distributed import DistInfo

_SHAPES = [(5, 3), (7,), (2, 2, 3)]  # 34 elements: not a multiple of 2, so the flat buffer is padded


def _init_params() -> list[torch.nn.Parameter]:
    g = torch.Generator().manual_seed(0)
    return [torch.nn.Parameter((0.1 * torch.randn(s, generator=g)).bfloat16()) for s in _SHAPES]


def _grads(rank: int, step: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1000 * step + rank)
    return [torch.randn(s, generator=g).bfloat16() for s in _SHAPES]


def _reference(world: int, steps: int, scale: float, max_norm: float) -> list[torch.Tensor]:
    """fp32 AdamW on fp32 copies, grads summed over ranks then scaled and clipped."""
    params = [torch.nn.Parameter(p.detach().float()) for p in _init_params()]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.1)
    for step in range(steps):
        for i, p in enumerate(params):
            p.grad = sum(_grads(r, step)[i].float() for r in range(world)) * scale
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        opt.step()
    return [p.detach() for p in params]


def _train(info: DistInfo, steps: int, scale: float, max_norm: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    params = _init_params()
    opt = ShardedMasterAdamW(params, info, lr=1e-2, weight_decay=0.1, chunk_numel=4)  # several chunks per shard
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for p, g in zip(params, _grads(info.rank, step), strict=True):
            p.grad.add_(g)  # what autograd's accumulation does: in place, into the flat buffer
        opt.reduce_and_clip_(scale=scale, max_norm=max_norm)
        opt.step()
    return [p.detach().clone() for p in params], opt.master.detach().clone()


def test_one_rank_matches_fp32_adamw_and_rounds_params_from_the_master() -> None:
    ref = _reference(world=1, steps=4, scale=0.5, max_norm=1.0)
    params, master = _train(DistInfo(), steps=4, scale=0.5, max_norm=1.0)
    flat_ref = torch.cat([r.reshape(-1) for r in ref])
    torch.testing.assert_close(master[: flat_ref.numel()], flat_ref, rtol=1e-6, atol=1e-7)
    for p, r in zip(params, ref, strict=True):
        assert p.dtype == torch.bfloat16
        assert torch.equal(p, r.bfloat16())


def test_load_full_sets_params_and_only_this_ranks_master_shard() -> None:
    values = [torch.randn(s) for s in _SHAPES]
    flat = torch.cat([v.reshape(-1) for v in values])
    for rank in (0, 1):  # no collectives in load_full_: a 2-rank layout works without a process group
        params = _init_params()
        opt = ShardedMasterAdamW(params, DistInfo(rank=rank, world_size=2), lr=1e-2)
        opt.load_full_(values)
        assert all(torch.equal(p, v.bfloat16()) for p, v in zip(params, values, strict=True))
        shard = torch.cat([flat, torch.zeros(2 * opt.shard_numel - flat.numel())])[rank * opt.shard_numel :][: opt.shard_numel]
        assert torch.equal(opt.master.detach(), shard)  # fp32, not rounded through the bf16 params
    got = ShardedMasterAdamW(_init_params(), DistInfo(), lr=1e-2)
    got.load_full_(values)
    full = got.full_master()
    assert all(torch.equal(f, v) for f, v in zip(full, values, strict=True))
    got.master.data.zero_()
    assert torch.equal(full[0], values[0]), "full_master must copy, not alias the master"


def test_state_dict_round_trip_continues_like_an_uninterrupted_run() -> None:
    def run(opt, params, steps):
        for step in steps:
            opt.zero_grad()
            for p, g in zip(params, _grads(0, step), strict=True):
                p.grad.add_(g)
            opt.reduce_and_clip_(scale=0.5, max_norm=1.0)
            opt.step()

    straight = _init_params()
    a = ShardedMasterAdamW(straight, DistInfo(), lr=1e-2, weight_decay=0.1)
    run(a, straight, range(4))
    first = _init_params()
    b = ShardedMasterAdamW(first, DistInfo(), lr=1e-2, weight_decay=0.1)
    run(b, first, range(2))
    second = _init_params()
    c = ShardedMasterAdamW(second, DistInfo(), lr=1e-2, weight_decay=0.1)
    c.load_state_dict(b.state_dict())
    run(c, second, range(2, 4))
    assert torch.equal(c.master, a.master)
    assert all(torch.equal(p, q) for p, q in zip(second, straight, strict=True))


def _two_rank_worker(rank: int, world: int, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29541")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        params, _ = _train(DistInfo(rank=rank, world_size=world), steps=4, scale=0.25, max_norm=1.0)
        torch.save(params, Path(out) / f"rank{rank}.pt")
        opt = ShardedMasterAdamW(_init_params(), DistInfo(rank=rank, world_size=world), lr=1e-2, chunk_numel=4)
        opt.load_full_([p.float() * (rank + 1) for p in params])  # each rank's master shard comes from its own values
        full = opt.full_master()
        assert (full is None) == (rank != 0)
        if full is not None:
            torch.save(full, Path(out) / "full_master.pt")
        # state_dict gathers whole tensors on rank 0; load_state_dict hands each rank back its own shard.
        trained = ShardedMasterAdamW(_init_params(), DistInfo(rank=rank, world_size=world), lr=1e-2, chunk_numel=4)
        trained.load_full_([p.float() for p in params])
        trained.master.grad.normal_()
        trained.step()
        sd = trained.state_dict()
        sd = [sd]
        dist.broadcast_object_list(sd, src=0)
        again = ShardedMasterAdamW(_init_params(), DistInfo(rank=rank, world_size=world), lr=1e-2, chunk_numel=4)
        again.load_state_dict(sd[0])
        assert torch.equal(again.master, trained.master)
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(again.inner.state[again.master][k], trained.inner.state[trained.master][k])
        if rank == 0:
            torch.save(sd[0], Path(out) / "state_dict.pt")
    finally:
        dist.destroy_process_group()


def test_two_ranks_match_one_process_on_the_summed_gradient(tmp_path) -> None:
    import torch.multiprocessing as mp

    mp.get_context("spawn")
    mp.spawn(_two_rank_worker, args=(2, str(tmp_path)), nprocs=2)
    ref = _reference(world=2, steps=4, scale=0.25, max_norm=1.0)
    for rank in range(2):
        params = torch.load(tmp_path / f"rank{rank}.pt")
        for p, r in zip(params, ref, strict=True):
            assert torch.equal(p, r.bfloat16()), "every rank must hold the same params as the fp32 reference"
    # full_master gathers shard j from rank j: rank 0 loaded params x 1, rank 1 params x 2.
    s = ShardedMasterAdamW(_init_params(), DistInfo(rank=0, world_size=2), lr=1e-2).shard_numel
    p0 = torch.cat([p.float().reshape(-1) for p in torch.load(tmp_path / "rank0.pt")])
    expected = torch.cat([p0[:s], 2 * p0[s:]])
    got = torch.cat([t.reshape(-1) for t in torch.load(tmp_path / "full_master.pt")])
    assert torch.equal(got, expected)
    # A 2-rank state_dict resumes on 1 rank: the whole master, unpadded.
    sd = torch.load(tmp_path / "state_dict.pt")
    one = ShardedMasterAdamW(_init_params(), DistInfo(), lr=1e-2)
    one.load_state_dict(sd)
    assert torch.equal(one.master.detach(), sd["master"]) and sd["master"].numel() == p0.numel()
    assert torch.equal(one.inner.state[one.master]["exp_avg_sq"], sd["exp_avg_sq"])
