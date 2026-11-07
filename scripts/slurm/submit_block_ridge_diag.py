#!/usr/bin/env python
"""Rerun block_ridge grid cells with their fitted artifacts kept, for diagnose_block_ridge.py.

The grid drops ``<task>_steer_artifacts.pt`` for block_ridge (~218 MB a run), and
``grid.py submit`` only ever submits *missing* cells, so the completed ones cannot
be rerun through it. This resubmits the cells of one grid group with the exact
overrides and resources ``grid.py`` would use -- ``overrides_for`` and
``resources_for`` are imported, not copied -- but into a separate results root,
so nothing here can overwrite a finished grid run.

The feature cache is warm for every ntk task, so each rerun only refits Stage 1/2
and re-evaluates; the fit uses the same ``selected`` support as the original cell
because ``_few_shot`` is deterministic in (labels, few_shot, seed).

Usage:
    python scripts/slurm/submit_block_ridge_diag.py --group ntk-br-fs200 [--task mnli] [--seed 33] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

DIAG_ROOT = grid.WORK / "t5enc_block_ridge_diag"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True)
    p.add_argument("--task", action="append", choices=grid.TASKS, help="Restrict to these tasks (repeatable).")
    p.add_argument("--seed", action="append", type=int, help="Restrict to these seeds (repeatable).")
    p.add_argument("--results-root", type=Path, default=DIAG_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    group = next((g for g in grid.GROUPS if g.name == args.group), None)
    if group is None:
        print(f"error: unknown group {args.group!r}", file=sys.stderr)
        return 2
    if group.strategy != "block_ridge":
        print(f"error: group {group.name} is {group.strategy}, not block_ridge", file=sys.stderr)
        return 2
    if args.results_root.resolve() == grid.ARMS[group.arm].results_root.resolve():
        print("error: --results-root is the grid's own root; pick another.", file=sys.stderr)
        return 2

    cells = [
        c for c in grid.all_cells()
        if c.group == group.name
        and (not args.task or c.task in args.task)
        and (not args.seed or c.seed in args.seed)
    ]
    arm = grid.ARMS[group.arm]
    cold = sorted({c.task for c in cells if not grid.cache_is_warm(arm, c.task)})
    if cold:
        # A cold cache would make these reruns *write* features, racing the grid.
        print(f"error: cold feature cache for {', '.join(cold)}; these reruns are meant to be cache hits.", file=sys.stderr)
        return 1

    print(f"group {group.name}: {len(cells)} cells -> {args.results_root}{' (dry run)' if args.dry_run else ''}")
    for cell in cells:
        mem, walltime = grid.resources_for(cell, cold=False)
        env = {
            **os.environ,
            "RESULTS_ROOT": str(args.results_root),
            "PARTITION": "all_usr_prod",
            "ACCOUNT": "intesasanpaolo_phd",
            "GRES": "gpu:1",
            "CPUS": str(arm.cpus),
            "MEM": mem,
            "TIME": walltime,
            "ENTRYPOINT": arm.entrypoint,
            "DRY_RUN": "1" if args.dry_run else "0",
        }
        # save_artifacts=True leaves save_steer_artifacts_dir unset, so
        # submit_text_rebase.sh defaults it into the experiment directory.
        cmd = [str(grid.REPO / "scripts/slurm/submit_text_rebase.sh"), cell.name, arm.base_config]
        cmd += grid.overrides_for(cell, save_artifacts=True)
        result = subprocess.run(cmd, env=env, cwd=grid.REPO, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  FAILED {cell.name}\n{result.stdout}{result.stderr}", file=sys.stderr)
            continue
        job = next((ln.split()[-1] for ln in result.stdout.splitlines() if ln.startswith("submitted job ")), "dry-run")
        print(f"  [{mem} {walltime}] {cell.name} -> {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
