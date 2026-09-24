#!/usr/bin/env python
"""Check a set of runs of the configs in this directory against the recorded reference results.

Each run directory is expected to be named after its config (``<config stem>/summary.json``,
which is what ``--local-log-dir <root>/<config stem> --run-name summary`` produces).

For every run found it prints stage 0 (B alone), stage 1 (oracle), stage 2 in cached-feature
space, and the live "rebased" accuracy, then checks:

1. live == cached stage 2, exactly (the correction hook sees what the fit saw);
2. every number equals ``<task>/expected_results.json`` within ``--tol`` (default 0: exact),
   and every expected run is present;
3. with ``--compare <other-root>``, every number equals the other root's run exactly.

Usage:
    python configs/block_ridge_ntk_attention/check_results.py <results-root> [--task snli] [--tol 0] [--compare <root>]
    python configs/block_ridge_ntk_attention/check_results.py <results-root> --write-expected   # record a reference

Exit status 0 when every check passes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ("stage0", "stage1", "stage2_cached", "stage2_live")


def read_run(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    task = summary["tasks"][0]
    diag = summary["steer_diagnostics"][task]
    return {
        "stage0": diag["stage0_test_acc"],
        "stage1": diag["stage1_test_acc"],
        "stage2_cached": diag["stage2_test_acc"],
        "stage2_live": summary["test_results"]["per_task_rebased"][task],
    }


def collect(root: Path, task: str) -> dict[str, dict[str, float]]:
    runs = {}
    for config in sorted((HERE / task).glob("*.json")):
        if config.name == "expected_results.json":
            continue
        got = read_run(root / config.stem)
        if got is not None:
            runs[config.stem] = got
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", default="snli")
    parser.add_argument("--tol", type=float, default=0.0, help="allowed |difference| from the expected results")
    parser.add_argument("--compare", type=Path, default=None, help="another results root that must match exactly")
    parser.add_argument("--write-expected", action="store_true", help="record these runs as the reference")
    args = parser.parse_args()

    runs = collect(args.root, args.task)
    if not runs:
        print(f"no runs of {HERE / args.task}/*.json found under {args.root}")
        return 1
    ok = True

    print(f"{'run':62s} " + " ".join(f"{k:>14s}" for k in KEYS))
    for name, got in runs.items():
        print(f"{name:62s} " + " ".join(f"{got[k]:14.6f}" for k in KEYS))
    cells: dict[str, list[dict[str, float]]] = {}
    for name, got in runs.items():
        cells.setdefault(name.rsplit("_seed", 1)[0], []).append(got)
    print("\nmean ± sd over seeds (stage 2, live):")
    for cell, rs in cells.items():
        vals = [r["stage2_live"] for r in rs]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {cell:56s} {st.mean(vals):.4f} ± {sd:.4f}  (n={len(vals)})")

    bad_live = [n for n, r in runs.items() if r["stage2_live"] != r["stage2_cached"]]
    print(f"\n[check] live == cached stage 2 in every run: {'OK' if not bad_live else 'FAIL ' + str(bad_live)}")
    ok &= not bad_live

    expected_path = HERE / args.task / "expected_results.json"
    if args.write_expected:
        expected_path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
        print(f"[write] recorded {len(runs)} runs as the reference in {expected_path}")
    elif expected_path.is_file():
        expected = json.loads(expected_path.read_text())
        missing = sorted(set(expected) - set(runs))
        diffs = {
            n: {k: runs[n][k] - expected[n][k] for k in KEYS if abs(runs[n][k] - expected[n][k]) > args.tol}
            for n in expected
            if n in runs
        }
        diffs = {n: d for n, d in diffs.items() if d}
        print(f"[check] matches {expected_path.name} (tol={args.tol}): "
              f"{'OK' if not diffs else 'FAIL ' + json.dumps(diffs)}")
        if missing:
            print(f"[check] expected runs missing: {missing}")
        ok &= not diffs and not missing
    else:
        print(f"[check] no {expected_path.name} to compare against")

    if args.compare is not None:
        other = collect(args.compare, args.task)
        common = sorted(set(runs) & set(other))
        diffs = {n: {k: runs[n][k] - other[n][k] for k in KEYS if runs[n][k] != other[n][k]} for n in common}
        diffs = {n: d for n, d in diffs.items() if d}
        print(f"[check] identical to {args.compare} on {len(common)} common runs: "
              f"{'OK' if not diffs else 'FAIL ' + json.dumps(diffs)}")
        ok &= not diffs and bool(common)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
