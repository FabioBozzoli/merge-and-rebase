"""Aggregate the segment-pooled vs global linear probe (configs/text_probe_segments) and set it against the
steer_text study cells (configs/ntk_rebasin_study).

    python scripts/analyze_text_probe_segments.py [--probe-root DIR] [--study-root DIR] [--out-dir DIR]

Writes to ``--out-dir`` (default ``<probe-root>/_analysis``):

- ``probe_runs.csv``: one row per (target, pooling, task, few_shot, seed): ``acc`` the probed test accuracy
  (lr chosen on val), ``init`` the nearest-mean head before training, ``lr`` the selected lr.
- ``probe_table.md``: mean ± std over seeds per target × task × few_shot, probe global vs segments.
- ``probe_paired.md``: paired (task, few_shot, seed) comparisons per target: probe segments - global, and
  every steer cell - probe (global and segments). The probe's support is steer_text's own ``_few_shot`` draw
  (``linear_probe_support="steer"``), so at the same task/few_shot/seed both saw the same examples.
- ``overview.md``: mean over tasks and seeds per target × few_shot for every method, the one-glance table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ntk_rebasin_study import _markdown, load_runs  # noqa: E402

PROBE_RE = re.compile(r"^(?P<pooling>global|segments)_fs(?P<few_shot>\d+)_seed(?P<seed>\d+)$")
STEER_CELLS = (
    "lin_global_ridge", "lin_global_ridge_seg", "lin_global_mlp", "lin_global_mlp_seg",
    "nl_global_ridge", "nl_global_ridge_seg", "nl_global_mlp", "nl_global_mlp_seg", "lin_block_ridge_attn_seg",
)
KEYS = ["target", "task", "few_shot", "seed"]


def load_probe(root: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(root.glob("*/*/summary.json")):
        m = PROBE_RE.match(summary.parent.name)
        target = summary.parent.parent.name
        if m is None or target.startswith("_"):
            continue
        d = json.loads(summary.read_text())
        if "test_results" not in d:  # a failed run leaves a summary with only run_logging
            print(f"skipping {summary.parent}: no test_results (failed run)")
            continue
        tr = d["test_results"]
        for task, acc in tr["per_task_absolute_accuracy"].items():
            rows.append({
                "target": target, "method": f"probe_{m['pooling']}", "task": task,
                "few_shot": int(m["few_shot"]), "seed": int(m["seed"]), "acc": acc,
                "init": tr["per_task_baseline_accuracy"][task], "lr": d["per_task_selected_lr"][task],
            })
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame) -> str:
    g = df.groupby(["target", "task", "few_shot", "method"])["acc"].agg(["mean", "std", "count"]).reset_index()
    g["v"] = g.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f} (n={int(r['count'])})", axis=1)
    out = []
    for target, gt in g.groupby("target"):
        piv = gt.pivot_table(index=["task", "few_shot"], columns="method", values="v", aggfunc="first")
        out.append(f"## {target} — probe test accuracy\n\n{_markdown(piv)}\n")
    return "\n".join(out)


def paired_rows(df: pd.DataFrame, pairs: list[tuple[str, str, str]], rng: np.random.Generator) -> list[str]:
    lines = []
    for target in sorted(df.target.unique()):
        dt = df[df.target == target]
        for label, a, b in pairs:
            m = dt[dt.method == a][KEYS + ["acc"]].merge(dt[dt.method == b][KEYS + ["acc"]], on=KEYS, suffixes=("_a", "_b"))
            if m.empty:
                continue
            diff = (m["acc_a"] - m["acc_b"]).to_numpy()
            boot = rng.choice(diff, size=(10000, diff.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else float("nan")
            lines.append(f"| {target} | {label} | {diff.size} | {diff.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.2e} | "
                         f"{int((diff > 0).sum())} / {int((diff < 0).sum())} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-root", type=Path, default=Path("/work/intesasanpaolo_phd/merge-and-rebase/text_probe_segments"))
    p.add_argument("--study-root", type=Path, default=Path("/work/intesasanpaolo_phd/merge-and-rebase/ntk_rebasin_study"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = p.parse_args()
    out = args.out_dir or args.probe_root / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    probe = load_probe(args.probe_root)
    if probe.empty:
        raise SystemExit(f"no probe runs under {args.probe_root}")
    probe.sort_values(["target", "method", "task", "few_shot", "seed"]).to_csv(out / "probe_runs.csv", index=False)
    (out / "probe_table.md").write_text(mean_std(probe))

    study = load_runs(args.study_root)
    study = study[study.cell.isin(STEER_CELLS)].rename(columns={"cell": "method"})
    zero_shot = study.groupby(["target", "task", "few_shot", "seed"])["baseline"].first().reset_index()
    zero_shot = zero_shot.rename(columns={"baseline": "acc"}).assign(method="zero_shot_B")
    both = pd.concat([probe[KEYS + ["method", "acc"]], study[KEYS + ["method", "acc"]], zero_shot[KEYS + ["method", "acc"]]])

    rng = np.random.default_rng(args.seed)
    pairs = [("probe segments vs probe global", "probe_segments", "probe_global")]
    pairs += [(f"{c} vs probe {pool}", c, f"probe_{pool}") for c in STEER_CELLS for pool in ("global", "segments")]
    header = ["## Paired comparisons — test accuracy (difference a - b over task × few_shot × seed)\n",
              "| target | comparison | n pairs | mean diff | 95% bootstrap CI | Wilcoxon p | wins a / b |",
              "|---|---|---|---|---|---|---|"]
    (out / "probe_paired.md").write_text("\n".join(header + paired_rows(both, pairs, rng)) + "\n")

    # Overview: mean over tasks and seeds. Only (target, task, few_shot, seed) cells every method has, so the
    # columns average the same examples.
    complete = both.groupby(KEYS)["method"].nunique()
    full = complete[complete == both.method.nunique()].index
    sub = both.set_index(KEYS).loc[lambda d: d.index.isin(full)].reset_index()
    if sub.empty:
        (out / "overview.md").write_text("No (target, task, few_shot, seed) has a run of every method yet.\n")
    else:
        ov = sub.groupby(["target", "few_shot", "method"])["acc"].mean().unstack("method")
        order = ["zero_shot_B", "probe_global", "probe_segments", *STEER_CELLS]
        ov = ov[[c for c in order if c in ov.columns]].map(lambda v: f"{v:.4f}")
        ov["n (task×seed)"] = sub[KEYS].drop_duplicates().groupby(["target", "few_shot"]).size()
        (out / "overview.md").write_text(
            "## Mean test accuracy over tasks and seeds (only cells where every method has a run)\n\n"
            + _markdown(ov) + "\n"
        )
    print(f"{len(probe)} probe rows, {len(study)} study rows\nwritten to {out}")


if __name__ == "__main__":
    main()
