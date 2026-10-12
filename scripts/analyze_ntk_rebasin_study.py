"""Aggregate the NTK-vs-nonlinear steer_text study (configs/ntk_rebasin_study).

    python scripts/analyze_ntk_rebasin_study.py [--root /work/.../ntk_rebasin_study] [--out-dir DIR]

Reads every ``<root>/<target>/<task>/<cell>_fs<k>_seed<s>/summary.json`` and writes to ``--out-dir``
(default ``<root>/_analysis``):

- ``runs.csv``: one row per run. ``acc`` is the rebased test accuracy, ``baseline`` the target zero-shot
  accuracy (nearest-mean head, no correction), ``source_acc`` Stage-1 test accuracy, i.e. A's accuracy as
  reconstructed in the cached feature space of that arm (linearized for ``lin``, plain forward for ``nl``),
  ``gap_recovered`` = (acc - baseline) / (source_acc - baseline).
- ``table_mean_std.md``: mean ± std over seeds per target × task × few_shot × cell.
- ``paired_tests.md``: paired comparisons over (task, few_shot, seed) per target, with the mean difference,
  a 95% bootstrap CI and a two-sided Wilcoxon signed-rank test, for lin vs nl (global ridge, global MLP) and
  for block ridge vs global ridge/MLP within ``lin``. Seeds are not independent across cells (same support
  draw), which is exactly why the tests are paired.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUN_RE = re.compile(r"^(?P<cell>.+)_fs(?P<few_shot>\d+)_seed(?P<seed>\d+)$")
COMPARISONS = (
    # (label, metric column, cell a, cell b): difference is a - b
    ("lin vs nl, global ridge", "lin_global_ridge", "nl_global_ridge"),
    ("lin vs nl, global MLP", "lin_global_mlp", "nl_global_mlp"),
    ("lin: block ridge (attn) vs global ridge", "lin_block_ridge_attn_seg", "lin_global_ridge"),
    ("lin: block ridge (attn) vs global MLP", "lin_block_ridge_attn_seg", "lin_global_mlp"),
    # Segment-pooled Stage-2 input vs the same cell on the global feature (head and Stage 1 identical).
    ("lin: global ridge, segments vs global input", "lin_global_ridge_seg", "lin_global_ridge"),
    ("lin: global MLP, segments vs global input", "lin_global_mlp_seg", "lin_global_mlp"),
    ("nl: global ridge, segments vs global input", "nl_global_ridge_seg", "nl_global_ridge"),
    ("nl: global MLP, segments vs global input", "nl_global_mlp_seg", "nl_global_mlp"),
    ("lin: block ridge (attn) vs global ridge seg", "lin_block_ridge_attn_seg", "lin_global_ridge_seg"),
)


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(root.glob("*/*/*/summary.json")):
        run_dir = summary.parent
        m = RUN_RE.match(run_dir.name)
        if m is None or run_dir.parts[-3].startswith("_"):
            continue
        task = run_dir.parent.name
        d = json.loads(summary.read_text())
        if "test_results" not in d:  # a failed run leaves a summary with only run_logging
            print(f"skipping {run_dir}: no test_results (failed run)")
            continue
        diag = d.get("steer_diagnostics", {}).get(task, {})
        tr = d["test_results"]
        src_ctrl = d.get("source_finetuned_control") or [{}]
        rows.append(
            {
                "target": run_dir.parts[-3],
                "task": task,
                "cell": m["cell"],
                "arm": m["cell"].split("_", 1)[0],
                "few_shot": int(m["few_shot"]),
                "seed": int(m["seed"]),
                "acc": tr["per_task_absolute_accuracy"][task],
                "baseline": tr["per_task_baseline_accuracy"][task],
                "source_acc": diag.get("stage1_test_acc"),
                "stage2_cached_acc": diag.get("stage2_test_acc"),
                "source_finetuned_acc": src_ctrl[0].get("source_finetuned_accuracy"),
                "prepare_seconds": d.get("transport_timings", {}).get(task, {}).get("prepare_seconds"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap_recovered"] = (df["acc"] - df["baseline"]) / (df["source_acc"] - df["baseline"])
    return df


def _markdown(df: pd.DataFrame) -> str:
    """``df.to_markdown()`` without the optional ``tabulate`` dependency."""
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def mean_std_table(df: pd.DataFrame, metric: str) -> str:
    g = df.groupby(["target", "task", "few_shot", "cell"])[metric].agg(["mean", "std", "count"]).reset_index()
    g["cell_value"] = g.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f} (n={int(r['count'])})", axis=1)
    out = []
    for target, gt in g.groupby("target"):
        piv = gt.pivot_table(index=["task", "few_shot"], columns="cell", values="cell_value", aggfunc="first")
        base = df[df.target == target].groupby(["task", "few_shot"])["baseline"].mean().rename("zero-shot B")
        piv = piv.join(base.map(lambda v: f"{v:.4f}"))
        out.append(f"## {target} — {metric}\n\n{_markdown(piv)}\n")
    return "\n".join(out)


def paired(df: pd.DataFrame, metric: str, rng: np.random.Generator, n_boot: int = 10000) -> str:
    lines = [f"## Paired comparisons — {metric}\n",
             "| target | comparison | n pairs | mean diff | 95% bootstrap CI | Wilcoxon p | wins a / b |",
             "|---|---|---|---|---|---|---|"]
    keys = ["target", "task", "few_shot", "seed"]
    for target in sorted(df.target.unique()):
        dt = df[df.target == target]
        for label, a, b in COMPARISONS:
            m = dt[dt.cell == a][keys + [metric]].merge(dt[dt.cell == b][keys + [metric]], on=keys, suffixes=("_a", "_b"))
            m = m.dropna()
            if m.empty:
                continue
            diff = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
            boot = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else float("nan")
            lines.append(
                f"| {target} | {label} | {diff.size} | {diff.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.2e} | "
                f"{int((diff > 0).sum())} / {int((diff < 0).sum())} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/work/intesasanpaolo_phd/merge-and-rebase/ntk_rebasin_study"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = p.parse_args()
    out = args.out_dir or args.root / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.root)
    if df.empty:
        raise SystemExit(f"no runs under {args.root}")
    df.sort_values(["target", "task", "cell", "few_shot", "seed"]).to_csv(out / "runs.csv", index=False)
    (out / "table_mean_std.md").write_text(mean_std_table(df, "acc") + "\n" + mean_std_table(df, "gap_recovered"))
    rng = np.random.default_rng(args.seed)
    (out / "paired_tests.md").write_text(paired(df, "acc", rng) + "\n" + paired(df, "gap_recovered", rng))
    counts = df.groupby(["target", "cell"]).size().unstack(fill_value=0)
    print(f"{len(df)} runs\n{counts}\nwritten to {out}")


if __name__ == "__main__":
    main()
