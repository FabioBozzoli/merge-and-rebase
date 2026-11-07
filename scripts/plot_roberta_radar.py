#!/usr/bin/env python
"""Radar plots of the t5-base (NTK) -> roberta-base experiments: one plot per few-shot, one axis per dataset.

Four series per plot, each the test accuracy averaged over the seeds:

- Zero-shot            the fixed nearest-mean target head (the ``target_zeroshot`` column of the steer logs)
- Linear probing       ``eval/text_linear_probe.py`` summaries (lr chosen on val)
- steering attn+global    steer_text, block_source=attention, target_block_pooling=global
- steering attn+segments  steer_text, block_source=attention, target_block_pooling=segments

Steering numbers are read from the run logs (``{task}_roberta_{variant}_fs{K}_seed{S}.log``, the final
"Benchmark" table), probe numbers from ``probe_roberta_fs{K}_seed{S}.json``.

The figures are vector PDFs meant for LaTeX: TrueType fonts embedded (fonttype 42), sans-serif text in
Bitstream Vera Sans when installed, else DejaVu Sans (its direct descendant, same glyphs), and the mathtext
font set to match. Steering uses the paper's pink (block stage 2) / light blue (global stage 2).

    python scripts/plot_roberta_radar.py --steer-logs DIR --probe-dir DIR [--out-dir DIR] [--few-shots 200 300 500]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASKS = ("mnli", "qnli", "rte", "scitail", "snli", "sick")
BLOCK_STAGE_TWO = "#F48FB1"  # \definecolor{blockstagetwo}{HTML}{F48FB1}   light pink
GLOBAL_STAGE_TWO = "#90CAF9"  # \definecolor{globalstagetwo}{HTML}{90CAF9}  light blue
STYLE = {  # series -> (color, linestyle)
    "Zero-shot": ("#9E9E9E", ":"),
    "Linear probing": ("#424242", "--"),
    "Steering (attention, global)": (GLOBAL_STAGE_TWO, "-"),
    "Steering (attention, segments)": (BLOCK_STAGE_TWO, "-"),
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Bitstream Vera Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",  # the Vera-derived math font set
    "pdf.fonttype": 42,
})


def steer_result(logs: Path, task: str, variant: str, few_shot: int, seed: int) -> tuple[float, float]:
    """(target_zeroshot, rebased) test accuracy of one steer_text run, from the last row of its final table."""
    text = (logs / f"{task}_roberta_{variant}_fs{few_shot}_seed{seed}.log").read_text().split("Benchmark nli6")[-1]
    rows = re.findall(rf"^\s*{task}\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", text, re.M)
    if not rows:
        raise ValueError(f"no final result row for {task} in the {variant} fs{few_shot} seed{seed} log")
    return float(rows[-1][0]), float(rows[-1][1])


def collect(logs: Path, probe_dir: Path, few_shot: int, seeds: list[int]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {name: [] for name in STYLE}
    probe = {s: json.loads((probe_dir / f"probe_roberta_fs{few_shot}_seed{s}.json").read_text()) for s in seeds}
    for task in TASKS:
        series["Zero-shot"].append(mean(steer_result(logs, task, "attn_segments", few_shot, s)[0] for s in seeds))
        series["Linear probing"].append(mean(probe[s]["test_results"]["per_task_rebased"][task] for s in seeds))
        for name, variant in (("Steering (attention, global)", "attn_global"), ("Steering (attention, segments)", "attn_segments")):
            series[name].append(mean(steer_result(logs, task, variant, few_shot, s)[1] for s in seeds))
    return series


def radar(series: dict[str, list[float]], few_shot: int, out: Path, rmin: float, rmax: float) -> None:
    angles = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    fig, ax = plt.subplots(figsize=(6.4, 6.6), subplot_kw={"polar": True})
    for name, values in series.items():
        color, ls = STYLE[name]
        closed = values + values[:1]
        ax.plot(closed_angles, closed, color=color, linestyle=ls, linewidth=2.2, marker="o", markersize=4.5, label=name)
        ax.fill(closed_angles, closed, color=color, alpha=0.10)
    ax.set_ylim(rmin, rmax)
    ax.set_xticks(angles)
    ax.set_xticklabels([t.upper() for t in TASKS], fontsize=13)
    ax.tick_params(axis="x", pad=14)  # dataset names sit clear of the outer ring
    # No number sits inside the disc: with four series spread over 0.38-0.82 any spot on the plane is crossed by
    # some line. The rings are drawn every 0.1 and the scale is stated below the legend instead.
    ax.set_yticks(np.arange(np.ceil(rmin * 10) / 10, rmax + 1e-9, 0.1))
    ax.set_yticklabels([])
    ax.grid(color="#BDBDBD", linewidth=0.8)
    ax.spines["polar"].set_color("#BDBDBD")
    ax.set_title(f"{few_shot} shots per class", fontsize=14, pad=30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10.5, frameon=False, columnspacing=1.6)
    ax.text(0.5, -0.27, f"Test accuracy. Rings every 0.1: {rmin:.1f} at the centre, {rmax:.1f} at the outer edge.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steer-logs", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/roberta_radar"))
    p.add_argument("--few-shots", nargs="+", type=int, default=[200, 300, 500])
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 54, 89])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = {k: collect(args.steer_logs, args.probe_dir, k, args.seeds) for k in args.few_shots}
    # One radial range for all plots so the three are comparable at a glance.
    values = [v for s in data.values() for vs in s.values() for v in vs]
    rmin, rmax = max(0.0, np.floor(min(values) * 10) / 10 - 0.1), min(1.0, np.ceil(max(values) * 10) / 10)
    for k, series in data.items():
        out = args.out_dir / f"radar_roberta_fs{k}.pdf"
        radar(series, k, out, rmin, rmax)
        print(f"wrote {out}")
        for name, vs in series.items():
            print(f"  {name:>24}: " + "  ".join(f"{t}={v:.4f}" for t, v in zip(TASKS, vs)))


if __name__ == "__main__":
    main()
