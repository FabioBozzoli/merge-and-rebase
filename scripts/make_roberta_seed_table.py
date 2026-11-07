#!/usr/bin/env python
"""Per-dataset, per-seed table of the t5-base (NTK) -> roberta-base results, as markdown.

One section per dataset with a row per (few-shot, method) and one column per seed plus the mean over
seeds, then a section with the mean over the datasets. Methods: RoBERTa zero-shot (fixed nearest-mean head, so identical across seeds), linear probing
(lr chosen on val), steering global_mlp (last layer only) and steering attention+segments. Test accuracy,
4 decimals. Steering numbers come from the run logs, probe numbers from the probe summaries.

    python scripts/make_roberta_seed_table.py --steer-logs DIR --probe-dir DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

TASKS = ("mnli", "qnli", "rte", "scitail", "snli", "sick")
SICK_MAX_FEW_SHOT = 500
METHODS = ("Zero-shot", "Linear probing", "Steering global MLP", "Steering attention+segments")


def steer(logs: Path, task: str, variant: str, few_shot: int, seed: int) -> tuple[float, float]:
    text = (logs / f"{task}_roberta_{variant}_fs{few_shot}_seed{seed}.log").read_text().split("Benchmark nli6")[-1]
    rows = re.findall(rf"^\s*{task}\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", text, re.M)
    if not rows:
        raise ValueError(f"no final result row for {task} {variant} fs{few_shot} seed{seed}")
    return float(rows[-1][0]), float(rows[-1][1])


def values(logs: Path, probe_dir: Path, task: str, few_shot: int, seeds: list[int]) -> dict[str, list[float]]:
    probe = [
        json.loads((probe_dir / f"probe_roberta_fs{few_shot}_seed{s}.json").read_text())["test_results"]["per_task_rebased"][task]
        for s in seeds
    ]
    return {
        "Zero-shot": [steer(logs, task, "global_mlp", few_shot, s)[0] for s in seeds],
        "Linear probing": probe,
        "Steering global MLP": [steer(logs, task, "global_mlp", few_shot, s)[1] for s in seeds],
        "Steering attention+segments": [steer(logs, task, "attn_segments", few_shot, s)[1] for s in seeds],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steer-logs", type=Path, required=True)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--few-shots", nargs="+", type=int, default=[200, 300, 500, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 54, 89])
    args = p.parse_args()

    out = ["# t5-base (NTK) → roberta-base: test accuracy per dataset and seed", "",
           "Test accuracy, 4 decimals, one section per dataset: a row per (few-shot, method), a column per seed and "
           "the mean over seeds. Zero-shot is the fixed nearest-mean RoBERTa head (same for every seed); the linear "
           "probe picks its lr on val. `sick` has no 1000-shot row (only 606 class-2 rows in the train pool). The last "
           "section is the mean over the datasets, per seed.", ""]
    header = ["| Shots | Method | " + " | ".join(f"Seed {s}" for s in args.seeds) + " | Mean |",
              "|---|---|" + "---|" * (len(args.seeds) + 1)]
    table = {(t, k): values(args.steer_logs, args.probe_dir, t, k, args.seeds)
             for k in args.few_shots for t in TASKS if not (t == "sick" and k > SICK_MAX_FEW_SHOT)}
    for t in TASKS:
        out += [f"## {t.upper()}", ""] + header
        for k in args.few_shots:
            if (t, k) not in table:
                continue
            for m in METHODS:
                v = table[(t, k)][m]
                out.append(f"| {k} | {m} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {mean(v):.4f} |")
        out.append("")
    out += ["## AVG over datasets (5 datasets at 1000 shots: no sick)", ""] + header
    for k in args.few_shots:
        tasks = [t for t in TASKS if (t, k) in table]
        for m in METHODS:
            per_seed = [mean(table[(t, k)][m][i] for t in tasks) for i in range(len(args.seeds))]
            out.append(f"| {k} | {m} | " + " | ".join(f"{x:.4f}" for x in per_seed) + f" | {mean(per_seed):.4f} |")
    out.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
