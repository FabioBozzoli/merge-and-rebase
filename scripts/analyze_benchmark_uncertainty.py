#!/usr/bin/env python3
"""Compute paired task-level uncertainty summaries from benchmark JSON runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _task_scores(payload: dict[str, Any]) -> dict[str, float]:
    results = payload.get("test_results", {})
    if not isinstance(results, dict):
        raise ValueError("Missing test_results object.")
    for key in ("per_task_absolute_accuracy", "per_task_rebased", "per_task_acc"):
        values = results.get(key)
        if isinstance(values, dict):
            return {str(task): float(score) for task, score in values.items()}
    raise ValueError("No per-task absolute accuracy map found in test_results.")


def _two_sided_sign_test_pvalue(differences: list[float]) -> float:
    wins = sum(value > 0.0 for value in differences)
    losses = sum(value < 0.0 for value in differences)
    n = wins + losses
    if n == 0:
        return 1.0
    lower_tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def analyze_pairs(candidate_runs: list[Path], reference_runs: list[Path]) -> dict[str, Any]:
    if len(candidate_runs) != len(reference_runs) or not candidate_runs:
        raise ValueError("Candidate and reference run lists must be non-empty and have equal length.")

    seed_means: list[float] = []
    all_differences: list[float] = []
    paired_tasks: list[list[str]] = []
    for candidate_path, reference_path in zip(candidate_runs, reference_runs, strict=True):
        candidate = _task_scores(json.loads(candidate_path.read_text(encoding="utf-8")))
        reference = _task_scores(json.loads(reference_path.read_text(encoding="utf-8")))
        tasks = sorted(set(candidate) & set(reference))
        if not tasks:
            raise ValueError(f"No shared tasks for {candidate_path} and {reference_path}.")
        differences = [candidate[task] - reference[task] for task in tasks]
        seed_means.append(mean(differences))
        all_differences.extend(differences)
        paired_tasks.append(tasks)

    ci_half_width = 0.0
    if len(seed_means) > 1:
        ci_half_width = 1.96 * stdev(seed_means) / math.sqrt(len(seed_means))
    return {
        "num_seeds": len(seed_means),
        "paired_tasks_per_seed": paired_tasks,
        "mean_seed_level_difference": mean(seed_means),
        "normal_95_ci": [mean(seed_means) - ci_half_width, mean(seed_means) + ci_half_width],
        "pooled_task_level_sign_test_pvalue": _two_sided_sign_test_pvalue(all_differences),
        "pooled_task_wins": sum(value > 0.0 for value in all_differences),
        "pooled_task_losses": sum(value < 0.0 for value in all_differences),
        "notes": [
            "The sign test is descriptive when multiple seeds share tasks; it is not a replacement for seed-level uncertainty.",
            "Apply Holm correction across the pre-specified family of method comparisons before claiming significance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path, help="Candidate summary JSONs, one per seed.")
    parser.add_argument("--reference", nargs="+", required=True, type=Path, help="Reference summary JSONs, one per seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    report = analyze_pairs(args.candidate, args.reference)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
