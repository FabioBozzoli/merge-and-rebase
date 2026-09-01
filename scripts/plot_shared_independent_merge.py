"""Plot paired shared-vs-independent multi-task merge results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("results/brace_lmc_corrected")
MERGE = ROOT / "merge"
OUT = ROOT / "plots"


PAIRS = {
    "eurosat_gtsrb": "EuroSAT + GTSRB",
    "dtd_svhn": "DTD + SVHN",
}


def _load(pair: str, mode: str) -> dict:
    path = MERGE / f"{pair}_{mode}_task_arithmetic.json"
    return json.loads(path.read_text())


def _plot(pair: str, title: str) -> None:
    shared = _load(pair, "shared")["test_results"]
    independent = _load(pair, "independent")["test_results"]
    tasks = list(shared["per_task_rebased"])
    baseline = [shared["per_task_baseline"][task] for task in tasks]
    shared_acc = [shared["per_task_rebased"][task] for task in tasks]
    independent_acc = [independent["per_task_rebased"][task] for task in tasks]

    OUT.mkdir(parents=True, exist_ok=True)
    x = list(range(len(tasks)))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = [
        ax.bar([value - 0.25 for value in x], baseline, width=0.25, label="target zero-shot"),
        ax.bar(x, shared_acc, width=0.25, label="shared merge"),
        ax.bar([value + 0.25 for value in x], independent_acc, width=0.25, label="independent merge"),
    ]
    for group in bars:
        ax.bar_label(group, labels=[f"{bar.get_height():.1%}" for bar in group], padding=2, fontsize=8)
    ax.set(xticks=x, xticklabels=tasks, ylim=(0, 1), ylabel="Held-out target test accuracy")
    ax.legend()
    fig.suptitle(f"Matched {title} task-arithmetic merge")
    suffix = "" if pair == "eurosat_gtsrb" else f"_{pair}"
    fig.savefig(OUT / f"shared_independent{suffix}_merge_accuracy.png", dpi=180)


def main() -> None:
    for pair, title in PAIRS.items():
        _plot(pair, title)


if __name__ == "__main__":
    main()
