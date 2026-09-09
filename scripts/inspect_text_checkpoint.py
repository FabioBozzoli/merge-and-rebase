#!/usr/bin/env python
"""
Print what a ``finetune/train_text.py`` checkpoint recorded about its own
training run: the accuracy it measured, the label space it trained against,
the strategy used, and how far its weights actually moved from the base model.

Needs no GPU and no dataset -- it only reads the ``.pt``. Use it to answer, in
seconds, whether a fine-tuned checkpoint ever learned its task, before spending
a job on a rebase experiment whose every number is read through that checkpoint.

Usage:
    python -m scripts.inspect_text_checkpoint <ckpt.pt> [<ckpt.pt> ...]
    python -m scripts.inspect_text_checkpoint <run_summary.json>   # every task at once
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_META_KEYS = (
    "task",
    "strategy",
    "forward_mode",
    "format",
    "num_labels",
    "labels",
    "head_class_ids",
    "best_epoch",
    "last_epoch",
)


def _state_dict_of(payload: dict) -> dict | None:
    import torch

    for key in ("state_dict", "head", "model", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    return None


def inspect(path: Path) -> None:
    # Imported here, not at module scope: the run-summary mode below is pure
    # JSON and stays usable on a machine with no torch installed.
    import torch

    print(f"\n=== {path} ===")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print(f"  not a payload dict (got {type(payload).__name__}) -- raw tensors only, no metadata to read")
        return

    metrics = payload.get("metrics") or {}
    val, test = metrics.get("val_top1"), metrics.get("test_top1")
    print(f"  recorded metrics: val_top1={val} test_top1={test}")
    for key in _META_KEYS:
        if key in payload:
            print(f"  {key}: {payload[key]}")
    if "backbone" in payload:
        print(f"  backbone: {payload['backbone']}")

    sd = _state_dict_of(payload)
    if sd is None:
        print("  (no state dict in this payload -- peft adapter or metadata-only checkpoint)")
        return
    head_keys = [k for k in sd if "classification_head" in k or k.startswith(("score.", "classifier."))]
    finite = all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
    print(f"  tensors: {len(sd)} ({len(head_keys)} head), all finite: {finite}")
    if head_keys:
        print(f"  head keys present: {sorted(head_keys)}")

    if val is None and test is None:
        print(
            "  NOTE: no recorded metrics in this checkpoint. It cannot confirm the run converged;\n"
            "        evaluate the checkpoint directly (text_rebase.py --eval-source-finetuned) instead."
        )
    elif test is not None and float(test) < 0.4:
        print(
            "  WARNING: training itself measured near-chance accuracy for this checkpoint. The weights\n"
            "           differ from the base model but never learned the task -- any task vector derived\n"
            "           from this checkpoint carries no task signal."
        )


def _walk_task_entries(node: Any, found: dict[str, dict]) -> None:
    """Collect every ``{... "metrics": {"test_top1": ...} ...}`` entry in a run summary."""
    if isinstance(node, dict):
        metrics = node.get("metrics")
        if isinstance(metrics, dict) and ("test_top1" in metrics or "val_top1" in metrics):
            task = str(node.get("task") or node.get("meta", {}).get("task") or f"entry{len(found)}")
            found.setdefault(task, node)
        for value in node.values():
            _walk_task_entries(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk_task_entries(value, found)


def summarize(path: Path) -> None:
    """Print every task's recorded accuracy from a ``finetune.train_text`` run summary."""
    entries: dict[str, dict] = {}
    _walk_task_entries(json.loads(path.read_text(encoding="utf-8")), entries)
    if not entries:
        print(f"{path}: no task entries with recorded metrics found.")
        return

    print(f"\n=== {path} ===")
    header = f"  {'task':<10} {'val_top1':>9} {'test_top1':>10} {'classes':>8} {'chance':>7} {'vs chance':>10}  verdict"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for task, node in entries.items():
        metrics = node.get("metrics", {})
        val, test = metrics.get("val_top1"), metrics.get("test_top1")
        labels = node.get("labels") or node.get("meta", {}).get("labels") or []
        n_classes = len(labels) if labels else 0
        chance = 1.0 / n_classes if n_classes else float("nan")
        margin = (float(test) - chance) if (test is not None and n_classes) else float("nan")
        if margin != margin:
            verdict = "no class count"
        elif margin < 0.05:
            verdict = "AT CHANCE - did not learn"
        elif margin < 0.15:
            verdict = "barely above chance"
        else:
            verdict = "learned something"
        print(
            f"  {task:<10} {_fmt(val):>9} {_fmt(test):>10} {n_classes:>8} "
            f"{chance:>7.3f} {margin:>+10.3f}  {verdict}"
        )
    print(
        "\n  Caution: 'chance' here is uniform (1/classes). On an imbalanced task the majority-class\n"
        "  baseline is higher, so a model that collapsed to always predicting the majority label can\n"
        "  clear uniform chance while still having learned nothing. Compare against the majority\n"
        "  frequency of the task before calling a run successful."
    )


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "-"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint .pt files, or a run-summary .json to tabulate every task.")
    args = p.parse_args()
    for path in args.checkpoints:
        if path.suffix.lower() == ".json":
            summarize(path)
        else:
            inspect(path)


if __name__ == "__main__":
    main()
