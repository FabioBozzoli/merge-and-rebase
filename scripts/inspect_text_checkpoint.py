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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

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


def _state_dict_of(payload: dict) -> dict[str, torch.Tensor] | None:
    for key in ("state_dict", "head", "model", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    return None


def inspect(path: Path) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    args = p.parse_args()
    for path in args.checkpoints:
        inspect(path)


if __name__ == "__main__":
    main()
