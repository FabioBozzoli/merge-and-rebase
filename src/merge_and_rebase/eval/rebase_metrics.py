from __future__ import annotations

import math
from typing import Any


def validate_accuracy(value: float, *, label: str) -> float:
    """Validate a raw top-1 accuracy before it is normalized or serialized."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite value in [0, 1], got {value!r}.")
    return value


def normalized_accuracy_ratio(result_acc: float, baseline_acc: float) -> float:
    result_acc = validate_accuracy(result_acc, label="result accuracy")
    baseline_acc = validate_accuracy(baseline_acc, label="baseline accuracy")
    if baseline_acc <= 0.0:
        return float("nan")
    return result_acc / baseline_acc


def audit_rebase_summary(payload: dict[str, Any]) -> list[str]:
    """Check that a rebase summary keeps raw accuracies and ratios distinct."""
    results = payload.get("test_results")
    if not isinstance(results, dict):
        raise ValueError("Missing object: test_results.")

    baseline = results.get("per_task_baseline_accuracy", results.get("per_task_baseline"))
    absolute = results.get("per_task_absolute_accuracy", results.get("per_task_rebased"))
    normalized = results.get("per_task_normalized_accuracy_ratio", results.get("per_task_norm"))
    if not all(isinstance(values, dict) for values in (baseline, absolute, normalized)):
        raise ValueError("Rebase summary must include per-task baseline, absolute, and normalized metrics.")
    if set(baseline) != set(absolute) or set(baseline) != set(normalized):
        raise ValueError("Per-task baseline, absolute, and normalized metric keys must match.")

    messages: list[str] = []
    for task in sorted(baseline):
        baseline_acc = validate_accuracy(baseline[task], label=f"{task} baseline accuracy")
        absolute_acc = validate_accuracy(absolute[task], label=f"{task} raw accuracy")
        ratio = float(normalized[task])
        expected = float("nan") if baseline_acc == 0.0 else absolute_acc / baseline_acc
        if not math.isfinite(ratio) or not math.isclose(ratio, expected, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"{task} normalized ratio is inconsistent with raw values: "
                f"expected {expected!r}, got {ratio!r}."
            )
        messages.append(
            f"{task}: absolute={absolute_acc:.6f}, baseline={baseline_acc:.6f}, ratio={ratio:.6f}"
        )
    return messages
