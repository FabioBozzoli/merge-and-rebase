from __future__ import annotations

import pytest

from merge_and_rebase.eval.vision_rebase import _norm_acc
from merge_and_rebase.eval.rebase_metrics import audit_rebase_summary


def test_normalized_rebase_ratio_can_exceed_one() -> None:
    # Transport can improve over an untransported task vector.
    assert _norm_acc(0.639, 0.582) == pytest.approx(1.0979381443)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_normalized_rebase_ratio_rejects_invalid_raw_accuracy(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _norm_acc(value, 0.582)


def test_rebase_summary_audit_keeps_raw_accuracy_and_ratio_distinct() -> None:
    payload = {
        "test_results": {
            "per_task_baseline_accuracy": {"PCAM": 0.582},
            "per_task_absolute_accuracy": {"PCAM": 0.639},
            "per_task_normalized_accuracy_ratio": {"PCAM": 0.639 / 0.582},
        }
    }

    assert audit_rebase_summary(payload) == ["PCAM: absolute=0.639000, baseline=0.582000, ratio=1.097938"]


def test_rebase_summary_audit_rejects_ratio_as_raw_accuracy() -> None:
    payload = {
        "test_results": {
            "per_task_baseline_accuracy": {"PCAM": 0.582},
            "per_task_absolute_accuracy": {"PCAM": 1.098},
            "per_task_normalized_accuracy_ratio": {"PCAM": 1.098},
        }
    }

    with pytest.raises(ValueError, match="raw accuracy"):
        audit_rebase_summary(payload)
