from __future__ import annotations

import pytest

from merge_and_rebase.eval.vision_checkpoints import _format_results_table, _resolve_checkpoint_map


def test_resolve_checkpoint_map_accepts_named_cli_entries() -> None:
    resolved = _resolve_checkpoint_map(
        config_value=None,
        cli_values=["Cars=/tmp/cars.pt", "DTD=/tmp/dtd.pt"],
        checkpoint_overrides=None,
        tasks=["Cars", "DTD"],
    )

    assert resolved == {"Cars": "/tmp/cars.pt", "DTD": "/tmp/dtd.pt"}


def test_resolve_checkpoint_map_accepts_ordered_cli_entries() -> None:
    resolved = _resolve_checkpoint_map(
        config_value=None,
        cli_values=["/tmp/cars.pt", "/tmp/dtd.pt"],
        checkpoint_overrides=None,
        tasks=["Cars", "DTD"],
    )

    assert resolved == {"Cars": "/tmp/cars.pt", "DTD": "/tmp/dtd.pt"}


def test_checkpoint_override_replaces_config_value() -> None:
    resolved = _resolve_checkpoint_map(
        config_value={"Cars": "old.pt", "DTD": "dtd.pt"},
        cli_values=None,
        checkpoint_overrides=["Cars=new.pt"],
        tasks=["Cars", "DTD"],
    )

    assert resolved == {"Cars": "new.pt", "DTD": "dtd.pt"}


def test_full_config_checkpoint_map_allows_a_task_subset() -> None:
    resolved = _resolve_checkpoint_map(
        config_value={"Cars": "cars.pt", "DTD": "dtd.pt"},
        cli_values=None,
        checkpoint_overrides=None,
        tasks=["Cars"],
    )

    assert resolved == {"Cars": "cars.pt"}


def test_resolve_checkpoint_map_reports_missing_tasks() -> None:
    with pytest.raises(ValueError, match="Missing fine-tuned checkpoints.*DTD"):
        _resolve_checkpoint_map(
            config_value={"Cars": "cars.pt"},
            cli_values=None,
            checkpoint_overrides=None,
            tasks=["Cars", "DTD"],
        )


def test_results_table_contains_each_accuracy_and_average() -> None:
    table = _format_results_table(
        [
            {"task": "Cars", "checkpoint": "/tmp/cars.pt", "top1": 0.8, "seconds": 1.2},
            {"task": "DTD", "checkpoint": "/tmp/dtd.pt", "top1": 0.6, "seconds": 2.3},
        ],
        split="test",
    )

    assert "Cars" in table
    assert "DTD" in table
    assert "80.00" in table
    assert "60.00" in table
    assert "avg" in table
    assert "70.00" in table
