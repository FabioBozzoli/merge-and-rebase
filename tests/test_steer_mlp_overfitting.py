from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from merge_and_rebase.rebase.methods import steer as steer_mod


def _analysis_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_steer_global_mlp_overfitting.py"
    spec = importlib.util.spec_from_file_location("steer_mlp_overfitting", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_traced_global_mlp_preserves_original_fit() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    targets = torch.randn(12, 3, generator=generator, dtype=torch.float64)

    original = steer_mod._fit_global_mlp(features, targets, seed=33, epochs=8, hidden_dim=6)
    traced, trace = steer_mod._fit_global_mlp_with_trace(
        features,
        targets,
        seed=33,
        epochs=8,
        hidden_dim=6,
        checkpoint_epochs=(0, 4, 8),
        metric_fn=lambda model, _epoch: {"mse": float((model(features) - targets).square().mean())},
    )

    assert [int(row["epoch"]) for row in trace] == [0, 4, 8]
    for expected, actual in zip(original.parameters(), traced.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_traced_global_mlp_rejects_invalid_checkpoints() -> None:
    features = torch.zeros(2, 2, dtype=torch.float64)
    targets = torch.zeros(2, 1, dtype=torch.float64)
    try:
        steer_mod._fit_global_mlp_with_trace(
            features,
            targets,
            seed=0,
            epochs=2,
            checkpoint_epochs=(3,),
            metric_fn=lambda _model, _epoch: {},
        )
    except ValueError as error:
        assert "checkpoint epochs" in str(error)
    else:
        raise AssertionError("out-of-range checkpoints must fail")


def _diagnostic_rows(kind: str):
    rows = []
    for seed in (33, 54, 89):
        for epoch, support, holdout in (
            (0, 1.0, 1.0),
            (100, 0.6, 0.7),
            (200, 0.4, 0.6),
            (300, 0.3, 0.8 if kind == "overfit" else 0.5),
        ):
            for split, mse in (("support", support), ("holdout", holdout)):
                rows.append(
                    {
                        "seed": seed,
                        "model": "global_mlp",
                        "split": split,
                        "epoch": epoch,
                        "mse": mse,
                        "accuracy": 1.0 - mse / 2.0,
                        "target_rms": 1.0,
                    }
                )
        rows.append(
            {
                "seed": seed,
                "model": "global_ridge",
                "split": "holdout",
                "epoch": -1,
                "mse": 0.55,
                "accuracy": 0.7,
                "target_rms": 1.0,
            }
        )
    return rows


def test_diagnosis_detects_overfitting() -> None:
    module = _analysis_module()
    diagnosis = module._diagnose(_diagnostic_rows("overfit"), (33, 54, 89), 300)
    assert diagnosis["verdict"] == "overfitting"
    assert diagnosis["seed_counts"]["overfitting"] == 3


def test_diagnosis_detects_underfitting_stable_mixed_and_degenerate() -> None:
    module = _analysis_module()
    underfit_rows = _diagnostic_rows("underfit")
    assert module._diagnose(underfit_rows, (33, 54, 89), 300)["verdict"] == "underfitting/non-convergence"

    stable_rows = _diagnostic_rows("underfit")
    for row in stable_rows:
        if row["model"] == "global_mlp" and row["epoch"] == 200:
            row["mse"] = 0.31 if row["split"] == "support" else 0.51
        if row["model"] == "global_mlp" and row["epoch"] == 300:
            row["mse"] = 0.30 if row["split"] == "support" else 0.50
    assert module._diagnose(stable_rows, (33, 54, 89), 300)["verdict"] == "stable fit"

    overfit_rows = _diagnostic_rows("overfit")
    mixed_rows = []
    for seed, source in ((33, overfit_rows), (54, underfit_rows), (89, stable_rows)):
        mixed_rows.extend(row.copy() for row in source if row["seed"] == seed)
    assert module._diagnose(mixed_rows, (33, 54, 89), 300)["verdict"] == "mixed evidence"

    degenerate_rows = [row.copy() for row in stable_rows]
    for row in degenerate_rows:
        if row["model"] == "global_mlp" and row["seed"] == 33:
            row["target_rms"] = 0.0
    assert module._diagnose(degenerate_rows, (33, 54, 89), 300)["verdict"] == "degenerate"


def test_sweep_deduplicates_shared_anchor() -> None:
    module = _analysis_module()
    points = module._sweep_points((1, 2, 5), (128, 1024, 2048), 2)
    assert len(points) == 5
    assert sum(point.shots == 2 and point.hidden_dim == 1024 for point in points) == 1


def test_report_renderer_writes_markdown_pdf_and_card(tmp_path: Path) -> None:
    module = _analysis_module()
    rows = _diagnostic_rows("overfit")
    for row in rows:
        row.update(
            {
                "architecture": "ViT-B-16 -> ViT-L-14",
                "task": "Synthetic",
                "shots": 2,
                "hidden_dim": 128,
                "nrmse": float(row["mse"]) ** 0.5,
                "cosine": 1.0 - float(row["mse"]) / 2.0,
                "stage0_accuracy": 0.4,
                "oracle_accuracy": 0.9,
            }
        )
    key = ("ViT-B-16 -> ViT-L-14", "Synthetic", 2, 128)
    diagnosis = module._diagnose(rows, (33, 54, 89), 300)
    summaries = module._render_reports(rows, {key: diagnosis}, tmp_path)

    assert summaries[0]["verdict"] == "overfitting"
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.pdf").stat().st_size > 0
    assert len(list((tmp_path / "figures").glob("*.png"))) == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Verdetto: overfitting" in report
    assert "| 33 | overfitting |" in report
