#!/usr/bin/env python3
"""Diagnose over/underfitting of Steer's global MLP from nonlinear caches.

The script deliberately fits Stage 1 on the few-shot support set, diagnoses
Stage 2 on the unused part of the original training split, and only evaluates
the test cache at the predeclared epochs 100 and 300.  It never uses test
metrics to assign a verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from merge_and_rebase.rebase.methods.steer import (
    _few_shot,
    _fit_global_mlp_with_trace,
    _load_cached_head,
    _ridge,
    _stage1_projection,
)

DEFAULT_CONFIGS = (
    Path("configs/vision8_steer_nonlinear.json"),
    Path("configs/vision8_steer_vitb_vitl_nonlinear.json"),
)
DEFAULT_SEEDS = (33, 54, 89)
DEFAULT_SHOTS = (1, 2, 5, 10, 20)
DEFAULT_HIDDEN_DIMS = (128, 512, 1024, 2048)
DEFAULT_CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300)
TEST_EPOCHS = frozenset({100, 300})
T_CRITICAL_95_DF2 = 4.3026527297


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated integer list")
    return values


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-15 else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    return mean(kept) if kept else float("nan")


@dataclass(frozen=True)
class CacheSpec:
    config_path: Path
    task: str
    transfer: str
    architecture: str
    cache_dir: Path
    stage1_lambda: float
    ridge_lambda: float


@dataclass(frozen=True)
class SweepPoint:
    shots: int
    hidden_dim: int


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _cache_specs(config_paths: Sequence[Path], selected_tasks: set[str] | None) -> list[CacheSpec]:
    specs: list[CacheSpec] = []
    for config_path in config_paths:
        cfg = _load_json(config_path)
        params = cfg.get("method_params", {})
        if cfg.get("method") != "steer" or params.get("stage_2_strategy") != "global_mlp":
            raise ValueError(f"{config_path} is not a steer/global_mlp config")
        if params.get("feature_regime") != "standard":
            raise ValueError(f"{config_path} must use feature_regime='standard' for nonlinear caches")
        tuned = cfg.get("tuned_ckpts")
        if not isinstance(tuned, dict) or not tuned:
            raise ValueError(f"{config_path} must provide task-keyed tuned_ckpts")

        source_model = str(cfg["source_clip_model"])
        source_pretrained = str(cfg["source_clip_pretrained"])
        target_model = str(cfg["target_clip_model"])
        target_pretrained = str(cfg["target_clip_pretrained"])
        source_tag = f"{source_model}_{source_pretrained}"
        target_tag = f"{target_model}_{target_pretrained}"
        cache_root = Path(str(params["feature_cache_dir"])).expanduser()
        transfer = f"{source_model}/{source_pretrained} -> {target_model}/{target_pretrained}"
        architecture = f"{source_model} -> {target_model}"
        for task in tuned:
            if selected_tasks is not None and task.lower() not in selected_tasks:
                continue
            specs.append(
                CacheSpec(
                    config_path=config_path,
                    task=str(task),
                    transfer=transfer,
                    architecture=architecture,
                    cache_dir=cache_root / f"{source_tag}_to_{target_tag}" / str(task) / "standard",
                    stage1_lambda=float(params.get("stage1_lambda", 1.0)),
                    ridge_lambda=float(params.get("ridge_lambda", 1.0)),
                )
            )
    if not specs:
        raise ValueError("No tasks selected")
    return specs


def _sweep_points(shots: Sequence[int], hidden_dims: Sequence[int], capacity_shot: int) -> list[SweepPoint]:
    points = {SweepPoint(int(value), 1024) for value in shots}
    points.update(SweepPoint(int(capacity_shot), int(value)) for value in hidden_dims)
    return sorted(points, key=lambda point: (point.shots, point.hidden_dim))


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing nonlinear Steer cache tensor: {path}. Run vision_rebase once with the corresponding "
            "global_mlp config, or update method_params.feature_cache_dir to the materialized cache."
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    return value


def _load_cache(spec: CacheSpec) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for split in ("train", "test"):
        for name in ("features_A", "delta_A", "features_B", "y_A"):
            out[f"{split}_{name}"] = _load_tensor(spec.cache_dir / split / f"{name}.pt")
    for name in ("A", "B"):
        path = spec.cache_dir / "train" / f"head_{name}.pt"
        head = _load_cached_head(path)
        if head is None:
            raise FileNotFoundError(f"Missing or invalid cached Steer head: {path}")
        out[f"head_{name}"] = head
    return out


def _parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim


def _classification_accuracy(
    features: torch.Tensor,
    corrections: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
) -> float:
    corrected = F.normalize(features + corrections, dim=-1)
    return float((corrected @ head.T).argmax(dim=1).eq(labels).double().mean())


def _constant_accuracies(
    features: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    stage0_correct = 0
    oracle_correct = 0
    for start in range(0, len(features), batch_size):
        stop = min(len(features), start + batch_size)
        zeros = torch.zeros_like(targets[start:stop])
        stage0_correct += round(
            _classification_accuracy(features[start:stop], zeros, labels[start:stop], head) * (stop - start)
        )
        oracle_correct += round(
            _classification_accuracy(features[start:stop], targets[start:stop], labels[start:stop], head)
            * (stop - start)
        )
    return stage0_correct / len(features), oracle_correct / len(features)


def _evaluate_predictor(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    targets: torch.Tensor,
    base_features: torch.Tensor,
    labels: torch.Tensor,
    head: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    squared_error = 0.0
    squared_target = 0.0
    cosine_sum = 0.0
    correct = 0
    n = len(features)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        x = features[start:stop].to(device=device, dtype=torch.float64)
        target = targets[start:stop].to(device=device, dtype=torch.float64)
        base = base_features[start:stop].to(device=device, dtype=torch.float64)
        y = labels[start:stop].to(device=device)
        prediction = predictor(x)
        squared_error += float((prediction - target).square().sum())
        squared_target += float(target.square().sum())
        cosine_sum += float(F.cosine_similarity(prediction, target, dim=-1, eps=1e-12).sum())
        logits = F.normalize(base + prediction, dim=-1) @ head.T
        correct += int(logits.argmax(dim=1).eq(y).sum())
    elements = targets.numel()
    mse = squared_error / elements
    target_mse = squared_target / elements
    return {
        "mse": mse,
        "nrmse": _safe_div(math.sqrt(mse), math.sqrt(target_mse)),
        "cosine": cosine_sum / n,
        "accuracy": correct / n,
        "target_rms": math.sqrt(target_mse),
        "n_examples": float(n),
    }


def _split_row(
    *,
    spec: CacheSpec,
    point: SweepPoint,
    seed: int,
    model: str,
    epoch: int,
    split: str,
    metrics: Mapping[str, float],
    stage0_accuracy: float,
    oracle_accuracy: float,
    param_count: int,
    support_examples: int,
) -> dict[str, Any]:
    return {
        "config_path": str(spec.config_path),
        "transfer": spec.transfer,
        "architecture": spec.architecture,
        "task": spec.task,
        "seed": seed,
        "shots": point.shots,
        "hidden_dim": point.hidden_dim,
        "model": model,
        "epoch": epoch,
        "split": split,
        "mse": float(metrics["mse"]),
        "nrmse": float(metrics["nrmse"]),
        "cosine": float(metrics["cosine"]),
        "accuracy": float(metrics["accuracy"]),
        "target_rms": float(metrics["target_rms"]),
        "n_examples": int(metrics["n_examples"]),
        "stage0_accuracy": stage0_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "param_count": param_count,
        "parameters_per_support_example": param_count / max(1, support_examples),
    }


def _run_one(
    spec: CacheSpec,
    point: SweepPoint,
    seed: int,
    *,
    data: Mapping[str, torch.Tensor],
    checkpoints: Sequence[int],
    max_epochs: int,
    device: torch.device,
    metric_batch_size: int,
) -> list[dict[str, Any]]:
    f_a = data["train_features_A"].double()
    delta_a = data["train_delta_A"].double()
    f_b = data["train_features_B"].double()
    labels = data["train_y_A"].long()
    test_delta_a = data["test_delta_A"].double()
    test_f_b = data["test_features_B"].double()
    test_labels = data["test_y_A"].long()
    w_a = data["head_A"].double()
    w_b = data["head_B"].double()

    selected = _few_shot(labels, point.shots, seed)
    holdout_mask = torch.ones(len(labels), dtype=torch.bool)
    holdout_mask[selected] = False
    holdout = torch.where(holdout_mask)[0]
    if not len(holdout):
        raise ValueError(f"No Stage-2 holdout remains for {spec.task}, shots={point.shots}, seed={seed}")

    logit_map = _stage1_projection(
        f_a=f_a,
        delta_a=delta_a,
        w_a=w_a,
        f_b=f_b,
        w_b=w_b,
        selected=selected,
        regularization=spec.stage1_lambda,
    )
    p_b = torch.linalg.pinv(w_b)
    train_targets = delta_a @ logit_map.T @ p_b.T
    test_targets = test_delta_a @ logit_map.T @ p_b.T
    split_data = {
        "support": (f_b[selected], train_targets[selected], f_b[selected], labels[selected]),
        "holdout": (f_b[holdout], train_targets[holdout], f_b[holdout], labels[holdout]),
        "test": (test_f_b, test_targets, test_f_b, test_labels),
    }
    head_device = w_b.to(device=device, dtype=torch.float64)
    constants = {
        split: _constant_accuracies(base, target, y, w_b, batch_size=metric_batch_size)
        for split, (_, target, base, y) in split_data.items()
    }
    param_count = _parameter_count(f_b.shape[1], point.hidden_dim, train_targets.shape[1])

    rows: list[dict[str, Any]] = []
    support_examples = int(selected.numel())
    ridge_coef = _ridge(f_b[selected], train_targets[selected], regularization=spec.ridge_lambda).to(device)
    for split, (x, target, base, y) in split_data.items():
        metrics = _evaluate_predictor(
            lambda batch, coefficient=ridge_coef: batch @ coefficient,
            x,
            target,
            base,
            y,
            head_device,
            device=device,
            batch_size=metric_batch_size,
        )
        rows.append(
            _split_row(
                spec=spec,
                point=point,
                seed=seed,
                model="global_ridge",
                epoch=-1,
                split=split,
                metrics=metrics,
                stage0_accuracy=constants[split][0],
                oracle_accuracy=constants[split][1],
                param_count=f_b.shape[1] * train_targets.shape[1],
                support_examples=support_examples,
            )
        )

    train_x = f_b[selected].to(device=device, dtype=torch.float64)
    train_y = train_targets[selected].to(device=device, dtype=torch.float64)

    def metric_fn(model: nn.Module, epoch: int) -> Mapping[str, float]:
        flattened: dict[str, float] = {}
        splits = ("support", "holdout", "test") if epoch in TEST_EPOCHS else ("support", "holdout")
        for split in splits:
            x, target, base, y = split_data[split]
            metrics = _evaluate_predictor(
                model,
                x,
                target,
                base,
                y,
                head_device,
                device=device,
                batch_size=metric_batch_size,
            )
            for key, value in metrics.items():
                flattened[f"{split}/{key}"] = value
        return flattened

    mlp_seed = 10_000 + seed * 100 + int(selected.numel())
    _, trace = _fit_global_mlp_with_trace(
        train_x,
        train_y,
        seed=mlp_seed,
        epochs=max_epochs,
        hidden_dim=point.hidden_dim,
        checkpoint_epochs=checkpoints,
        metric_fn=metric_fn,
    )
    for snapshot in trace:
        epoch = int(snapshot["epoch"])
        for split in ("support", "holdout", "test"):
            prefix = f"{split}/"
            if f"{prefix}mse" not in snapshot:
                continue
            metrics = {
                key: snapshot[f"{prefix}{key}"]
                for key in ("mse", "nrmse", "cosine", "accuracy", "target_rms", "n_examples")
            }
            rows.append(
                _split_row(
                    spec=spec,
                    point=point,
                    seed=seed,
                    model="global_mlp",
                    epoch=epoch,
                    split=split,
                    metrics=metrics,
                    stage0_accuracy=constants[split][0],
                    oracle_accuracy=constants[split][1],
                    param_count=param_count,
                    support_examples=support_examples,
                )
            )
    return rows


def _run_key(spec: CacheSpec, point: SweepPoint, seed: int, checkpoints: Sequence[int]) -> str:
    payload = {
        "config": str(spec.config_path.resolve()),
        "task": spec.task,
        "shots": point.shots,
        "hidden_dim": point.hidden_dim,
        "seed": seed,
        "checkpoints": list(checkpoints),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_partial(path: Path) -> dict[str, list[dict[str, Any]]]:
    completed: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            completed[str(record["key"])] = list(record["rows"])
        except Exception as exc:
            raise ValueError(f"Invalid resume record at {path}:{line_number}") from exc
    return completed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


ConfigKey = tuple[str, str, int, int]


def _config_key(row: Mapping[str, Any]) -> ConfigKey:
    return str(row["architecture"]), str(row["task"]), int(row["shots"]), int(row["hidden_dim"])


def _find_row(
    rows: Sequence[Mapping[str, Any]], *, seed: int, split: str, epoch: int, model: str = "global_mlp"
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if int(row["seed"]) == seed
            and row["split"] == split
            and int(row["epoch"]) == epoch
            and row["model"] == model
        ),
        None,
    )


def _relative_change(old: float, new: float) -> float:
    return _safe_div(new - old, abs(old))


def _diagnose(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], max_epochs: int) -> dict[str, Any]:
    seed_details: list[dict[str, Any]] = []
    for seed in seeds:
        trajectory = sorted(
            (
                row
                for row in rows
                if int(row["seed"]) == seed and row["model"] == "global_mlp" and row["split"] == "holdout"
            ),
            key=lambda row: int(row["epoch"]),
        )
        support = {
            int(row["epoch"]): row
            for row in rows
            if int(row["seed"]) == seed and row["model"] == "global_mlp" and row["split"] == "support"
        }
        ridge = _find_row(rows, seed=seed, split="holdout", epoch=-1, model="global_ridge")
        if not trajectory or ridge is None:
            continue
        final = trajectory[-1]
        minimum = min(trajectory, key=lambda row: float(row["mse"]))
        minimum_epoch = int(minimum["epoch"])
        support_at_min = support[minimum_epoch]
        support_final = support[int(final["epoch"])]
        holdout_rise = _relative_change(float(minimum["mse"]), float(final["mse"]))
        support_drop = -_relative_change(float(support_at_min["mse"]), float(support_final["mse"]))
        overfit = holdout_rise >= 0.10 and support_drop >= 0.05

        row_200 = _find_row(rows, seed=seed, split="holdout", epoch=min(200, max_epochs))
        support_200 = _find_row(rows, seed=seed, split="support", epoch=min(200, max_epochs))
        continuing = False
        if row_200 is not None and support_200 is not None and int(final["epoch"]) > int(row_200["epoch"]):
            continuing = (
                -_relative_change(float(row_200["mse"]), float(final["mse"])) >= 0.05
                and -_relative_change(float(support_200["mse"]), float(support_final["mse"])) >= 0.05
            )
        worse_than_ridge = float(final["mse"]) >= 1.05 * float(ridge["mse"])
        underfit = not overfit and (continuing or worse_than_ridge)
        target_rms = float(final["target_rms"])
        if not math.isfinite(target_rms) or target_rms < 1e-10:
            verdict = "degenerate"
        elif overfit:
            verdict = "overfitting"
        elif underfit:
            verdict = "underfitting/non-convergence"
        elif holdout_rise > 0.10:
            verdict = "mixed evidence"
        else:
            verdict = "stable fit"
        seed_details.append(
            {
                "seed": seed,
                "verdict": verdict,
                "minimum_holdout_epoch": minimum_epoch,
                "minimum_holdout_mse": float(minimum["mse"]),
                "final_holdout_mse": float(final["mse"]),
                "holdout_rise_from_min": holdout_rise,
                "support_drop_after_holdout_min": support_drop,
                "final_holdout_accuracy": float(final["accuracy"]),
                "peak_holdout_accuracy": max(float(row["accuracy"]) for row in trajectory),
                "ridge_holdout_mse": float(ridge["mse"]),
                "ridge_holdout_accuracy": float(ridge["accuracy"]),
                "continuing_improvement": continuing,
                "worse_than_ridge": worse_than_ridge,
            }
        )
    counts: dict[str, int] = defaultdict(int)
    for detail in seed_details:
        counts[str(detail["verdict"])] += 1
    if counts["degenerate"] >= 1:
        verdict = "degenerate"
    elif counts["overfitting"] >= 2:
        verdict = "overfitting"
    elif counts["underfitting/non-convergence"] >= 2:
        verdict = "underfitting/non-convergence"
    elif counts["stable fit"] >= 2:
        verdict = "stable fit"
    else:
        verdict = "mixed evidence"
    return {"verdict": verdict, "seed_counts": dict(counts), "seeds": seed_details}


def _pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def _configuration_comment(
    key: ConfigKey,
    diagnosis: Mapping[str, Any],
    all_diagnoses: Mapping[ConfigKey, Mapping[str, Any]],
) -> str:
    architecture, task, shots, hidden = key
    details = list(diagnosis["seeds"])
    min_epoch = median([float(item["minimum_holdout_epoch"]) for item in details])
    holdout_rise = _finite_mean(float(item["holdout_rise_from_min"]) for item in details)
    support_drop = _finite_mean(float(item["support_drop_after_holdout_min"]) for item in details)
    final_acc = _finite_mean(float(item["final_holdout_accuracy"]) for item in details)
    peak_acc = _finite_mean(float(item["peak_holdout_accuracy"]) for item in details)
    ridge_mse = _finite_mean(float(item["ridge_holdout_mse"]) for item in details)
    final_mse = _finite_mean(float(item["final_holdout_mse"]) for item in details)
    counts = diagnosis["seed_counts"]
    agreement = ", ".join(f"{name}: {count}/3" for name, count in sorted(counts.items()))
    sentences = [
        f"**Verdetto: {diagnosis['verdict']}.** Il minimo della loss holdout cade mediamente intorno "
        f"all'epoca {min_epoch:.0f}; da quel punto alla fine la loss holdout varia di {_pct(holdout_rise)}, "
        f"mentre la loss sul support continua a ridursi di {_pct(support_drop)}. Accordo fra seed: {agreement}.",
        f"L'accuracy holdout finale media è {final_acc:.4f}, contro un massimo lungo la traiettoria di "
        f"{peak_acc:.4f}. La MSE finale dell'MLP è {final_mse:.6g}; il riferimento global ridge ottiene "
        f"{ridge_mse:.6g}.",
    ]
    if diagnosis["verdict"] == "overfitting":
        sentences.append(
            "Il giudizio deriva dalla divergenza fra ottimizzazione e generalizzazione: l'MLP continua a "
            "memorizzare il support dopo che la loss holdout ha già raggiunto il proprio minimo. Il confronto "
            "con più shot e dimensioni inferiori indica se la causa più plausibile è il rapporto capacità/dati."
        )
    elif diagnosis["verdict"] == "underfitting/non-convergence":
        continuing_count = sum(bool(item["continuing_improvement"]) for item in details)
        ridge_count = sum(bool(item["worse_than_ridge"]) for item in details)
        sentences.append(
            f"Il segnale di underfitting è sostenuto da miglioramento ancora in corso in {continuing_count}/3 seed "
            f"e da una loss peggiore di ridge in {ridge_count}/3 seed. Il primo caso suggerisce ottimizzazione "
            "non conclusa; il secondo segnala che la non linearità non sta compensando la maggiore complessità."
        )
    elif diagnosis["verdict"] == "stable fit":
        sentences.append(
            "Le curve non mostrano una separazione sistematica support-holdout né un miglioramento tardivo "
            "sufficiente a indicare mancata convergenza; la stabilità va comunque letta insieme al livello "
            "assoluto di accuracy e al confronto con ridge."
        )
    elif diagnosis["verdict"] == "mixed evidence":
        sentences.append(
            "I seed non supportano una conclusione unica: il report conserva quindi le traiettorie individuali "
            "invece di nascondere l'instabilità dietro la sola curva media."
        )

    same_axis: list[tuple[str, ConfigKey]] = []
    if hidden == 1024:
        shot_values = sorted(
            candidate[2] for candidate in all_diagnoses if candidate[:2] == key[:2] and candidate[3] == 1024
        )
        position = shot_values.index(shots)
        for index in (position - 1, position + 1):
            if 0 <= index < len(shot_values):
                same_axis.append((f"{shot_values[index]}-shot", (architecture, task, shot_values[index], 1024)))
    if shots == 2:
        dims = sorted(candidate[3] for candidate in all_diagnoses if candidate[:3] == key[:3])
        position = dims.index(hidden)
        for index in (position - 1, position + 1):
            if 0 <= index < len(dims):
                same_axis.append((f"hidden={dims[index]}", (architecture, task, 2, dims[index])))
    if same_axis:
        comparisons = []
        for label, candidate in same_axis:
            candidate_diagnosis = all_diagnoses[candidate]
            candidate_mse = _finite_mean(item["final_holdout_mse"] for item in candidate_diagnosis["seeds"])
            delta = _relative_change(final_mse, candidate_mse)
            comparisons.append(f"{label}: {candidate_diagnosis['verdict']}, MSE finale {_pct(delta)} vs questa scheda")
        sentences.append("Configurazioni adiacenti: " + "; ".join(comparisons) + ".")

    counterpart = next(
        (candidate for candidate in all_diagnoses if candidate[1:] == key[1:] and candidate[0] != architecture),
        None,
    )
    if counterpart is not None:
        counterpart_diagnosis = all_diagnoses[counterpart]
        counterpart_mse = _finite_mean(item["final_holdout_mse"] for item in counterpart_diagnosis["seeds"])
        architecture_delta = _relative_change(final_mse, counterpart_mse)
        sentences.append(
            f"A parità di dataset, shot e dimensione, il trasferimento `{counterpart[0]}` è classificato "
            f"come **{counterpart_diagnosis['verdict']}** e ha una MSE finale diversa di {_pct(architecture_delta)}; "
            "questo confronto aiuta a separare l'effetto della capacità MLP da quello del mismatch architetturale."
        )
    return "\n\n".join(sentences)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = (
            row["architecture"],
            row["task"],
            row["shots"],
            row["hidden_dim"],
            row["model"],
            row["epoch"],
            row["split"],
        )
        groups[group].append(row)
    output: list[dict[str, Any]] = []

    def summarize(group: tuple[Any, ...], members: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
        record = dict(
            zip(
                ("architecture", "task", "shots", "hidden_dim", "model", "epoch", "split"),
                group,
                strict=True,
            )
        )
        record["scope"] = scope
        for metric in ("mse", "nrmse", "cosine", "accuracy"):
            values = [float(member[metric]) for member in members]
            avg = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            half_width = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values)) if len(values) == 3 else float("nan")
            record[f"{metric}_mean"] = avg
            record[f"{metric}_std"] = sd
            record[f"{metric}_ci95_low"] = avg - half_width
            record[f"{metric}_ci95_high"] = avg + half_width
        return record

    for group, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append(summarize(group, members, scope="per_task_across_seeds"))

    # Do not pool task/seed observations as if independent. First average the
    # eight tasks within each seed, then compute uncertainty across the three
    # seed-level means.
    seed_task_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_task_groups[
            (
                row["architecture"],
                row["shots"],
                row["hidden_dim"],
                row["model"],
                row["epoch"],
                row["split"],
                row["seed"],
            )
        ].append(row)
    seed_means: list[dict[str, Any]] = []
    for group, members in seed_task_groups.items():
        architecture, shots, hidden_dim, model, epoch, split, seed = group
        seed_means.append(
            {
                "architecture": architecture,
                "task": "__all_tasks__",
                "shots": shots,
                "hidden_dim": hidden_dim,
                "model": model,
                "epoch": epoch,
                "split": split,
                "seed": seed,
                **{
                    metric: _finite_mean(float(member[metric]) for member in members)
                    for metric in ("mse", "nrmse", "cosine", "accuracy")
                },
            }
        )
    across_task_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_means:
        group = (
            row["architecture"],
            row["task"],
            row["shots"],
            row["hidden_dim"],
            row["model"],
            row["epoch"],
            row["split"],
        )
        across_task_groups[group].append(row)
    for group, members in sorted(across_task_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append(summarize(group, members, scope="task_mean_then_across_seeds"))
    return output


def _plot_card(
    key: ConfigKey,
    rows: Sequence[Mapping[str, Any]],
    comment: str,
    figure_path: Path,
    pdf: Any,
) -> None:
    import matplotlib.pyplot as plt

    architecture, task, shots, hidden = key
    mlp = [row for row in rows if row["model"] == "global_mlp" and row["split"] != "test"]
    seeds = sorted({int(row["seed"]) for row in mlp})
    epochs = sorted({int(row["epoch"]) for row in mlp})
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    colors = {"support": "#1f77b4", "holdout": "#d62728"}

    for split in ("support", "holdout"):
        matrix: list[list[float]] = []
        for seed in seeds:
            values = [
                float(
                    next(
                        row["mse"]
                        for row in mlp
                        if row["split"] == split and int(row["seed"]) == seed and int(row["epoch"]) == epoch
                    )
                )
                for epoch in epochs
            ]
            matrix.append(values)
            axes[0, 0].plot(epochs, values, color=colors[split], alpha=0.22, linewidth=0.9)
        tensor = torch.tensor(matrix, dtype=torch.float64)
        avg = tensor.mean(0).numpy()
        sd = tensor.std(0, unbiased=True).numpy()
        axes[0, 0].plot(epochs, avg, color=colors[split], linewidth=2.0, label=split)
        axes[0, 0].fill_between(epochs, avg - sd, avg + sd, color=colors[split], alpha=0.14)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Loss MSE: curve per seed e media +/- SD")
    axes[0, 0].set_xlabel("Epoca")
    test_rows = [row for row in rows if row["model"] == "global_mlp" and row["split"] == "test"]
    for epoch in sorted({int(row["epoch"]) for row in test_rows}):
        axes[0, 0].scatter(
            epoch,
            _finite_mean(float(row["mse"]) for row in test_rows if int(row["epoch"]) == epoch),
            marker="*",
            s=90,
            color="black",
            label="test prefissato" if epoch == min(int(row["epoch"]) for row in test_rows) else None,
        )
    axes[0, 0].legend()

    for seed in seeds:
        support = {
            int(row["epoch"]): float(row["mse"])
            for row in mlp
            if row["split"] == "support" and int(row["seed"]) == seed
        }
        holdout = {
            int(row["epoch"]): float(row["mse"])
            for row in mlp
            if row["split"] == "holdout" and int(row["seed"]) == seed
        }
        gap = [_safe_div(holdout[epoch] - support[epoch], support[epoch]) for epoch in epochs]
        axes[0, 1].plot(epochs, gap, marker=".", label=f"seed {seed}")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Gap relativo holdout-support")
    axes[0, 1].set_xlabel("Epoca")
    axes[0, 1].legend(fontsize=8)

    for split in ("support", "holdout"):
        averages = [
            _finite_mean(float(row["accuracy"]) for row in mlp if row["split"] == split and int(row["epoch"]) == epoch)
            for epoch in epochs
        ]
        axes[1, 0].plot(epochs, averages, color=colors[split], label=f"MLP {split}")
    ridge_holdout = _finite_mean(
        float(row["accuracy"]) for row in rows if row["model"] == "global_ridge" and row["split"] == "holdout"
    )
    stage0 = _finite_mean(float(row["stage0_accuracy"]) for row in mlp if row["split"] == "holdout")
    oracle = _finite_mean(float(row["oracle_accuracy"]) for row in mlp if row["split"] == "holdout")
    axes[1, 0].axhline(ridge_holdout, color="#9467bd", linestyle="--", label="ridge holdout")
    axes[1, 0].axhline(stage0, color="gray", linestyle=":", label="Stage 0")
    axes[1, 0].axhline(oracle, color="green", linestyle=":", label="Stage 1 oracle")
    for epoch in sorted({int(row["epoch"]) for row in test_rows}):
        axes[1, 0].scatter(
            epoch,
            _finite_mean(float(row["accuracy"]) for row in test_rows if int(row["epoch"]) == epoch),
            marker="*",
            s=90,
            color="black",
            label="test prefissato" if epoch == min(TEST_EPOCHS) else None,
        )
    axes[1, 0].set_title("Accuracy e riferimenti")
    axes[1, 0].set_xlabel("Epoca")
    axes[1, 0].legend(fontsize=7, ncol=2)

    for metric, linestyle in (("cosine", "-"), ("nrmse", "--")):
        for split in ("support", "holdout"):
            averages = [
                _finite_mean(float(row[metric]) for row in mlp if row["split"] == split and int(row["epoch"]) == epoch)
                for epoch in epochs
            ]
            axes[1, 1].plot(epochs, averages, color=colors[split], linestyle=linestyle, label=f"{metric} {split}")
    axes[1, 1].set_title("Qualità della correzione")
    axes[1, 1].set_xlabel("Epoca")
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(f"{task} | {architecture} | {shots}-shot | hidden={hidden}", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    compact_comment = " ".join(comment.replace("**", "").replace("`", "").split())
    fig.text(0.03, 0.018, compact_comment[:850], fontsize=7.5, va="bottom", wrap=True)
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # A dedicated text page keeps the PDF as informative as the Markdown
    # report instead of squeezing or truncating the configuration commentary.
    detail_fig = plt.figure(figsize=(8.27, 11.69))
    detail_fig.suptitle(f"Commento — {task} | {architecture} | {shots}-shot | hidden={hidden}", y=0.97, fontsize=13)
    wrapped = "\n\n".join(
        textwrap.fill(paragraph.replace("**", "").replace("`", ""), width=105) for paragraph in comment.split("\n\n")
    )
    detail_fig.text(0.06, 0.91, wrapped, fontsize=9, va="top", family="sans-serif")
    detail_fig.text(
        0.06,
        0.08,
        "Nota metodologica: il verdetto usa solo support e holdout. I punti test alle epoche 100 e 300 "
        "sono riportati esclusivamente come controllo esterno.",
        fontsize=8,
        va="bottom",
        wrap=True,
    )
    pdf.savefig(detail_fig, bbox_inches="tight")
    plt.close(detail_fig)


def _render_reports(
    rows: Sequence[Mapping[str, Any]],
    diagnoses: Mapping[ConfigKey, Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    del plt
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[ConfigKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_config_key(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    markdown = [
        "# Analisi overfitting/underfitting di Steer `global_mlp`",
        "",
        "Il verdetto usa esclusivamente support e holdout del training. Il test compare soltanto alle epoche "
        "100 e 300 e non partecipa ad alcuna decisione.",
        "",
        "## Regole di lettura",
        "",
        "- Overfitting: loss holdout +10% dal minimo e loss support ancora in calo di almeno 5%, in almeno 2/3 seed.",
        "- Underfitting/non-convergenza: miglioramento ancora marcato a fine training oppure MLP almeno 5% peggiore di ridge.",
        "- Fit stabile: nessuno dei due segnali precedenti e andamento concorde in almeno 2/3 seed.",
        "- Evidenza mista: i seed non concordano.",
        "",
    ]
    with PdfPages(output_dir / "report.pdf") as pdf:
        for index, key in enumerate(sorted(grouped), start=1):
            diagnosis = diagnoses[key]
            comment = _configuration_comment(key, diagnosis, diagnoses)
            architecture, task, shots, hidden = key
            slug = f"{index:03d}_{architecture}_{task}_{shots}shot_h{hidden}".lower()
            slug = "".join(character if character.isalnum() else "_" for character in slug)
            figure_name = f"{slug}.png"
            _plot_card(key, grouped[key], comment, figures_dir / figure_name, pdf)
            markdown.extend(
                [
                    f"## {task} — {architecture} — {shots}-shot — hidden={hidden}",
                    "",
                    f"![Curve diagnostiche](figures/{figure_name})",
                    "",
                    comment,
                    "",
                    "| Seed | Diagnosi | Epoca minimo holdout | Crescita holdout | Calo support successivo | Accuracy finale |",
                    "|---:|---|---:|---:|---:|---:|",
                    *[
                        f"| {item['seed']} | {item['verdict']} | {item['minimum_holdout_epoch']} | "
                        f"{_pct(item['holdout_rise_from_min'])} | {_pct(item['support_drop_after_holdout_min'])} | "
                        f"{item['final_holdout_accuracy']:.4f} |"
                        for item in diagnosis["seeds"]
                    ],
                    "",
                ]
            )
            details = diagnosis["seeds"]
            summary_rows.append(
                {
                    "architecture": architecture,
                    "task": task,
                    "shots": shots,
                    "hidden_dim": hidden,
                    "verdict": diagnosis["verdict"],
                    "seed_verdicts": json.dumps({item["seed"]: item["verdict"] for item in details}, sort_keys=True),
                    "median_minimum_holdout_epoch": median(item["minimum_holdout_epoch"] for item in details),
                    "mean_holdout_rise_from_min": _finite_mean(item["holdout_rise_from_min"] for item in details),
                    "mean_support_drop_after_min": _finite_mean(
                        item["support_drop_after_holdout_min"] for item in details
                    ),
                    "mean_final_holdout_accuracy": _finite_mean(item["final_holdout_accuracy"] for item in details),
                    "comment": " ".join(comment.replace("**", "").split()),
                }
            )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task subset.")
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--shots", type=_csv_ints, default=DEFAULT_SHOTS)
    parser.add_argument("--hidden-dims", type=_csv_ints, default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--capacity-sweep-shot", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--checkpoint-epochs", type=_csv_ints, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("results/steer_global_mlp_overfitting"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if tuple(args.seeds) != DEFAULT_SEEDS:
        print(f"[warning] Requested seeds {tuple(args.seeds)} differ from the Stage-2 seeds {DEFAULT_SEEDS}.")
    checkpoints = tuple(sorted(set(epoch for epoch in args.checkpoint_epochs if epoch <= args.max_epochs)))
    required = {0, args.max_epochs} | ({100, 300} & set(range(args.max_epochs + 1)))
    checkpoints = tuple(sorted(set(checkpoints) | required))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable; pass --device cpu explicitly")
    selected_tasks = (
        None if args.tasks is None else {item.strip().lower() for item in args.tasks.split(",") if item.strip()}
    )
    specs = _cache_specs(args.configs, selected_tasks)
    points = _sweep_points(args.shots, args.hidden_dims, args.capacity_sweep_shot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "metrics.partial.jsonl"
    completed = _read_partial(partial_path) if args.resume else {}

    total = len(specs) * len(points) * len(args.seeds)
    done = 0
    for spec in specs:
        spec_data: Mapping[str, torch.Tensor] | None = None
        for point in points:
            for seed in args.seeds:
                key = _run_key(spec, point, seed, checkpoints)
                done += 1
                if key in completed:
                    print(
                        f"[{done}/{total}] resume {spec.task} | {spec.architecture} | shots={point.shots} hidden={point.hidden_dim} seed={seed}"
                    )
                    continue
                if spec_data is None:
                    spec_data = _load_cache(spec)
                print(
                    f"[{done}/{total}] run {spec.task} | {spec.architecture} | shots={point.shots} hidden={point.hidden_dim} seed={seed}"
                )
                run_rows = _run_one(
                    spec,
                    point,
                    seed,
                    data=spec_data,
                    checkpoints=checkpoints,
                    max_epochs=args.max_epochs,
                    device=device,
                    metric_batch_size=args.metric_batch_size,
                )
                completed[key] = run_rows
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "rows": run_rows}, sort_keys=True) + "\n")

    rows = [row for run_rows in completed.values() for row in run_rows]
    _write_csv(args.output_dir / "metrics.csv", rows)
    aggregate = _aggregate_rows(rows)
    _write_csv(args.output_dir / "aggregate.csv", aggregate)
    grouped: dict[ConfigKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_config_key(row)].append(row)
    diagnoses = {key: _diagnose(group_rows, args.seeds, args.max_epochs) for key, group_rows in grouped.items()}
    summaries = _render_reports(rows, diagnoses, args.output_dir)
    _write_csv(args.output_dir / "configurations.csv", summaries)
    analysis = {
        "rules": {
            "overfitting": "holdout MSE rises >=10% from its minimum while support MSE falls >=5%, in >=2/3 seeds",
            "underfitting": "no overfitting and late support+holdout improvement >=5%, or final holdout MSE >=5% worse than ridge",
            "test_policy": "test is evaluated only at epochs 100 and 300 and never affects the verdict",
        },
        "configurations": [
            {
                "architecture": key[0],
                "task": key[1],
                "shots": key[2],
                "hidden_dim": key[3],
                **diagnosis,
            }
            for key, diagnosis in sorted(diagnoses.items())
        ],
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "configs": [str(path) for path in args.configs],
        "seeds": list(args.seeds),
        "shots": list(args.shots),
        "hidden_dims": list(args.hidden_dims),
        "capacity_sweep_shot": args.capacity_sweep_shot,
        "max_epochs": args.max_epochs,
        "checkpoint_epochs": list(checkpoints),
        "test_epochs": sorted(TEST_EPOCHS & set(checkpoints)),
        "device": str(device),
        "cache_directories": sorted({str(spec.cache_dir) for spec in specs}),
        "completed_runs": len(completed),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Analysis complete: {args.output_dir / 'report.md'} and {args.output_dir / 'report.pdf'}")


if __name__ == "__main__":
    main()
