#!/usr/bin/env python
"""Post-hoc regression diagnostics for one steer_text ``block_ridge`` run.

steer_text only reports three downstream accuracies (stage0 = B alone, stage1 =
the oracle correction built from A's *true* test delta, stage2 = the fitted
predictor). This script opens up the gap between stage1 and stage2 by treating
Stage 2 as the regression it is -- per-block ridge from B's grouped block
activations to the Stage-1 target -- and asking how well it fits:

1. on the support it was fit on (train, ``selected`` rows),
2. on unseen rows from the same distribution (the unselected train rows, and the
   test split),
3. and on which test examples it fails: a per-example stage0/stage1/stage2
   correctness cross-tab, plus accuracy and fit quality binned by how large a
   correction each example needs and by how wrong B was to begin with.

It also refits ``global_ridge`` on the same support with the production
``_ridge`` (same ``selected``, same Stage-1 map), so every number has a
side-by-side for the strategy block_ridge underperforms.

Nothing is refit or reimplemented for block_ridge itself: the Stage-1 map,
``pinv(w_B)``, the support indices and the ridge coefficients all come from the
run's ``<task>_steer_artifacts.pt`` (see scripts/slurm/submit_block_ridge_diag.py),
the features from the run's own feature cache, and grouping/prediction from
``steer.py``. Before reporting anything the script rebuilds stage0/1/2 test
accuracy and aborts unless they match what the run recorded.

Usage:
    python scripts/diagnose_block_ridge.py --exp-dir <run-dir> --output <report.json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from merge_and_rebase.rebase.methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _cache_split_dir,
    _load_cached_split,
    _predict_block_ridge,
    _ridge,
)
from merge_and_rebase.rebase.text.steer_text import _accuracy

GRID_ROOT = Path("/work/intesasanpaolo_phd/merge-and-rebase/t5enc_steer_text")
NUM_QUANTILES = 4


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _model_tag(model: str, kind: str) -> str:
    """Mirror of ``text_rebase._model_tag``; the cache path is keyed on it."""
    tag = str(model).replace("/", "__")
    kind = str(kind).strip().lower()
    return f"{tag}__{kind}" if kind and kind != "sequence_classification" else tag


def _load_split(cfg: Mapping[str, Any], task: str, split: str) -> dict[str, Any]:
    mp = cfg["method_params"]
    kind = cfg.get("model_kind", "sequence_classification")
    cache_dir = _cache_split_dir(
        str(mp["feature_cache_dir"]),
        _model_tag(cfg["source_model_name_or_path"], kind),
        _model_tag(cfg["target_model_name_or_path"], kind),
        task,
        str(mp["feature_regime"]),
        split,
    )
    data = _load_cached_split(cache_dir, need_blocks=True)
    if data is None:
        # _load_cached_split swallows load errors and returns None; never let that
        # read as "no data" further down.
        raise FileNotFoundError(f"feature cache missing or unreadable: {cache_dir}")
    return data


def _recorded_diagnostics(exp_dir: Path, task: str) -> dict[str, float]:
    summary = json.loads((exp_dir / "summary.json").read_text())
    diag = summary.get("steer_diagnostics", {}).get(task)
    if not diag:
        raise ValueError(f"{exp_dir}/summary.json has no steer_diagnostics for {task}")
    return diag


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def r2_scores(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """R^2 with sklearn's multioutput conventions, on [n, d] tensors.

    ``variance_weighted`` = 1 - sum_d SS_res / sum_d SS_tot, i.e. the fraction of
    the total (all-dims) target variance explained. ``uniform_average`` averages
    per-dim R^2, so a few well-fit high-variance dims cannot hide many bad ones.
    Per sklearn, a zero-variance dim scores 1 if predicted exactly, else 0.
    """
    ss_res = (y_true - y_pred).square().sum(dim=0)
    ss_tot = (y_true - y_true.mean(dim=0, keepdim=True)).square().sum(dim=0)
    nonconst = ss_tot > 0
    per_dim = torch.where(nonconst, 1.0 - ss_res / ss_tot.clamp_min(1e-300), (ss_res == 0).double())
    total = float(ss_tot.sum())
    return {
        "variance_weighted": float(1.0 - ss_res.sum() / total) if total > 0 else float("nan"),
        "uniform_average": float(per_dim.mean()),
    }


def vector_agreement(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Per-example direction and scale agreement, summarized.

    ``norm_ratio`` = |pred| / |true| is the shrinkage lens: ridge's L2 penalty
    pulls predictions toward zero, which shows up as a ratio well below 1.
    """
    eps = 1e-12
    true_norm = y_true.norm(dim=1)
    pred_norm = y_pred.norm(dim=1)
    cosine = (y_true * y_pred).sum(dim=1) / (true_norm * pred_norm).clamp_min(eps)
    ratio = pred_norm / true_norm.clamp_min(eps)
    rel_err = (y_pred - y_true).norm(dim=1) / true_norm.clamp_min(eps)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "norm_ratio_median": float(ratio.median()),
        "relative_error_median": float(rel_err.median()),
    }


def fit_report(
    y_true: torch.Tensor, y_pred: torch.Tensor, head: torch.Tensor, *, per_example: bool = True
) -> dict[str, Any]:
    """Feature-space and logit-space fit of a predicted correction.

    The Stage-1 target is ``delta @ logit_map.T @ pinv(w_B).T``, so it lives in
    the (<= num-classes)-dim row space of w_B; the head only ever sees the
    correction through ``head`` (w_B restricted to this task's classes). The
    logit-space numbers are therefore the ones that decide accuracy, and the
    feature-space ones say how the regression itself is doing.
    """
    report: dict[str, Any] = {
        "n": int(y_true.shape[0]),
        "feature_r2": r2_scores(y_true, y_pred),
        "logit_r2": r2_scores(y_true @ head.T, y_pred @ head.T),
    }
    if per_example:
        report["feature_agreement"] = vector_agreement(y_true, y_pred)
        report["logit_agreement"] = vector_agreement(y_true @ head.T, y_pred @ head.T)
    return report


def predictions(features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mask_class: Sequence[int]) -> torch.Tensor:
    """Per-example head-space predictions, exactly as ``steer_text._accuracy`` makes them."""
    logits = features @ weight.T
    if bias is not None:
        logits = logits + bias
    index = torch.tensor([int(c) for c in mask_class], dtype=torch.long)
    return index[logits.index_select(dim=1, index=index).argmax(dim=1)]


def margins(features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, labels: torch.Tensor, mask_class: Sequence[int]) -> torch.Tensor:
    """True-class logit minus the best other allowed logit; < 0 means B is wrong."""
    logits = features @ weight.T
    if bias is not None:
        logits = logits + bias
    index = torch.tensor([int(c) for c in mask_class], dtype=torch.long)
    allowed = logits.index_select(dim=1, index=index)
    col = {int(c): i for i, c in enumerate(mask_class)}
    true_col = torch.tensor([col[int(y)] for y in labels], dtype=torch.long)
    true_logit = allowed.gather(1, true_col[:, None]).squeeze(1)
    others = allowed.scatter(1, true_col[:, None], float("-inf"))
    return true_logit - others.max(dim=1).values


# --------------------------------------------------------------------------
# Block_ridge reconstruction (mirrors steer_text.prepare's block_ridge branch)
# --------------------------------------------------------------------------


def grouped_blocks(features_b_blocks: Mapping[int, torch.Tensor], state: Mapping[str, Any], rows: torch.Tensor | None) -> dict[int, torch.Tensor]:
    """B's blocks grouped to A's block count, as prepare()/correction_fn do.

    Grouping is row-wise, so selecting ``rows`` first gives the same tensors as
    prepare()'s group-then-select, at a fraction of the memory.
    """
    n_target = int(state["num_target_residual"])
    n_source = int(state["num_source_residual_blocks"])
    pick = (lambda t: t[rows]) if rows is not None else (lambda t: t)
    residual = {b: pick(features_b_blocks[b]).double() for b in range(n_target)}
    if n_target != n_source:
        residual = _BLOCK_GROUP_STRATEGIES[str(state["block_group_strategy"])](residual, n_source)
    blocks = dict(residual)
    blocks[n_source] = pick(features_b_blocks[n_target]).double()
    return blocks


def per_block_contributions(coefficients: Sequence[torch.Tensor], blocks: Mapping[int, torch.Tensor]) -> list[torch.Tensor]:
    return [blocks[b] @ c for b, c in enumerate(coefficients)]


# --------------------------------------------------------------------------
# Hard vs easy
# --------------------------------------------------------------------------


def correctness_crosstab(correct: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    names = list(correct)
    n = int(next(iter(correct.values())).numel())
    rows = []
    for combo in itertools.product((False, True), repeat=len(names)):
        mask = torch.ones(n, dtype=torch.bool)
        for name, want in zip(names, combo):
            mask &= correct[name] == want
        count = int(mask.sum())
        rows.append({**{name: ("right" if want else "wrong") for name, want in zip(names, combo)},
                     "count": count, "fraction": count / n})
    return rows


def quantile_report(
    key: torch.Tensor,
    *,
    correct: Mapping[str, torch.Tensor],
    y_true: torch.Tensor,
    y_pred: Mapping[str, torch.Tensor],
    head: torch.Tensor,
    num_quantiles: int = NUM_QUANTILES,
) -> list[dict[str, Any]]:
    """Accuracy and fit quality per quantile of ``key`` (low to high)."""
    order = torch.argsort(key)
    bins = torch.tensor_split(order, num_quantiles)
    out = []
    for q, idx in enumerate(bins):
        row: dict[str, Any] = {
            "quantile": q + 1,
            "n": int(idx.numel()),
            "key_min": float(key[idx].min()),
            "key_max": float(key[idx].max()),
            "accuracy": {name: float(c[idx].double().mean()) for name, c in correct.items()},
        }
        for name, pred in y_pred.items():
            row[f"{name}_fit"] = fit_report(y_true[idx], pred[idx], head)
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def _check(label: str, got: float, want: float, n: int, failures: list[str]) -> dict[str, Any]:
    # Recomputed on CPU vs the run's GPU float64: identical up to an argmax tie,
    # so allow at most one flipped example and no more.
    tol = 1.0 / n + 1e-12
    ok = abs(got - want) <= tol
    if not ok:
        failures.append(f"{label}: recomputed {got:.6f} vs recorded {want:.6f}")
    return {"recomputed": got, "recorded": want, "ok": ok}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", type=Path, required=True, help="A block_ridge run with <task>_steer_artifacts.pt.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid-root", type=Path, default=GRID_ROOT,
                   help="Where the original grid runs live; used to cross-check against the block_ridge "
                        "cell of the same name and its global_ridge twin, when present.")
    args = p.parse_args()

    exp_dir = args.exp_dir.resolve()
    cfg = json.loads((exp_dir / "config.json").read_text())
    task = str(cfg["tasks"])
    mp = cfg["method_params"]
    if mp.get("stage_2_strategy") != "block_ridge":
        raise ValueError(f"{exp_dir} is stage_2_strategy={mp.get('stage_2_strategy')!r}, not block_ridge")
    art_path = exp_dir / f"{task}_steer_artifacts.pt"
    if not art_path.is_file():
        raise FileNotFoundError(f"{art_path} missing -- rerun via scripts/slurm/submit_block_ridge_diag.py")
    art = torch.load(art_path, map_location="cpu", weights_only=False)
    state = art["stage2_state"]
    if state.get("kind") != "block_ridge":
        raise ValueError(f"{art_path} holds a {state.get('kind')!r} stage 2, not block_ridge")

    logit_map = art["stage1_logit_map"].double()
    p_b = art["stage1_pinv_w_b"].double()
    w_b = art["w_b"].double()
    b_b = None if art["b_b"] is None else art["b_b"].double()
    selected = art["selected"].long()
    coefficients = [c.double() for c in state["coefficients"]]
    mask_class = sorted(set(int(c) for c in art["head_class_ids"]))  # text_loaders.py: sorted(set(mapped_class_ids))
    head = w_b[mask_class]  # the logit rows this task's argmax can see
    mode = str(state["block_ridge_mode"])
    print(f"[diag] {exp_dir.name}: task={task} |selected|={selected.numel()} blocks={len(coefficients)} "
          f"grouping={state['block_group_strategy']} mode={mode} mask_class={mask_class}")

    train = _load_split(cfg, task, "train")
    test = _load_split(cfg, task, "test")
    n_train = int(train["features_B"].shape[0])
    unselected = torch.ones(n_train, dtype=torch.bool)
    unselected[selected] = False
    unselected = unselected.nonzero().squeeze(1)

    def target_of(data: Mapping[str, Any], rows: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """(per-block targets [n, blocks, d_B], total target [n, d_B]) -- prepare()'s formulas."""
        pick = (lambda t: t[rows]) if rows is not None else (lambda t: t)
        per_block = pick(data["delta_A_blocks"]).double() @ logit_map.T @ p_b.T
        total = pick(data["delta_A"]).double() @ logit_map.T @ p_b.T
        return per_block, total

    splits = {
        "train_support": (train, selected),
        "train_unselected": (train, unselected),
        "test": (test, None),
    }
    # Refit global_ridge on the same support with the production _ridge.
    _, train_total_sel = target_of(train, selected)
    global_coef = _ridge(train["features_B"][selected].double(), train_total_sel, float(mp["ridge_lambda"]))

    regression: dict[str, Any] = {}
    cached: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, (data, rows) in splits.items():
        y_blocks, y_total = target_of(data, rows)
        blocks = grouped_blocks(data["features_B_blocks"], state, rows)
        contribs = per_block_contributions(coefficients, blocks)
        pred_total = _predict_block_ridge(coefficients, blocks)
        f_b = (data["features_B"][rows] if rows is not None else data["features_B"]).double()
        pred_global = f_b @ global_coef
        # Sanity: per-block deltas must sum to the full delta, or block_ridge is
        # fitting a different quantity than the oracle scores.
        block_sum_gap = float((y_blocks.sum(dim=1) - y_total).norm() / y_total.norm().clamp_min(1e-300))

        total_energy = float(y_blocks.square().sum())
        per_block = []
        for b, contrib in enumerate(contribs):
            y_b = y_blocks[:, b]
            per_block.append({
                "block": b,
                "target_energy_fraction": float(y_b.square().sum()) / total_energy if total_energy > 0 else float("nan"),
                "input_dim": int(blocks[b].shape[1]),
                **fit_report(y_b, contrib, head, per_example=False),
            })
        regression[split_name] = {
            "n": int(y_total.shape[0]),
            "block_sum_vs_full_delta_rel_err": block_sum_gap,
            "block_ridge_total": fit_report(y_total, pred_total, head),
            "global_ridge_total": fit_report(y_total, pred_global, head),
            "block_ridge_per_block": per_block,
        }
        cached[split_name] = {"y_total": y_total, "pred_block": pred_total, "pred_global": pred_global, "f_b": f_b}

    # ---- validation gate ---------------------------------------------------
    t = cached["test"]
    labels = test["y_A"].long()
    n_test = int(labels.numel())
    recomputed = {
        "stage0_test_acc": _accuracy(t["f_b"], w_b, b_b, labels, mask_class=mask_class),
        "stage1_test_acc": _accuracy(t["f_b"] + t["y_total"], w_b, b_b, labels, mask_class=mask_class),
        "stage2_test_acc": _accuracy(t["f_b"] + t["pred_block"], w_b, b_b, labels, mask_class=mask_class),
    }
    failures: list[str] = []
    gate: dict[str, Any] = {"this_run": {}, "grid_twin": None, "global_ridge_twin": None}
    for key, value in recomputed.items():
        gate["this_run"][key] = _check(f"this run {key}", value, float(art["diagnostics"][key]), n_test, failures)

    twin = args.grid_root / exp_dir.name
    if twin.resolve() != exp_dir and (twin / "summary.json").is_file():
        recorded = _recorded_diagnostics(twin, task)
        gate["grid_twin"] = {"path": str(twin)}
        for key, value in recomputed.items():
            gate["grid_twin"][key] = _check(f"grid twin {key}", value, float(recorded[key]), n_test, failures)
    global_acc = _accuracy(t["f_b"] + t["pred_global"], w_b, b_b, labels, mask_class=mask_class)
    gr_twin = args.grid_root / exp_dir.name.replace("_block_ridge_", "_global_ridge_")
    if (gr_twin / "summary.json").is_file():
        recorded = _recorded_diagnostics(gr_twin, task)
        gate["global_ridge_twin"] = {
            "path": str(gr_twin),
            "stage2_test_acc": _check("global_ridge twin stage2_test_acc", global_acc, float(recorded["stage2_test_acc"]), n_test, failures),
        }
    if failures:
        print("[diag] VALIDATION FAILED -- the artifacts/cache do not reproduce the recorded run:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    # ---- hard vs easy (test only) ------------------------------------------
    pred = {
        "stage0": predictions(t["f_b"], w_b, b_b, mask_class),
        "stage1_oracle": predictions(t["f_b"] + t["y_total"], w_b, b_b, mask_class),
        "stage2_block_ridge": predictions(t["f_b"] + t["pred_block"], w_b, b_b, mask_class),
    }
    correct = {k: v.eq(labels) for k, v in pred.items()}
    # The per-example path must agree with _accuracy, or every bucket below is suspect.
    for key, name in (("stage0_test_acc", "stage0"), ("stage1_test_acc", "stage1_oracle"), ("stage2_test_acc", "stage2_block_ridge")):
        if not math.isclose(float(correct[name].double().mean()), recomputed[key], abs_tol=1e-12):
            raise AssertionError(f"per-example {name} disagrees with _accuracy")
    correct_gr = predictions(t["f_b"] + t["pred_global"], w_b, b_b, mask_class).eq(labels)

    fit_preds = {"block_ridge": t["pred_block"], "global_ridge": t["pred_global"]}
    all_correct = {**correct, "global_ridge": correct_gr}
    hard_easy = {
        "crosstab_stage0_stage1_stage2": correctness_crosstab(correct),
        "block_vs_global_ridge": correctness_crosstab({"block_ridge": correct["stage2_block_ridge"], "global_ridge": correct_gr}),
        "by_logit_correction_norm": quantile_report(
            (t["y_total"] @ head.T).norm(dim=1), correct=all_correct, y_true=t["y_total"], y_pred=fit_preds, head=head),
        "by_stage0_margin": quantile_report(
            margins(t["f_b"], w_b, b_b, labels, mask_class), correct=all_correct, y_true=t["y_total"], y_pred=fit_preds, head=head),
    }

    report = {
        "exp_dir": str(exp_dir),
        "task": task,
        "seed": cfg.get("seed"),
        "method_params": mp,
        "stage2_state_meta": {k: v for k, v in state.items() if k != "coefficients"},
        "num_selected": int(selected.numel()),
        "mask_class": mask_class,
        "validation": gate,
        "accuracy": {**recomputed, "global_ridge_refit_test_acc": global_acc},
        "regression": regression,
        "hard_vs_easy": hard_easy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"[diag] wrote {args.output}")
    return 0


def _print_summary(r: Mapping[str, Any]) -> None:
    acc = r["accuracy"]
    print(f"\n== {r['task']} seed={r['seed']}  (validation passed)")
    print(f"acc  stage0 {acc['stage0_test_acc']:.4f}  stage1 oracle {acc['stage1_test_acc']:.4f}  "
          f"block_ridge {acc['stage2_test_acc']:.4f}  global_ridge {acc['global_ridge_refit_test_acc']:.4f}")
    print(f"{'split':<17}{'model':<13}{'featR2vw':>9}{'logitR2vw':>10}{'logitR2ua':>10}{'cos':>7}{'|p|/|y|':>8}")
    for split, reg in r["regression"].items():
        for model in ("block_ridge", "global_ridge"):
            f = reg[f"{model}_total"]
            print(f"{split:<17}{model:<13}{f['feature_r2']['variance_weighted']:>9.3f}"
                  f"{f['logit_r2']['variance_weighted']:>10.3f}{f['logit_r2']['uniform_average']:>10.3f}"
                  f"{f['logit_agreement']['cosine_mean']:>7.3f}{f['logit_agreement']['norm_ratio_median']:>8.3f}")
    print("per-block logit R2 (vw), train_support / test, and target energy share:")
    for tr, te in zip(r["regression"]["train_support"]["block_ridge_per_block"], r["regression"]["test"]["block_ridge_per_block"]):
        print(f"  block {tr['block']:>2}  {tr['logit_r2']['variance_weighted']:>7.3f} / {te['logit_r2']['variance_weighted']:>7.3f}"
              f"   energy {te['target_energy_fraction']:.3f}")
    print("stage0/stage1/stage2 crosstab (count):")
    for row in r["hard_vs_easy"]["crosstab_stage0_stage1_stage2"]:
        if row["count"]:
            print(f"  s0={row['stage0']:<5} s1={row['stage1_oracle']:<5} s2={row['stage2_block_ridge']:<5} {row['count']:>5}  {row['fraction']:.3f}")
    for key in ("by_logit_correction_norm", "by_stage0_margin"):
        print(f"{key} (quartile: acc s0/s1/block/global, block logit R2, block |p|/|y|):")
        for row in r["hard_vs_easy"][key]:
            a = row["accuracy"]
            f = row["block_ridge_fit"]
            print(f"  Q{row['quantile']} [{row['key_min']:.2f},{row['key_max']:.2f}]  "
                  f"{a['stage0']:.3f}/{a['stage1_oracle']:.3f}/{a['stage2_block_ridge']:.3f}/{a['global_ridge']:.3f}  "
                  f"R2 {f['logit_r2']['variance_weighted']:.3f}  ratio {f['logit_agreement']['norm_ratio_median']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
