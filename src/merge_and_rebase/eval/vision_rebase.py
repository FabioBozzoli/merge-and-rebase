from __future__ import annotations

import argparse
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    compose_weighted_deltas,
    eval_task_top1,
    humanize,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..merge.methods._common import axpy_state_dict
from ..merge.task_vectors import TaskVector
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..rebase import get_method, list_methods
from .cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_suite_arg,
    add_tasks_arg,
    merge_non_none,
    parse_json_object_arg,
)
from .datasets.vision8_14_20 import SUITES
from .print_utils import pretty_print_task_accuracies
from .rebase_utils import (
    format_rebase_method_label,
    resolve_rebase_method_config,
    transport_vision_task_vector,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser("Rebase task vectors from source base A to target base B and evaluate")

    add_config_arg(p)
    add_suite_arg(p, choices=sorted(SUITES.keys()))
    add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

    # Source model (A) — pretrained base from which task vectors are computed
    p.add_argument("--source-clip-model", type=str, default=None)
    p.add_argument("--source-clip-pretrained", type=str, default=None)

    # Target model (B) — pretrained base on which we apply the transported vectors
    p.add_argument("--target-clip-model", type=str, default=None)
    p.add_argument("--target-clip-pretrained", type=str, default=None)

    add_device_dtype_args(p, device_default="cuda", dtype_default=None)

    # Eval
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-humanize",
        action="store_true",
        default=None,
        help="Use raw classnames.",
    )

    # Tuned checkpoints (finetuned w.r.t. source base A)
    p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
    p.add_argument("--weights", type=float, nargs="*", default=None)
    p.add_argument("--strict-load", action="store_true")
    p.add_argument("--method", type=str, choices=list_methods(), default=None)
    p.add_argument("--method-params", type=str, default=None, help="JSON object for rebase-method kwargs.")

    # GradFix knobs
    p.add_argument(
        "--mask-mode",
        type=str,
        default=None,
        choices=["normal", "force"],
        help="GradFix mask mode. Default: normal.",
    )
    p.add_argument(
        "--vote",
        type=str,
        default=None,
        choices=["mean", "majority", "max"],
        help="Gradient sign voting mode. Use majority for majority-vote aggregation. Default: mean.",
    )

    # GradFix data sampling
    p.add_argument(
        "--grad-batch-size",
        type=int,
        default=None,
        help="Batch size for gradient-sign computation (default: same as --batch-size).",
    )
    p.add_argument(
        "--grad-imgs-per-class",
        type=int,
        default=None,
        help="Balanced sampling: N images per class for gradient signs.",
    )
    p.add_argument(
        "--grad-num-batches",
        type=int,
        default=None,
        help="Use only the first N batches for gradient signs.",
    )

    # Alpha search (defaults are None so CLI doesn't override JSON config)
    add_alpha_args(
        p,
        alpha_default=None,
        alpha_min_default=None,
        alpha_max_default=None,
        alpha_step_default=None,
        alpha_search_default=None,
        alpha_search_help="Enable linear search over alpha.",
    )

    p.add_argument("--save-merged", type=str, default=None)

    args = p.parse_args()
    method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")

    # ---------------------------------------------------------------
    # Config resolution: JSON base + CLI overrides
    # ---------------------------------------------------------------
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)

    cli: dict[str, Any] = {
        "source_clip_model": args.source_clip_model,
        "source_clip_pretrained": args.source_clip_pretrained,
        "target_clip_model": args.target_clip_model,
        "target_clip_pretrained": args.target_clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_humanize": args.no_humanize,
        "suite": getattr(args, "suite", None),
        "tasks": getattr(args, "tasks", None),
        "device": args.device,
        "dtype": args.dtype,
        "tuned_ckpts": args.tuned_ckpts,
        "weights": args.weights,
        "strict_load": bool(args.strict_load),
        "method": args.method,
        "method_params": method_params_cli,
        "mask_mode": args.mask_mode,
        "vote": args.vote,
        "grad_batch_size": args.grad_batch_size,
        "grad_imgs_per_class": args.grad_imgs_per_class,
        "grad_num_batches": args.grad_num_batches,
        "alpha_search": getattr(args, "alpha_search", None),
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "alpha": args.alpha,
        "save_merged": args.save_merged,
    }
    cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})

    method_name, method_params = resolve_rebase_method_config(cfg)
    method = get_method(method_name)
    method_label = format_rebase_method_label(method_name, method_params)
    strict_load = bool(cfg.get("strict_load", False))
    device = str(cfg.get("device", "cuda"))

    # GradFix data sampling
    grad_batch_size: int | None = cfg.get("grad_batch_size", None)
    if grad_batch_size is not None:
        grad_batch_size = int(grad_batch_size)
    grad_imgs_per_class: int | None = cfg.get("grad_imgs_per_class", None)
    if grad_imgs_per_class is not None:
        grad_imgs_per_class = int(grad_imgs_per_class)
    grad_num_batches: int | None = cfg.get("grad_num_batches", None)
    if grad_num_batches is not None:
        grad_num_batches = int(grad_num_batches)

    # Alpha grid
    alpha_search = bool(cfg.get("alpha_search", False))
    if alpha_search:
        a_min = float(cfg.get("alpha_min", 0.0))
        a_max = float(cfg.get("alpha_max", 2.0))
        a_step = float(cfg.get("alpha_step", 0.1))
        alphas = torch.arange(a_min, a_max + 1e-9, a_step).tolist()
    else:
        alphas = [float(cfg.get("alpha", 1.0))]

    # Suite / tasks
    suite_name = cfg.get("suite", "vision8")
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]

    tasks_arg = cfg.get("tasks", "all")
    if tasks_arg == "all":
        tasks = list(suite.tasks)
    else:
        tasks = parse_csv(tasks_arg)
        bad = [t for t in tasks if t not in suite.tasks]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")

    tuned_by_task = cfg.get("tuned_ckpts", None)
    if not tuned_by_task:
        raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")

    merge_weights = cfg.get("weights", None)
    if merge_weights is None:
        merge_weights = [1.0] * len(tasks)
    merge_weights = [float(w) for w in merge_weights]

    # ---------------------------------------------------------------
    # Build source (A) and target (B) models
    # ---------------------------------------------------------------
    source_cfg = OpenClipBuildConfig(
        model_name=cfg.get("source_clip_model", "ViT-B-32"),
        pretrained=cfg.get("source_clip_pretrained", "openai"),
        device=device,
        dtype=cfg.get("dtype", None),
    )
    target_cfg = OpenClipBuildConfig(
        model_name=cfg.get("target_clip_model", "ViT-B-32"),
        pretrained=cfg.get("target_clip_pretrained", "laion2b_s34b_b79k"),
        device=device,
        dtype=cfg.get("dtype", None),
    )

    print(f"Source model (A): {source_cfg.model_name} / {source_cfg.pretrained}")
    print(f"Target model (B): {target_cfg.model_name} / {target_cfg.pretrained}")

    clf_source = OpenClipClassifier.build(source_cfg)
    clf_target = OpenClipClassifier.build(target_cfg)

    source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})
    target_base_sd = to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()})

    # ---------------------------------------------------------------
    # Load tuned checkpoints (assumed finetuned w.r.t. A) → task vectors
    # ---------------------------------------------------------------
    tuned_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
    for t in tasks:
        ckpt_path = str(tuned_by_task[t])
        sd = load_ckpt(ckpt_path)
        aligned = align_to_base_keys(sd, source_base_sd)
        if not aligned:
            raise ValueError(
                f"No tensors from tuned checkpoint aligned to source base keys for task '{t}': {ckpt_path}"
            )
        tuned_sds_by_task[t] = to_cpu_fp32(aligned)
        print(f"Loaded tuned checkpoint for '{t}' ({len(aligned)} keys)")

    # Build task vectors: Δ_i = tuned_i − source_base
    # Only visual backbone keys — matches original GradFix where
    # TaskVector(model.visual, model_ft.visual) produces visual-only deltas.
    def _visual_only_filter(k: str, v: torch.Tensor) -> bool:
        if not v.is_floating_point():
            return False
        return k.startswith("visual.")

    tvs_by_task: dict[str, TaskVector] = {}
    for t in tasks:
        tvs_by_task[t] = TaskVector.from_checkpoints(
            source_base_sd,
            tuned_sds_by_task[t],
            strict=False,
            key_filter=_visual_only_filter,
        )
    print(f"Computed {len(tvs_by_task)} task vectors (visual-only)")

    # ---------------------------------------------------------------
    # Classname / dataloader setup & task-wise transport
    # ---------------------------------------------------------------
    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    print(f"Classname mode: {'humanized' if use_humanized_classnames else 'raw'}")
    print(f"Rebase method: {method_label}")

    per_task: list[dict[str, Any]] = []
    transported_deltas: list[dict[str, torch.Tensor]] = []
    original_deltas: list[dict[str, torch.Tensor]] = []

    for task in tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf_target.preprocess,
            ft_epochs=1,
            split_map=split_map,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 6)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 42)),
        )

        classnames = list(loaders.classnames)
        if use_humanized_classnames:
            classnames = [humanize(c) for c in classnames]

        templates = get_templates(task)
        if not templates:
            raise ValueError(f"get_templates('{task}') returned empty list")

        build_cfg_task = OpenClipBuildConfig(
            model_name=target_cfg.model_name,
            pretrained=target_cfg.pretrained,
            device=target_cfg.device,
            dtype=target_cfg.dtype,
            prompt_templates=templates,
        )

        per_task.append(
            {
                "task": task,
                "loaders": loaders,
                "classnames": classnames,
                "build_cfg_task": build_cfg_task,
            }
        )

        print(f"\n--- Transporting '{task}' with method '{method.name}' ---")
        transported_delta = transport_vision_task_vector(
            method=method,
            source_base=source_base_sd,
            target_base=target_base_sd,
            delta=tvs_by_task[task].delta,
            clf_target=clf_target,
            task_name=task,
            loaders=loaders,
            classnames=classnames,
            build_cfg_task=build_cfg_task,
            device=device,
            strict=strict_load,
            method_params=method_params,
            grad_batch_size=grad_batch_size,
            grad_imgs_per_class=grad_imgs_per_class,
            grad_num_batches=grad_num_batches,
            num_workers=int(cfg.get("num_workers", 6)),
            seed=int(cfg.get("seed", 42)),
        )
        transported_deltas.append(transported_delta)
        original_deltas.append(tvs_by_task[task].delta)
        print(f"  {task}: transported delta computed for {len(transported_delta)} params")

    composed_rebased = compose_weighted_deltas(transported_deltas, merge_weights)
    composed_unmodified = compose_weighted_deltas(original_deltas, merge_weights)
    print(f"Composed {len(tasks)} deltas (keys={len(composed_rebased)})")

    # ---------------------------------------------------------------
    # Alpha sweep — evaluate BOTH untransported and rebased at each alpha
    # (matches original GradFix: all task-vector variants swept jointly)
    # ---------------------------------------------------------------
    def _eval_all_tasks(split: str) -> list[float]:
        """Evaluate clf_target on every task, return list of accuracies."""
        accs: list[float] = []
        for item in per_task:
            acc = eval_task_top1(
                clf=clf_target,
                loaders=item["loaders"],
                classnames=list(item["classnames"]),
                build_cfg_task=item["build_cfg_task"],
                device=device,
                split=split,
            )
            accs.append(float(acc))
        return accs

    baseline_label = "untransported"
    result_label = "rebased"
    task_col = max(max((len(str(item["task"])) for item in per_task), default=4), len("task"), len("avg"))
    metric_col = max(12, len(baseline_label) + 2, len(result_label) + 2, len("norm") + 2)

    best_rebase_avg = -1.0
    best_alpha = alphas[0]
    sweep_results: list[dict[str, Any]] = []

    for alpha in alphas:
        print(f"\n=== alpha {alpha:.3f} — {method_label} (split: val) ===")

        # --- Untransported baseline ---
        baseline_sd = axpy_state_dict(target_base_sd, composed_unmodified, alpha=float(alpha))
        load_into_model(clf_target.model, baseline_sd, strict=strict_load)
        del baseline_sd
        baseline_accs = _eval_all_tasks("val")

        # --- Rebasing method ---
        rebase_sd = axpy_state_dict(target_base_sd, composed_rebased, alpha=float(alpha))
        load_into_model(clf_target.model, rebase_sd, strict=strict_load)
        del rebase_sd
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()
        rebase_accs = _eval_all_tasks("val")

        print(
            f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
        )
        print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
        for i, item in enumerate(per_task):
            task_name = item["task"]
            baseline_acc = baseline_accs[i]
            rebase_acc = rebase_accs[i]
            norm = (rebase_acc / baseline_acc) if baseline_acc > 0 else 0.0
            print(
                f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
            )

        avg_rebase = sum(rebase_accs) / max(1, len(rebase_accs))
        avg_baseline = sum(baseline_accs) / max(1, len(baseline_accs))
        avg_norm = sum((r / b) if b > 0 else 0.0 for r, b in zip(rebase_accs, baseline_accs, strict=True)) / max(
            1, len(rebase_accs)
        )
        print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
        print(
            f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
        )

        sweep_results.append(
            {
                "alpha": float(alpha),
                "baseline_accs": baseline_accs,
                "rebase_accs": rebase_accs,
            }
        )

        if avg_rebase > best_rebase_avg and alpha > 0:
            best_rebase_avg = avg_rebase
            best_alpha = float(alpha)
        elif avg_rebase <= best_rebase_avg:
            print(f"  (alpha={alpha:.3f} did not beat best rebased avg {best_rebase_avg:.6f})")
            break

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n=== Alpha search summary ===")
    for r in sweep_results:
        a = r["alpha"]
        avg_r = sum(r["rebase_accs"]) / max(1, len(r["rebase_accs"]))
        avg_b = sum(r["baseline_accs"]) / max(1, len(r["baseline_accs"]))
        print(f"  alpha={a:.3f}  {baseline_label}={avg_b:.6f}  {result_label}={avg_r:.6f}")
    print(f"\nBest alpha: {best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f})")

    # ---------------------------------------------------------------
    # Re-run best alpha on test set (both untransported and rebased)
    # ---------------------------------------------------------------
    print(f"\n(Re-running best alpha ({best_alpha:.3f}) on test split)")

    # Untransported baseline on test
    baseline_sd = axpy_state_dict(target_base_sd, composed_unmodified, alpha=float(best_alpha))
    load_into_model(clf_target.model, baseline_sd, strict=strict_load)
    del baseline_sd
    baseline_test_accs = _eval_all_tasks("test")

    # Rebasing method on test
    rebase_sd = axpy_state_dict(target_base_sd, composed_rebased, alpha=float(best_alpha))
    load_into_model(clf_target.model, rebase_sd, strict=strict_load)
    rebase_test_accs = _eval_all_tasks("test")

    # Norm = rebased / untransported at the same alpha
    norm_accs = [(r / b) if b > 0 else 0.0 for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

    pretty_print_task_accuracies(
        suite_name,
        method_label,
        f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
        per_task,
        rebase_test_accs,
        norm_accs,
        single_accs=baseline_test_accs,
        baseline_label=baseline_label,
        result_label=result_label,
    )

    if cfg.get("save_merged"):
        outp = cfg["save_merged"]
        torch.save(rebase_sd, outp)
        print(f"Saved rebased state_dict to {outp}")

    del rebase_sd


if __name__ == "__main__":
    main()
