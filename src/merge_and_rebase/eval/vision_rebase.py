from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..alpha_search import PerTaskAlphaTracker, average_scores
from ..cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_logging_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    eval_task_top1,
    humanize,
    patch_base_for_attn,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..io.peft_helpers import normalize_attn_patch_cfg
from ..merge.methods._common import axpy_state_dict
from ..merge.task_vectors import TaskVector
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..rebase import get_method, list_methods
from ..rebase.runtime import (
    format_rebase_method_label,
    resolve_rebase_method_config,
    transport_vision_task_vector,
)
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .datasets.vision8_14_20 import SUITES
from .print_utils import pretty_print_task_accuracies


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser("Rebase task vectors from source base A to target base B and evaluate")

        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

        p.add_argument("--source-clip-model", type=str, default=None)
        p.add_argument("--source-clip-pretrained", type=str, default=None)
        p.add_argument("--target-clip-model", type=str, default=None)
        p.add_argument("--target-clip-pretrained", type=str, default=None)

        add_device_dtype_args(p, device_default=None, dtype_default=None)

        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--val-fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--no-humanize", action="store_true", default=None, help="Use raw classnames.")

        p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
        p.add_argument("--weights", type=float, nargs="*", default=None)
        p.add_argument("--strict-load", action="store_true", default=None)
        p.add_argument("--method", type=str, choices=list_methods(), default=None)
        p.add_argument("--method-params", type=str, default=None, help="JSON object for rebase-method kwargs.")

        p.add_argument("--mask-mode", type=str, default=None, choices=["normal", "force"])
        p.add_argument("--vote", type=str, default=None, choices=["mean", "majority", "max"])
        p.add_argument("--grad-batch-size", type=int, default=None)
        p.add_argument("--grad-imgs-per-class", type=int, default=None)
        p.add_argument("--grad-num-batches", type=int, default=None)

        add_alpha_args(
            p,
            alpha_default=None,
            alpha_min_default=None,
            alpha_max_default=None,
            alpha_step_default=None,
            alpha_search_default=None,
            alpha_search_help="Enable linear search over alpha.",
        )
        p.add_argument("--alpha-selection", type=str, choices=["shared", "per_task"], default=None)
        p.add_argument("--alpha-patience", type=int, default=None)
        p.add_argument("--save-merged", type=str, default=None)
        p.add_argument("--save-transported-tvs-dir", type=str, default=None)
        add_logging_args(p)

        args = p.parse_args()
        method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")

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
            "strict_load": args.strict_load,
            "method": args.method,
            "method_params": method_params_cli,
            "mask_mode": args.mask_mode,
            "vote": args.vote,
            "grad_batch_size": args.grad_batch_size,
            "grad_imgs_per_class": args.grad_imgs_per_class,
            "grad_num_batches": args.grad_num_batches,
            "alpha_search": getattr(args, "alpha_search", None),
            "alpha_selection": getattr(args, "alpha_selection", None),
            "alpha_patience": args.alpha_patience,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_step": args.alpha_step,
            "alpha": args.alpha,
            "save_merged": args.save_merged,
            "save_transported_tvs_dir": args.save_transported_tvs_dir,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg

        method_name, method_params = resolve_rebase_method_config(cfg)
        method = get_method(method_name)
        method_label = format_rebase_method_label(method_name, method_params)
        strict_load = bool(cfg.get("strict_load", False))
        device = str(cfg.get("device", "cuda"))

        grad_batch_size = int(cfg["grad_batch_size"]) if cfg.get("grad_batch_size") is not None else None
        grad_imgs_per_class = int(cfg["grad_imgs_per_class"]) if cfg.get("grad_imgs_per_class") is not None else None
        grad_num_batches = int(cfg["grad_num_batches"]) if cfg.get("grad_num_batches") is not None else None

        alpha_search = bool(cfg.get("alpha_search", False))
        alpha_patience_raw = cfg.get("alpha_patience", 0)
        alpha_patience = int(alpha_patience_raw) if alpha_patience_raw is not None else 0
        if alpha_patience < 0:
            raise ValueError("alpha_patience must be >= 0")

        if alpha_search:
            a_min = float(cfg.get("alpha_min", 0.0))
            a_max = float(cfg.get("alpha_max", 2.0))
            a_step = float(cfg.get("alpha_step", 0.1))
            alphas = torch.arange(a_min, a_max + 1e-9, a_step).tolist()
        else:
            alphas = [float(cfg.get("alpha", 1.0))]

        alpha_selection = str(cfg.get("alpha_selection", "shared")).strip().lower()
        if alpha_selection not in {"shared", "per_task"}:
            raise ValueError("alpha_selection must be one of: shared, per_task")

        positive_alphas = [float(alpha) for alpha in alphas if float(alpha) > 0.0]
        if alpha_search and not positive_alphas:
            raise ValueError("alpha_search requires at least one alpha > 0.")

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

        run_summary_path = default_summary_path(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            default_parent=(Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None),
        )
        run_logger = start_run(
            entrypoint="eval.vision_rebase",
            logging_cfg=logging_cfg,
            summary_path=run_summary_path,
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": suite_name,
                "tasks": tasks,
                "summary_path": str(run_summary_path),
            },
        )

        tuned_by_task = cfg.get("tuned_ckpts", None)
        if not tuned_by_task:
            raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")

        merge_weights = cfg.get("weights", None)
        if merge_weights is None:
            merge_weights = [1.0] * len(tasks)
        merge_weights = [float(w) for w in merge_weights]

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

        attn_patch_cfg_raw = cfg.get("attn_patch_cfg", None)
        if attn_patch_cfg_raw is not None and not isinstance(attn_patch_cfg_raw, dict):
            raise ValueError("config['attn_patch_cfg'] must be a dict when provided.")
        patch_attn_before_rebase = bool(cfg.get("patched_attn", attn_patch_cfg_raw is not None))
        attn_patch_cfg = normalize_attn_patch_cfg(attn_patch_cfg_raw) if patch_attn_before_rebase else None

        if patch_attn_before_rebase:
            print(f"Patching source/target attention before rebase: {attn_patch_cfg}")
            source_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_source,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
            target_base_sd = to_cpu_fp32(
                patch_base_for_attn(
                    clf=clf_target,
                    base_ckpt=None,
                    strict_load=strict_load,
                    attn_patch_cfg=attn_patch_cfg,
                )
            )
        else:
            source_base_sd = to_cpu_fp32({k: v for k, v in clf_source.model.state_dict().items()})
            target_base_sd = to_cpu_fp32({k: v for k, v in clf_target.model.state_dict().items()})

        tuned_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for t in tasks:
            ckpt_path = str(tuned_by_task[t])
            sd = load_ckpt(ckpt_path)
            aligned = align_to_base_keys(sd, source_base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from tuned checkpoint aligned to source base keys for task '{t}': {ckpt_path}. "
                    f"{'The base model was attention-patched before rebase, so the checkpoint must use the same patched keyspace.' if patch_attn_before_rebase else ''}"
                )
            tuned_sds_by_task[t] = to_cpu_fp32(aligned)
            print(f"Loaded tuned checkpoint for '{t}' ({len(aligned)} keys)")

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

        def _legacy_visual_key(key: str) -> str | None:
            if not key.startswith("visual."):
                return None
            out = key[len("visual.") :]
            replacements = (
                (".attn.q_proj.", ".attn.q."),
                (".attn.k_proj.", ".attn.k."),
                (".attn.v_proj.", ".attn.v."),
                (".attn.out_proj.", ".attn.proj."),
                (".mlp.c_fc.", ".mlp.fc1."),
                (".mlp.c_proj.", ".mlp.fc2."),
            )
            for src, dst in replacements:
                out = out.replace(src, dst)
            return out

        def _legacy_visual_delta(delta: dict[str, torch.Tensor], *, drop_conv1: bool = False) -> dict[str, torch.Tensor]:
            out: dict[str, torch.Tensor] = {}
            for key, value in delta.items():
                legacy_key = _legacy_visual_key(key)
                if legacy_key is None:
                    continue
                if drop_conv1 and legacy_key == "conv1.weight":
                    continue
                out[legacy_key] = value.detach().to(device="cpu", dtype=torch.float32)
            return out

        use_humanized_classnames = not bool(cfg.get("no_humanize", True))
        print(f"Classname mode: {'humanized' if use_humanized_classnames else 'raw'}")
        print(f"Rebase method: {method_label}")

        per_task: list[dict[str, Any]] = []
        transported_deltas: list[dict[str, torch.Tensor]] = []
        original_deltas: list[dict[str, torch.Tensor]] = []
        transported_artifacts: dict[str, list[str]] = {}

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

            source_loaders = None
            if getattr(method, "name", None) == "theseus":
                source_loaders = build_vision_loaders(
                    hf_ds=hf_ds,
                    hf_path=hf_path,
                    preprocess=clf_source.preprocess,
                    ft_epochs=1,
                    split_map=split_map,
                    batch_size=int(cfg.get("batch_size", 128)),
                    num_workers=int(cfg.get("num_workers", 6)),
                    pin_memory=True,
                    val_fraction=float(cfg.get("val_fraction", 0.1)),
                    seed=int(cfg.get("seed", 42)),
                )

            print(f"\n--- Transporting '{task}' with method '{method.name}' ---")
            transported_delta = transport_vision_task_vector(
                method=method,
                source_base=source_base_sd,
                target_base=target_base_sd,
                delta=tvs_by_task[task].delta,
                clf_source=clf_source,
                clf_target=clf_target,
                task_name=task,
                loaders=loaders,
                source_loaders=source_loaders,
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
            run_logger.log_event(
                "transport_task_end",
                metrics={f"rebase/{task}/transported_param_count": float(len(transported_delta))},
                context={"task": task, "method": method.name},
            )

            save_transport_dir = cfg.get("save_transported_tvs_dir", None)
            if save_transport_dir:
                os.makedirs(save_transport_dir, exist_ok=True)
                native_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_native.pt")
                legacy_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported_legacy_visual.pt")
                legacy_no_conv1_path = os.path.join(
                    save_transport_dir, f"{task}_{method.name}_transported_legacy_visual_no_conv1.pt"
                )
                torch.save(to_cpu_fp32(transported_delta), native_path)
                torch.save(_legacy_visual_delta(transported_delta), legacy_path)
                torch.save(_legacy_visual_delta(transported_delta, drop_conv1=True), legacy_no_conv1_path)
                print(f"  {task}: saved transported TV -> {native_path}")
                print(f"  {task}: saved legacy visual TV -> {legacy_path}")
                print(f"  {task}: saved legacy visual TV without conv1 -> {legacy_no_conv1_path}")
                transported_artifacts[task] = [native_path, legacy_path, legacy_no_conv1_path]

        def _scale_delta(delta_sd: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
            w = float(weight)
            if w == 1.0:
                return delta_sd
            return {k: (v * w) for k, v in delta_sd.items()}

        rebased_deltas = [_scale_delta(d, w) for d, w in zip(transported_deltas, merge_weights, strict=True)]
        untransported_deltas = [_scale_delta(d, w) for d, w in zip(original_deltas, merge_weights, strict=True)]
        print(f"Prepared {len(tasks)} transported deltas (task-independent alpha mode)")

        def _check_untransported_compatibility(
            base_sd: dict[str, torch.Tensor],
            delta_sd: dict[str, torch.Tensor],
        ) -> tuple[bool, list[str]]:
            issues: list[str] = []
            for k, d in delta_sd.items():
                b = base_sd.get(k)
                if b is None:
                    issues.append(f"{k}: missing in target base")
                    continue
                if tuple(d.shape) != tuple(b.shape):
                    issues.append(f"{k}: delta shape {tuple(d.shape)} != target shape {tuple(b.shape)}")
            return (len(issues) == 0), issues

        can_eval_untransported_by_task: list[bool] = []
        for task_name, delta_sd in zip(tasks, untransported_deltas, strict=True):
            enabled, issues = _check_untransported_compatibility(target_base_sd, delta_sd)
            can_eval_untransported_by_task.append(enabled)
            if enabled:
                print(f"Untransported baseline for '{task_name}': enabled.")
            else:
                print(f"Untransported baseline for '{task_name}': skipped (incompatible with target model).")
                for msg in issues[:3]:
                    print(f"  - {msg}")
                if len(issues) > 3:
                    print(f"  - ... and {len(issues) - 3} more incompatibilities")

        def _eval_task(item: dict[str, Any], split: str) -> float:
            return float(
                eval_task_top1(
                    clf=clf_target,
                    loaders=item["loaders"],
                    classnames=list(item["classnames"]),
                    build_cfg_task=item["build_cfg_task"],
                    device=device,
                    split=split,
                )
            )

        def _eval_all_tasks(split: str) -> list[float]:
            return [_eval_task(item, split) for item in per_task]

        def _average_defined(values: list[float]) -> float:
            defined = [float(v) for v in values if float(v) == float(v)]
            return average_scores(defined) if defined else float("nan")

        def _norm_acc(result_acc: float, baseline_acc: float) -> float:
            baseline_acc = float(baseline_acc)
            if baseline_acc != baseline_acc or baseline_acc <= 0.0:
                return float("nan")
            return float(result_acc) / baseline_acc

        if all(can_eval_untransported_by_task):
            baseline_label = "untransported"
        elif any(can_eval_untransported_by_task):
            baseline_label = "mixed_baseline"
        else:
            baseline_label = "target_zeroshot"
        result_label = "rebased"
        task_col = max(max((len(str(item["task"])) for item in per_task), default=4), len("task"), len("avg"))
        metric_col = max(12, len(baseline_label) + 2, len(result_label) + 2, len("norm") + 2)

        baseline_cache_zeroshot: dict[str, list[float]] = {}

        if baseline_label == "untransported":
            print("Using untransported baseline evaluation for all tasks.")
        elif baseline_label == "mixed_baseline":
            print("Using mixed baseline evaluation: untransported where compatible, target zeroshot otherwise.")
        else:
            print("Using target zeroshot baseline for all tasks.")

        def _eval_zeroshot_all_tasks(split: str) -> list[float]:
            if split not in baseline_cache_zeroshot:
                load_into_model(clf_target.model, target_base_sd, strict=strict_load)
                baseline_cache_zeroshot[split] = _eval_all_tasks(split)
            return list(baseline_cache_zeroshot[split])

        def _eval_baseline_task(split: str, idx: int, alpha: float) -> float:
            if can_eval_untransported_by_task[idx]:
                baseline_sd = axpy_state_dict(target_base_sd, untransported_deltas[idx], alpha=float(alpha))
                load_into_model(clf_target.model, baseline_sd, strict=strict_load)
                del baseline_sd
                return _eval_task(per_task[idx], split)
            return _eval_zeroshot_all_tasks(split)[idx]

        def _eval_baseline_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            return {idx: _eval_baseline_task(split, idx, alpha) for idx in indices}

        def _eval_rebased_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            out: dict[int, float] = {}
            for idx in indices:
                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=float(alpha))
                load_into_model(clf_target.model, rebase_sd_task, strict=strict_load)
                del rebase_sd_task
                out[idx] = _eval_task(per_task[idx], split)
            return out

        if alpha_selection == "shared":
            best_rebase_avg = float("-inf")
            best_alpha = float(positive_alphas[0] if positive_alphas else alphas[0])
            sweep_results: list[dict[str, Any]] = []
            shared_bad_steps = 0

            for alpha in alphas:
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: val, mode: shared) ===")

                idxs = list(range(len(per_task)))
                baseline_by_idx = _eval_baseline_task_indices("val", idxs, float(alpha))
                rebase_by_idx = _eval_rebased_task_indices("val", idxs, float(alpha))
                baseline_accs = [baseline_by_idx[i] for i in idxs]
                rebase_accs = [rebase_by_idx[i] for i in idxs]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for i, item in enumerate(per_task):
                    task_name = item["task"]
                    baseline_acc = baseline_accs[i]
                    rebase_acc = rebase_accs[i]
                    norm = _norm_acc(rebase_acc, baseline_acc)
                    print(
                        f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                avg_rebase = average_scores(rebase_accs)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
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
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "baseline_label": baseline_label,
                        "per_task_baseline": {item["task"]: float(baseline_accs[i]) for i, item in enumerate(per_task)},
                        "per_task_rebased": {item["task"]: float(rebase_accs[i]) for i, item in enumerate(per_task)},
                    },
                )

                if float(alpha) > 0.0:
                    eps = 1e-12
                    if avg_rebase > best_rebase_avg + eps:
                        best_rebase_avg = avg_rebase
                        best_alpha = float(alpha)
                        shared_bad_steps = 0
                    elif avg_rebase + eps >= best_rebase_avg:
                        shared_bad_steps = 0
                    elif len(positive_alphas) > 1:
                        shared_bad_steps += 1
                        print(
                            f"  (alpha={alpha:.3f} fell below best shared avg {best_rebase_avg:.6f}; "
                            f"bad_steps={shared_bad_steps}/{alpha_patience + 1})"
                        )
                        if shared_bad_steps > alpha_patience:
                            break

            print("\n=== Alpha search summary (shared) ===")
            for r in sweep_results:
                a = r["alpha"]
                avg_r = average_scores(r["rebase_accs"])
                avg_b = _average_defined(r["baseline_accs"])
                print(f"  alpha={a:.3f}  {baseline_label}={avg_b:.6f}  {result_label}={avg_r:.6f}")
            print(f"\nBest alpha: {best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f})")

            print(f"\n(Re-running best alpha ({best_alpha:.3f}) on test split)")
            all_indices = list(range(len(per_task)))
            baseline_test_by_idx = _eval_baseline_task_indices("test", all_indices, float(best_alpha))
            rebase_test_by_idx = _eval_rebased_task_indices("test", all_indices, float(best_alpha))
            baseline_test_accs = [baseline_test_by_idx[i] for i in all_indices]
            rebase_test_accs = [rebase_test_by_idx[i] for i in all_indices]
            selected_alpha_by_task = [float(best_alpha)] * len(per_task)

        else:
            tracker = PerTaskAlphaTracker(
                task_names=[str(item["task"]) for item in per_task],
                initial_alpha=float(positive_alphas[0] if positive_alphas else alphas[0]),
                patience=alpha_patience,
            )
            sweep_results = []

            for alpha in alphas:
                active_indices = tracker.active_indices()
                if not active_indices:
                    print("\nAll tasks have early-stopped; ending per-task alpha sweep.")
                    break

                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: val, mode: per_task) ===")

                baseline_by_idx = _eval_baseline_task_indices("val", active_indices, float(alpha))
                rebase_by_idx = _eval_rebased_task_indices("val", active_indices, float(alpha))

                baseline_accs = [baseline_by_idx[idx] for idx in active_indices]
                rebase_accs = [rebase_by_idx[idx] for idx in active_indices]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for idx, baseline_acc, rebase_acc in zip(active_indices, baseline_accs, rebase_accs, strict=True):
                    task_name = per_task[idx]["task"]
                    norm = _norm_acc(rebase_acc, baseline_acc)
                    print(
                        f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                avg_rebase = average_scores(rebase_accs)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "active_tasks": [per_task[idx]["task"] for idx in active_indices],
                        "per_task_baseline": {per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in active_indices},
                        "per_task_rebased": {per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in active_indices},
                    },
                )

                stopped_indices: list[int] = []
                if float(alpha) > 0.0:
                    stopped_indices = tracker.update(
                        alpha=float(alpha),
                        indices=active_indices,
                        baseline_accs=baseline_accs,
                        rebase_accs=rebase_accs,
                    )
                    if stopped_indices:
                        stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_indices)
                        print(f"  Early-stopping tasks at alpha={alpha:.3f}: {stopped_names}")

                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "active_indices": list(active_indices),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                    }
                )

            print("\n=== Alpha search summary (per-task) ===")
            for idx, item in enumerate(per_task):
                print(
                    f"  {item['task']}: best_alpha={tracker.best_alpha[idx]:.3f}  "
                    f"best_val={tracker.best_rebase_acc[idx]:.6f}"
                )
            print(f"\nAvg per-task best val acc: {tracker.best_avg():.6f}")

            print("\n(Re-running per-task best alphas on test split)")
            baseline_test_accs = []
            rebase_test_accs = []
            selected_alpha_by_task = []
            for idx, item in enumerate(per_task):
                task_alpha = float(tracker.best_alpha[idx])
                selected_alpha_by_task.append(task_alpha)
                print(f"  {item['task']}: alpha={task_alpha:.3f}")

                baseline_test_accs.append(_eval_baseline_task("test", idx, task_alpha))

                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=task_alpha)
                load_into_model(clf_target.model, rebase_sd_task, strict=strict_load)
                del rebase_sd_task
                rebase_test_accs.append(_eval_task(item, "test"))
            best_alpha = float(sum(selected_alpha_by_task) / max(1, len(selected_alpha_by_task)))

        norm_accs = [_norm_acc(r, b) for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

        pretty_print_task_accuracies(
            suite_name,
            f"{method_label}, alpha={alpha_selection}",
            f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
            per_task,
            rebase_test_accs,
            norm_accs,
            single_accs=baseline_test_accs,
            baseline_label=baseline_label,
            result_label=result_label,
        )

        if alpha_selection == "per_task":
            print("\nSelected test-time alpha by task:")
            for item, alpha in zip(per_task, selected_alpha_by_task, strict=True):
                print(f"  {item['task']}: {alpha:.3f}")

        saved_merged_path: str | None = None
        if cfg.get("save_merged"):
            print(
                "save_merged was requested, but task-independent alpha mode does not produce a single merged checkpoint; skipping save."
            )

        final_summary = {
            "suite": suite_name,
            "tasks": tasks,
            "method": method.name,
            "method_label": method_label,
            "alpha_selection": alpha_selection,
            "best_alpha": float(best_alpha),
            "baseline_label": baseline_label,
            "test_results": {
                "per_task_baseline": {item["task"]: float(baseline_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_rebased": {item["task"]: float(rebase_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_norm": {item["task"]: float(norm_accs[i]) for i, item in enumerate(per_task)},
                "avg_rebased": float(sum(rebase_test_accs) / len(rebase_test_accs)),
                "avg_norm": float(sum(norm_accs) / len(norm_accs)),
            },
            "selected_alpha_by_task": {item["task"]: float(selected_alpha_by_task[i]) for i, item in enumerate(per_task)},
            "transported_artifacts": transported_artifacts,
            "saved_merged_path": saved_merged_path,
        }
        run_logger.log_summary(final_summary)
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
