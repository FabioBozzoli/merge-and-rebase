# src/merge_and_rebase/eval/vision_merge.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.io.peft_helpers import (
    is_peft_adapter_dir_ckpt,
    load_peft_adapter_dir_components,
)
from merge_and_rebase.io.utils import atomic_write_json, read_json_silent
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import (
    TaskAttentionMeta,
    acc_cache_key,
    assert_qkv_patched_before_linearizing,
    build_merged_state_for_alpha,
    ensure_peft_cfg_map,
    eval_norm_accs_for_split,
    eval_task_top1,
    extract_checkpoint_attn_patch_info,
    extract_peft_components,
    get_peft_cfg,
    humanize,
    is_peft_checkpoint,
    materialize_peft_sd_from_adapter,
    maybe_patch_base_for_task_attn,
    to_cpu_fp32,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..merge import subspaces as _subspaces  # noqa: F401
from ..merge.base import PreparedMergeMethod
from ..merge.registry import get_method, list_methods  # methods are registered on import
from ..merge.subspaces.registry import get_subspace, list_subspaces
from ..models.forward_modes import get_forward_mode, list_forward_modes
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from .cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_merge_io_args,
    add_suite_arg,
    add_tasks_arg,
    build_common_eval_overrides,
    build_common_merge_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from .datasets.vision8_14_20 import SUITES
from .print_utils import pretty_print_task_accuracies, print_latex_task_rows

# Backward-compatible test hooks for utility helpers.
_acc_cache_key = acc_cache_key
_assert_qkv_patched_before_linearizing = assert_qkv_patched_before_linearizing
_extract_checkpoint_attn_patch_info = extract_checkpoint_attn_patch_info


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    p = argparse.ArgumentParser("Merge checkpoints with selectable method and evaluate on a vision suite (open_clip)")

    add_config_arg(p)
    add_suite_arg(p, choices=sorted(SUITES.keys()))
    add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

    # open_clip model
    p.add_argument("--clip-model", type=str, default=None)
    p.add_argument("--clip-pretrained", type=str, default=None)
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
        help="Use raw classnames. Default is raw classnames for training/eval consistency.",
    )

    add_merge_io_args(
        p,
        method_choices=list_methods(),
        subspace_choices=list_subspaces(),
        tuned_help="Paths to tuned checkpoints to merge.",
        weights_help="Weights for tuned checkpoints.",
        strict_mode="store_true",
    )

    # Method knobs
    p.add_argument("--keep-ratio", type=float, default=None, help="Keep top |Δ| ratio (task arithmetic)")

    # Single-task accuracy cache (baseline for normalization)
    p.add_argument(
        "--single-acc-cache",
        type=str,
        default="src/.cache/single_task_acc.json",
        help="JSON cache for baseline accuracies keyed by model/pretrain/dataset/baseline mode/checkpoint.",
    )
    p.add_argument(
        "--recompute-single-acc",
        action="store_true",
        help="Ignore cached single-task accuracy and recompute it.",
    )
    p.add_argument(
        "--single-acc-zero-shot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also compute zero-shot base-model accuracy per task (reported only; not used for normalization).",
    )
    p.add_argument(
        "--text-features-source",
        type=str,
        choices=["auto", "zero_shot", "tuned_ckpt"],
        default=None,
        help=(
            "Text features for classification: "
            "'auto' (default: use tuned_text_features from checkpoint when present, else zero-shot), "
            "'zero_shot', or 'tuned_ckpt' (strict)."
        ),
    )

    # Alpha search
    add_alpha_args(
        p,
        alpha_default=None,
        alpha_min_default=0.0,
        alpha_max_default=2.0,
        alpha_step_default=0.1,
        alpha_search_default=None,
        alpha_search_help="Enable linear search over alpha.",
    )
    p.add_argument(
        "--forward-mode",
        type=str,
        choices=["auto", *list_forward_modes()],
        default="auto",
        help="Inference forward mode. 'auto' uses linearized_ntk when all tuned checkpoints have strategy='ntk'.",
    )

    args = p.parse_args()
    method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")

    # Load config file if provided (JSON), then override with CLI where meaningful.
    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)

    cli_overrides: dict[str, Any] = {
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "no_humanize": args.no_humanize,
        "single_acc_cache": args.single_acc_cache,
        "recompute_single_acc": bool(args.recompute_single_acc),
        "single_acc_zero_shot": args.single_acc_zero_shot,
        "text_features_source": args.text_features_source,
        "forward_mode": args.forward_mode,
    }
    cli_overrides = merge_non_none(cli_overrides, build_common_eval_overrides(args))
    cli_overrides = merge_non_none(
        cli_overrides,
        build_common_merge_overrides(args=args, method_params=method_params_cli, strict_as_bool=True),
    )
    cfg = merge_non_none(cfg, cli_overrides)

    alpha_search = bool(cfg.get("alpha_search", False))
    if alpha_search:
        a_min = float(cfg.get("alpha_min", 0.0))
        a_max = float(cfg.get("alpha_max", 2.0))
        a_step = float(cfg.get("alpha_step", 0.1))
        alphas = torch.arange(a_min, a_max + 1e-9, a_step).tolist()
    else:
        alphas = [float(cfg.get("alpha", 1.0))]

    suite_name = cfg.get("suite", "vision8")
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]

    tasks_arg = cfg.get("tasks", "all")
    if tasks_arg == "all":
        tasks = list(suite.tasks)
    else:
        tasks = parse_csv(tasks_arg)
        allowed = set(suite.tasks)
        bad = [t for t in tasks if t not in allowed]
        if bad:
            raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {sorted(allowed)}")

    tuned_by_task = cfg.get("tuned_ckpts", None)
    if not tuned_by_task:
        raise ValueError("You must provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")
    method_params = cfg.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("config['method_params'] must be a dict when provided.")
    strict_load = bool(cfg.get("strict_load", False))
    merge_weights = cfg.get("weights", None)

    # Build open_clip classifier (single instance used for everything to keep memory bounded)
    build_cfg = OpenClipBuildConfig(
        model_name=cfg.get("clip_model", "ViT-B-32"),
        pretrained=cfg.get("clip_pretrained", "openai"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", None),
    )
    clf = OpenClipClassifier.build(build_cfg)

    # Base state dict (CPU) for merging
    base_ckpt = cfg.get("base_ckpt", None)
    if base_ckpt is None:
        print(f"Using open_clip {build_cfg.model_name} (pretrain={build_cfg.pretrained}) weights as base checkpoint")
        base_sd = {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}
    else:
        print(f"Loading base checkpoint from {base_ckpt}")
        base_sd = load_ckpt(str(base_ckpt))

    # Load, align tuned checkpoints once (CPU)
    peft_subspace = str(cfg.get("peft_subspace", "full"))

    tuned_sds_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] = {}
    attn_meta_by_task: dict[str, TaskAttentionMeta] = {}
    strategy_by_task: dict[str, str | None] = {}
    tuned_text_features_by_task: dict[str, torch.Tensor | None] = {}
    peft_cfg_map: dict[str, Any] | None = None
    base_patched_for_attn = False

    for t in tasks:
        ckpt_path = str(tuned_by_task[t])
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Loaded checkpoint for task '{t}' from {ckpt_path}")
        strategy_by_task[t] = obj.get("strategy", None) if isinstance(obj, dict) else None
        attn_meta_by_task[t] = extract_checkpoint_attn_patch_info(obj=obj, ckpt_path=ckpt_path)
        # Merge/eval stays stage-agnostic: it only consumes final tuned_text_features.
        # Stage-specific artifacts (e.g. tuned_prompt_context) are ignored here.
        tuned_text_features_by_task[t] = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(
            obj=obj,
            ckpt_path=ckpt_path,
        )

        is_peft = False
        state: dict[str, torch.Tensor] | None = None
        cfg_map: dict[str, Any] | None = None

        if is_peft_adapter_dir_ckpt(obj):
            state, cfg_map = load_peft_adapter_dir_components(obj["peft_adapter_dir"])
            is_peft = True
        elif is_peft_checkpoint(obj) and isinstance(obj, dict):
            state, cfg_map = extract_peft_components(obj)
            is_peft = True

        if peft_subspace != "full":
            if not is_peft:
                raise ValueError(f"peft_subspace='{peft_subspace}' requires PEFT checkpoints. Got: {ckpt_path}")
            assert state is not None and cfg_map is not None
            peft_cfg_map = ensure_peft_cfg_map(peft_cfg_map, cfg_map)
            peft_state_by_task[t] = state
        else:
            base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
                task_meta=attn_meta_by_task[t],
                base_patched_for_attn=base_patched_for_attn,
                clf=clf,
                base_ckpt=base_ckpt,
                strict_load=strict_load,
                base_sd=base_sd,
            )
            if is_peft:
                assert state is not None and cfg_map is not None
                peft_cfg_map = ensure_peft_cfg_map(peft_cfg_map, cfg_map)

                # if checkpoint is PEFT but we want full, we construct full weights now
                sd = materialize_peft_sd_from_adapter(
                    peft_state=state,
                    base_sd=base_sd,
                    build_cfg=build_cfg,
                    peft_cfg=get_peft_cfg(cfg_map),
                    strict_load=strict_load,
                    patched_attn=attn_meta_by_task[t].patched_attn,
                    attn_patch_cfg=attn_meta_by_task[t].attn_patch_cfg,
                )
            else:
                sd = load_ckpt(ckpt_path)
            aligned = align_to_base_keys(sd, base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from tuned checkpoint aligned to base keys for task '{t}': {ckpt_path}. "
                    "Check checkpoint key prefixes and model compatibility."
                )
            tuned_sds_by_task[t] = to_cpu_fp32(aligned)

    # Also keep a list for merge methods that expect a list
    peft_cfg: dict[str, Any] | None = None
    subspace = None
    subspace_prepared = None
    # Attention mode (patched + linearized-vs-softmax) must be consistent across tuned checkpoints.
    attn_meta_tasks = [t for t in tasks if t in attn_meta_by_task]
    if attn_meta_tasks:
        flag0 = attn_meta_by_task[attn_meta_tasks[0]].patched_attn
        if any(attn_meta_by_task[t].patched_attn != flag0 for t in attn_meta_tasks):
            raise ValueError("Inconsistent patched_attn flags across tuned checkpoints.")
        patch_tasks = [t for t in attn_meta_tasks if attn_meta_by_task[t].patched_attn]
        if patch_tasks:
            patch_cfg0 = attn_meta_by_task[patch_tasks[0]].attn_patch_cfg or {}
            for t in patch_tasks[1:]:
                patch_cfgt = attn_meta_by_task[t].attn_patch_cfg or {}
                if patch_cfgt != patch_cfg0:
                    raise ValueError("Inconsistent attn_patch_cfg across tuned checkpoints.")
            if patch_cfg0:
                print(f"Using checkpoint attention mode: {patch_cfg0.get('attn_impl', 'softmax')}")
        # In subspace mode we still need patched attention keyspace in the base model/state dict
        # because lifted deltas target q_proj/k_proj/v_proj keys.
        base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
            task_meta=attn_meta_by_task[attn_meta_tasks[0]],
            base_patched_for_attn=base_patched_for_attn,
            clf=clf,
            base_ckpt=base_ckpt,
            strict_load=strict_load,
            base_sd=base_sd,
        )

    if peft_subspace != "full":
        if peft_cfg_map is None:
            raise ValueError(f"peft_subspace='{peft_subspace}' requires peft_config in checkpoints.")
        peft_cfg = get_peft_cfg(peft_cfg_map)
        subspace = get_subspace(peft_subspace)
        subspace_prepared = subspace.prepare(lora_by_task=peft_state_by_task, peft_cfg=peft_cfg)
        projected_by_task = subspace.project(subspace_prepared, lora_by_task=peft_state_by_task, peft_cfg=peft_cfg)
        if not projected_by_task:
            raise ValueError("Subspace projection returned empty projected_by_task.")
        tuned_sds_list = [projected_by_task[t] for t in tasks]
        base_sd_for_merge = {k: torch.zeros_like(v) for k, v in tuned_sds_list[0].items()}

        # Build full-space tuned checkpoints once for single-task baseline eval.
        for t in tasks:
            tuned_sd = materialize_peft_sd_from_adapter(
                peft_state=peft_state_by_task[t],
                base_sd=base_sd,
                build_cfg=build_cfg,
                peft_cfg=peft_cfg,
                strict_load=strict_load,
                patched_attn=attn_meta_by_task[t].patched_attn,
                attn_patch_cfg=attn_meta_by_task[t].attn_patch_cfg,
            )
            aligned = align_to_base_keys(tuned_sd, base_sd)
            if not aligned:
                raise ValueError(
                    f"No tensors from tuned checkpoint aligned to base keys for task '{t}'. "
                    "Check checkpoint key prefixes and model compatibility."
                )
            tuned_sds_by_task[t] = to_cpu_fp32(aligned)
    else:
        tuned_sds_list = [tuned_sds_by_task[t] for t in tasks]
        base_sd_for_merge = to_cpu_fp32(base_sd)

    if peft_subspace != "full":
        base_sd_for_merge = to_cpu_fp32(base_sd_for_merge)
    merge_base_sd = to_cpu_fp32(base_sd)

    needs_linear_attention = any(attn_meta_by_task[t].linearized_attn for t in tasks)
    assert_qkv_patched_before_linearizing(
        needs_linear_attention=needs_linear_attention,
        base_patched_for_attn=base_patched_for_attn,
        model_state_dict=clf.model.state_dict(),
    )
    if needs_linear_attention:
        print("Verified q/k/v attention patch is active before linearized attention evaluation.")

    requested_forward_mode = str(cfg.get("forward_mode", "auto"))
    if requested_forward_mode == "auto":
        all_ntk = bool(tasks) and all(strategy_by_task.get(t) == "ntk" for t in tasks)
        resolved_forward_mode = "linearized_ntk" if all_ntk else "standard"
    else:
        resolved_forward_mode = requested_forward_mode

    forward_mode = get_forward_mode(resolved_forward_mode)
    forward_mode.bind(
        clf=clf,
        base_sd=merge_base_sd,
        strict_load=strict_load,
    )
    print(f"Using forward mode: {resolved_forward_mode}")

    print("base keys:", len(base_sd_for_merge))
    print("example tuned aligned keys:", len(tuned_sds_list[0]))
    print("example intersection:", len(set(base_sd_for_merge).intersection(tuned_sds_list[0])))

    device = str(cfg.get("device", "cuda"))
    text_features_source = str(cfg.get("text_features_source", "auto")).strip().lower()
    if text_features_source not in {"auto", "zero_shot", "tuned_ckpt"}:
        raise ValueError("text_features_source must be one of: auto, zero_shot, tuned_ckpt")
    print(f"Text features source: {text_features_source}")
    use_humanized_classnames = not bool(cfg.get("no_humanize", True))
    classnames_mode = "humanized" if use_humanized_classnames else "raw"
    print(f"Classname mode: {classnames_mode}")

    # Pre-load datasets/loaders and compute single-task accuracies once (cached)
    single_cache_path = str(cfg.get("single_acc_cache", "src/.cache/single_task_acc.json"))
    single_cache = read_json_silent(single_cache_path)
    recompute_single = bool(cfg.get("recompute_single_acc", False))
    compute_zero_shot_acc = bool(cfg.get("single_acc_zero_shot", False))
    print("Single-accuracy baseline mode: single-task tuned (used for normalization)")
    if compute_zero_shot_acc:
        print("Zero-shot base-model accuracies will also be computed (not used for normalization).")

    per_task = []  # {task, loaders, classnames, build_cfg_task, single_acc, zero_shot_acc?}
    for task in tasks:
        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf.preprocess,
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
            model_name=build_cfg.model_name,
            pretrained=build_cfg.pretrained,
            device=build_cfg.device,
            dtype=build_cfg.dtype,
            prompt_templates=templates,
        )

        # Single-task baseline (used for normalization) always comes from the tuned checkpoint.
        task_text_features, task_text_features_mode = clf.resolve_eval_text_features(
            text_features_source=text_features_source,
            classnames=classnames,
            build_cfg=build_cfg_task,
            tuned_text_features=tuned_text_features_by_task.get(task, None),
            cache_dir="src/.cache/zs_cache",
            force_rebuild_zeroshot=False,
            task_name=task,
            ckpt_path=str(tuned_by_task[task]),
            verbose=True,
        )

        k = acc_cache_key(
            build_cfg.model_name,
            build_cfg.pretrained,
            task,
            chk_path=str(tuned_by_task[task]),
            baseline_mode="tuned",
            forward_mode=resolved_forward_mode,
            classnames_mode=classnames_mode,
            text_features_mode=task_text_features_mode,
        )
        single_acc: float | None = None
        if (not recompute_single) and k in single_cache:
            try:
                single_acc = float(single_cache[k]["top1"])
                print(f"{task}: [cache] single-task tuned acc: {single_acc:.6f}")
            except Exception:
                single_acc = None

        if single_acc is None:
            # Load tuned weights into the single model instance, evaluate, then overwrite later during merge eval.
            tuned_sd = tuned_sds_by_task[task]
            load_into_model(clf.model, tuned_sd, strict=strict_load)
            t0 = time.time()
            single_acc = eval_task_top1(
                clf=clf,
                loaders=loaders,
                classnames=classnames,
                build_cfg_task=build_cfg_task,
                device=device,
                split="test",
                text_features=task_text_features,
            )
            dt = time.time() - t0

            single_cache[k] = {
                "top1": float(single_acc),
                "model": build_cfg.model_name,
                "pretrained": build_cfg.pretrained,
                "dataset": task,
                "baseline_mode": "tuned",
                "baseline_checkpoint": str(tuned_by_task[task]),
                "ts": int(time.time()),
                "seconds": float(dt),
            }
            atomic_write_json(single_cache_path, single_cache)
            print(f"{task}: [computed] single-task tuned acc: {single_acc:.6f} (saved to {single_cache_path})")

        zero_shot_acc: float | None = None
        if compute_zero_shot_acc:
            k_zs = acc_cache_key(
                build_cfg.model_name,
                build_cfg.pretrained,
                task,
                chk_path=str(base_ckpt) if base_ckpt is not None else "open_clip_pretrained",
                baseline_mode="zero_shot",
                forward_mode=resolved_forward_mode,
                classnames_mode=classnames_mode,
                text_features_mode="zero_shot",
            )
            if (not recompute_single) and k_zs in single_cache:
                try:
                    zero_shot_acc = float(single_cache[k_zs]["top1"])
                    print(f"{task}: [cache] zero-shot acc: {zero_shot_acc:.6f}")
                except Exception:
                    zero_shot_acc = None

            if zero_shot_acc is None:
                load_into_model(clf.model, merge_base_sd, strict=strict_load)
                t0 = time.time()
                zero_shot_acc = eval_task_top1(
                    clf=clf,
                    loaders=loaders,
                    classnames=classnames,
                    build_cfg_task=build_cfg_task,
                    device=device,
                    split="test",
                    text_features=None,
                )
                dt = time.time() - t0
                single_cache[k_zs] = {
                    "top1": float(zero_shot_acc),
                    "model": build_cfg.model_name,
                    "pretrained": build_cfg.pretrained,
                    "dataset": task,
                    "baseline_mode": "zero_shot",
                    "baseline_checkpoint": str(base_ckpt) if base_ckpt is not None else "open_clip_pretrained",
                    "ts": int(time.time()),
                    "seconds": float(dt),
                }
                atomic_write_json(single_cache_path, single_cache)
                print(f"{task}: [computed] zero-shot acc: {zero_shot_acc:.6f} (saved to {single_cache_path})")

        per_task.append(
            {
                "task": task,
                "loaders": loaders,
                "classnames": classnames,
                "build_cfg_task": build_cfg_task,
                "text_features": task_text_features,
                "single_acc": float(single_acc),
                "zero_shot_acc": (float(zero_shot_acc) if zero_shot_acc is not None else None),
            }
        )

    print(
        f"Average single-task tuned acc across {len(per_task)} tasks: {sum(item['single_acc'] for item in per_task) / len(per_task):.6f}"
    )
    if compute_zero_shot_acc:
        zs_vals = [float(item["zero_shot_acc"]) for item in per_task if item.get("zero_shot_acc") is not None]
        if zs_vals:
            print(f"Average zero-shot acc across {len(zs_vals)} tasks: {sum(zs_vals) / len(zs_vals):.6f}")

    # Merge method
    method = get_method(str(cfg.get("method", "task_arithmetic")))
    prepared = None
    if isinstance(method, PreparedMergeMethod):
        print(f"\n🛠️ Preparing merge directions with method: {method.name}")
        prepared = method.prepare(
            base=base_sd_for_merge,
            tuned=tuned_sds_list,
            weights=merge_weights,
            strict=strict_load,
            tasks=tasks,
            method_params=method_params,
        )
    if prepared is not None:
        print("Prepared merge directions will be reused across all alpha evaluations.")

    # Alpha search results (normalized vs single-task baseline)
    alpha_results: dict[float, list[float]] = {}  # alpha -> list of accs over tasks
    alpha_results_norm: dict[float, list[float]] = {}  # alpha -> list of norm accs over tasks
    best_alpha_per_task: dict[str, float] = {}
    best_norm_per_task: dict[str, float] = {}
    max_avg_norm = 0.0

    for alpha in alphas:
        print(f"\n=== Method = {method.name} - Space = {peft_subspace} - Alpha = {alpha:.3f} ===")
        merged_sd = build_merged_state_for_alpha(
            method=method,
            prepared=prepared,
            base_sd_for_merge=base_sd_for_merge,
            tuned_sds_list=tuned_sds_list,
            weights=merge_weights,
            method_params=method_params,
            alpha=float(alpha),
            peft_subspace=peft_subspace,
            subspace=subspace,
            subspace_prepared=subspace_prepared,
            peft_cfg=peft_cfg,
            peft_state_by_task=peft_state_by_task,
            tasks=tasks,
            merge_base_sd=merge_base_sd,
        )

        miss, unexp = load_into_model(clf.model, merged_sd, strict=strict_load)
        print(f"Loaded merged weights (alpha={alpha:.3f}). missing={miss}, unexpected={unexp}")

        if (not alpha_search) and cfg.get("save_merged", None) is not None:
            outp = Path(str(cfg["save_merged"]))
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(merged_sd, str(outp))
            print(f"Saved merged state_dict to {outp}")

        del merged_sd

        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()

        accs, norm_accs = eval_norm_accs_for_split(
            clf=clf,
            per_task=per_task,
            device=device,
            split="val",
            print_per_task=True,
        )
        alpha_results[float(alpha)] = accs
        alpha_results_norm[float(alpha)] = [float(norm) for norm in norm_accs]
        for idx, item in enumerate(per_task):
            task = str(item["task"])
            norm = float(norm_accs[idx])
            if (task not in best_norm_per_task) or (norm > best_norm_per_task[task]):
                best_norm_per_task[task] = norm
                best_alpha_per_task[task] = float(alpha)

        avg_norm = sum(alpha_results_norm[float(alpha)]) / len(tasks)
        avg_abs = sum(alpha_results[float(alpha)]) / len(tasks)
        print(f"alpha={alpha:.3f}  avg_abs={avg_abs:.6f} avg_norm={avg_norm:.6f}")

        if avg_norm > max_avg_norm:
            max_avg_norm = avg_norm
        else:
            print("Avg norm did not improve, stopping alpha search early.")
            break

    # Summary
    avg_norm_per_alpha = {a: sum(v) / len(v) for a, v in alpha_results_norm.items()}
    best_alpha = max(avg_norm_per_alpha, key=lambda a: avg_norm_per_alpha[a])

    print("\n=== Alpha search summary ===")
    for a in sorted(avg_norm_per_alpha):
        print(
            f"alpha={a:.3f}  avg_abs={sum(alpha_results_norm[float(a)]) / len(tasks):.6f} avg_norm={avg_norm_per_alpha[a]:.6f}"
        )
    print(
        f"\nBest alpha: {best_alpha:.3f} -> avg_abs={sum(alpha_results_norm[float(best_alpha)]) / len(tasks):.6f} avg_norm={avg_norm_per_alpha[best_alpha]:.6f}"
    )

    print("\nBest alpha per task:")
    for t in tasks:
        if t in best_alpha_per_task:
            print(
                f"{t}: alpha={best_alpha_per_task[t]:.3f} avg_abs={sum(alpha_results_norm[float(best_alpha_per_task[t])]) / len(tasks):.6f} avg_norm={best_norm_per_task[t]:.6f}"
            )

    # Re-run best alpha once to report avg_top1 / avg_norm
    print(f"\n(Re-running best alpha ({best_alpha:.3f}) once to report avg_top1)")
    alpha = float(best_alpha)
    merged_sd = build_merged_state_for_alpha(
        method=method,
        prepared=prepared,
        base_sd_for_merge=base_sd_for_merge,
        tuned_sds_list=tuned_sds_list,
        weights=merge_weights,
        method_params=method_params,
        alpha=alpha,
        peft_subspace=peft_subspace,
        subspace=subspace,
        subspace_prepared=subspace_prepared,
        peft_cfg=peft_cfg,
        peft_state_by_task=peft_state_by_task,
        tasks=tasks,
        merge_base_sd=merge_base_sd,
    )
    load_into_model(clf.model, merged_sd, strict=strict_load)
    del merged_sd

    merged_accs, norm_accs = eval_norm_accs_for_split(
        clf=clf,
        per_task=per_task,
        device=device,
        split="test",
        print_per_task=False,
    )

    pretty_print_task_accuracies(
        suite_name,
        method.name,
        peft_subspace,
        per_task,
        merged_accs,
        norm_accs,
        single_accs=[item["single_acc"] for item in per_task],
    )

    print_latex_task_rows(per_task, merged_accs, norm_accs)


if __name__ == "__main__":
    main()
