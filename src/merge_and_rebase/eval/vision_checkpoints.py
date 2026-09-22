"""Evaluate one fine-tuned vision checkpoint per task with the shared top-1 evaluator.

The entrypoint intentionally performs no merge or rebase.  Each checkpoint is
loaded on top of the same pretrained/base model, evaluated on its matching
dataset through :func:`eval_task_top1`, and then discarded before the next one.

Checkpoint paths can be supplied in a JSON config as ``tuned_ckpts`` or from
the command line, for example::

    python -m merge_and_rebase.eval.vision_checkpoints \
      --suite vision8 --tasks all \
      --clip-model ViT-B-16 --clip-pretrained datacomp_xl_s13b_b90k \
      --tuned-ckpts Cars=/path/cars.pt DTD=/path/dtd.pt ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_suite_arg,
    add_tasks_arg,
    merge_non_none,
)
from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.eval.utils import (
    eval_task_top1,
    extract_checkpoint_attn_patch_info,
    extract_peft_components,
    get_peft_cfg,
    humanize,
    is_peft_checkpoint,
    load_vision_checkpoint_reference,
    materialize_peft_sd_from_adapter,
    maybe_patch_base_for_task_attn,
    to_cpu_fp32,
)
from merge_and_rebase.io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from merge_and_rebase.io.peft_helpers import (
    is_peft_adapter_dir_ckpt,
    load_peft_adapter_dir_components,
    normalize_peft_adapter_dir_checkpoint,
)
from merge_and_rebase.io.utils import atomic_write_json
from merge_and_rebase.models.forward_modes import get_forward_mode, list_forward_modes, normalize_forward_mode_params
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.utils.helpers import load_json, parse_csv


def _parse_tasks(raw: Any, *, suite_tasks: list[str]) -> list[str]:
    if raw is None or str(raw).strip().lower() == "all":
        return list(suite_tasks)
    tasks = parse_csv(str(raw))
    unknown = [task for task in tasks if task not in suite_tasks]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Allowed: {suite_tasks}")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Duplicate tasks are not allowed: {tasks}")
    return tasks


def _parse_named_checkpoints(values: list[str], *, arg_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{arg_name} entries must use TASK=PATH. Got: {value!r}")
        task, path = value.split("=", 1)
        task = task.strip()
        path = path.strip()
        if not task or not path:
            raise ValueError(f"{arg_name} entries must use non-empty TASK=PATH values. Got: {value!r}")
        if task in parsed:
            raise ValueError(f"Duplicate checkpoint for task '{task}' in {arg_name}.")
        parsed[task] = path
    return parsed


def _resolve_checkpoint_map(
    *,
    config_value: Any,
    cli_values: list[str] | None,
    checkpoint_overrides: list[str] | None,
    tasks: list[str],
) -> dict[str, str]:
    if config_value is None:
        checkpoint_map: dict[str, str] = {}
    elif isinstance(config_value, dict):
        checkpoint_map = {str(task): str(path) for task, path in config_value.items()}
    elif isinstance(config_value, list):
        if len(config_value) != len(tasks):
            raise ValueError(
                "A list-valued config 'tuned_ckpts' must have the same length as the selected tasks "
                f"({len(config_value)} != {len(tasks)})."
            )
        checkpoint_map = {task: str(path) for task, path in zip(tasks, config_value, strict=True)}
    else:
        raise TypeError("config['tuned_ckpts'] must be a TASK->PATH object or an ordered list of paths.")

    if cli_values:
        if all("=" in value for value in cli_values):
            checkpoint_map = _parse_named_checkpoints(cli_values, arg_name="--tuned-ckpts")
            unknown = [task for task in checkpoint_map if task not in tasks]
            if unknown:
                raise ValueError(f"--tuned-ckpts contains tasks not selected by --tasks/config: {unknown}")
        elif any("=" in value for value in cli_values):
            raise ValueError("Do not mix TASK=PATH and ordered PATH entries in --tuned-ckpts.")
        else:
            if len(cli_values) != len(tasks):
                raise ValueError(
                    "Ordered --tuned-ckpts must have the same length as --tasks "
                    f"({len(cli_values)} != {len(tasks)}). Prefer TASK=PATH entries."
                )
            checkpoint_map = {task: path for task, path in zip(tasks, cli_values, strict=True)}

    if checkpoint_overrides:
        overrides = _parse_named_checkpoints(checkpoint_overrides, arg_name="--checkpoint")
        unknown = [task for task in overrides if task not in tasks]
        if unknown:
            raise ValueError(f"--checkpoint contains tasks not selected by --tasks/config: {unknown}")
        checkpoint_map.update(overrides)

    missing = [task for task in tasks if task not in checkpoint_map]
    if missing:
        raise ValueError(f"Missing fine-tuned checkpoints for tasks: {missing}")
    return {task: checkpoint_map[task] for task in tasks}


def _format_results_table(rows: list[dict[str, Any]], *, split: str) -> str:
    task_width = max([len("task"), *(len(str(row["task"])) for row in rows)])
    checkpoint_width = max([len("checkpoint"), *(len(Path(str(row["checkpoint"])).name) for row in rows)])
    header = (
        f" {'task':<{task_width}}  {'checkpoint':<{checkpoint_width}}  "
        f"{f'{split}_top1':>12}  {'accuracy_%':>12}  {'seconds':>10}"
    )
    separator = (
        f" {'-' * task_width}  {'-' * checkpoint_width}  "
        f"{'-' * 12}  {'-' * 12}  {'-' * 10}"
    )
    lines = [f"\nFine-tuned checkpoint evaluation ({split})", header, separator]
    for row in rows:
        lines.append(
            f" {str(row['task']):<{task_width}}  {Path(str(row['checkpoint'])).name:<{checkpoint_width}}  "
            f"{float(row['top1']):>12.6f}  {100.0 * float(row['top1']):>12.2f}  "
            f"{float(row['seconds']):>10.1f}"
        )
    avg = sum(float(row["top1"]) for row in rows) / max(1, len(rows))
    total_seconds = sum(float(row["seconds"]) for row in rows)
    lines.extend(
        [
            separator,
            f" {'avg':<{task_width}}  {'-':<{checkpoint_width}}  {avg:>12.6f}  "
            f"{100.0 * avg:>12.2f}  {total_seconds:>10.1f}",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate independently fine-tuned OpenCLIP checkpoints with eval_task_top1"
    )
    add_config_arg(parser)
    add_suite_arg(parser, choices=sorted(SUITES))
    add_tasks_arg(parser, help_text="Comma-separated task names, or 'all'.")
    parser.add_argument("--backbone-name", choices=["openclip", "openai_clip"], default=None)
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--clip-pretrained", type=str, default=None)
    parser.add_argument("--base-ckpt", type=str, default=None)
    add_device_dtype_args(parser, device_default=None, dtype_default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=None)
    classname_group = parser.add_mutually_exclusive_group()
    classname_group.add_argument("--no-humanize", dest="no_humanize", action="store_true", default=None)
    classname_group.add_argument("--humanize", dest="no_humanize", action="store_false")
    parser.add_argument(
        "--text-features-source",
        choices=["auto", "zero_shot", "tuned_ckpt"],
        default=None,
        help="Use checkpoint-provided tuned text features or the standard zero-shot head.",
    )
    parser.add_argument(
        "--forward-mode",
        choices=["auto", *list_forward_modes()],
        default=None,
        help="'auto' reads forward_mode independently from every checkpoint.",
    )
    parser.add_argument(
        "--tuned-ckpts",
        nargs="+",
        default=None,
        metavar="TASK=PATH",
        help="TASK=PATH entries (recommended), or paths ordered exactly like --tasks.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="TASK=PATH",
        help="Override one config checkpoint; repeat once per task when needed.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg: dict[str, Any] = load_json(args.config) if args.config is not None else {}
    cfg = merge_non_none(
        cfg,
        {
            "suite": args.suite,
            "tasks": args.tasks,
            "backbone_name": args.backbone_name,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "base_ckpt": args.base_ckpt,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "split": args.split,
            "strict_load": args.strict_load,
            "no_humanize": args.no_humanize,
            "text_features_source": args.text_features_source,
            "forward_mode": args.forward_mode,
            "output_json": args.output_json,
        },
    )

    suite_name = str(cfg.get("suite", "vision8"))
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
    suite = SUITES[suite_name]
    tasks = _parse_tasks(cfg.get("tasks", "all"), suite_tasks=list(suite.tasks))
    checkpoint_map = _resolve_checkpoint_map(
        config_value=cfg.get("tuned_ckpts"),
        cli_values=args.tuned_ckpts,
        checkpoint_overrides=args.checkpoint,
        tasks=tasks,
    )

    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: requested device '{device}' is unavailable; falling back to CPU.")
        device = "cpu"
    strict_load = bool(cfg.get("strict_load", False))
    split = str(cfg.get("split", "test"))
    text_features_source = str(cfg.get("text_features_source", "auto"))
    requested_forward_mode = str(cfg.get("forward_mode", "auto"))
    use_humanized_classnames = not bool(cfg.get("no_humanize", True))

    build_cfg = OpenClipBuildConfig(
        loader=str(cfg.get("backbone_name", "openclip")),
        model_name=str(cfg.get("clip_model", "ViT-B-32")),
        pretrained=str(cfg.get("clip_pretrained", "openai")),
        device=device,
        dtype=cfg.get("dtype"),
    )
    print(f"Model: {build_cfg.loader} {build_cfg.model_name} / {build_cfg.pretrained}")
    print(
        f"Suite: {suite_name} | tasks: {', '.join(tasks)} | split: {split} | "
        f"text features: {text_features_source} | forward mode: {requested_forward_mode}"
    )
    clf = OpenClipClassifier.build(build_cfg)

    base_ckpt = cfg.get("base_ckpt")
    if base_ckpt is not None:
        raw_base = load_ckpt(str(base_ckpt))
        initial_model_sd = clf.model.state_dict()
        aligned_base = align_to_base_keys(raw_base, initial_model_sd)
        if not aligned_base:
            raise ValueError(f"No tensors from base checkpoint aligned to model keys: {base_ckpt}")
        load_into_model(clf.model, aligned_base, strict=strict_load)

    base_sd = to_cpu_fp32({key: value for key, value in clf.model.state_dict().items()})
    base_patched_for_attn = False
    expected_attn_meta: tuple[bool, dict[str, Any] | None] | None = None
    rows: list[dict[str, Any]] = []

    for task in tasks:
        checkpoint_ref = checkpoint_map[task]
        checkpoint_path, checkpoint_obj = load_vision_checkpoint_reference(ckpt_ref=checkpoint_ref)
        checkpoint_obj = normalize_peft_adapter_dir_checkpoint(
            checkpoint_obj,
            checkpoint_path=checkpoint_path,
        )
        attn_meta = extract_checkpoint_attn_patch_info(obj=checkpoint_obj, ckpt_path=checkpoint_path)
        current_attn_meta = (bool(attn_meta.patched_attn), attn_meta.attn_patch_cfg)
        if expected_attn_meta is None:
            expected_attn_meta = current_attn_meta
            base_sd, base_patched_for_attn = maybe_patch_base_for_task_attn(
                task_meta=attn_meta,
                base_patched_for_attn=base_patched_for_attn,
                clf=clf,
                base_ckpt=(str(base_ckpt) if base_ckpt is not None else None),
                strict_load=strict_load,
                base_sd=base_sd,
            )
            base_sd = to_cpu_fp32(base_sd)
        elif current_attn_meta != expected_attn_meta:
            raise ValueError(
                "All checkpoints in one evaluation must use the same attention representation. "
                f"Task '{task}' has {current_attn_meta}, expected {expected_attn_meta}."
            )

        tuned_text_features = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(
            obj=checkpoint_obj,
            ckpt_path=checkpoint_path,
        )
        dense_state = (
            dict(checkpoint_obj.get("peft_dense_state", {}))
            if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get("peft_dense_state", {}), dict)
            else {}
        )
        if is_peft_adapter_dir_ckpt(checkpoint_obj):
            peft_state, peft_cfg_map = load_peft_adapter_dir_components(
                checkpoint_obj["peft_adapter_dir"],
                checkpoint_path=checkpoint_path,
            )
            tuned_sd = materialize_peft_sd_from_adapter(
                peft_state=peft_state,
                base_sd=base_sd,
                build_cfg=build_cfg,
                peft_cfg=get_peft_cfg(peft_cfg_map),
                peft_dense_state=dense_state,
                strict_load=strict_load,
                patched_attn=attn_meta.patched_attn,
                attn_patch_cfg=attn_meta.attn_patch_cfg,
            )
        elif is_peft_checkpoint(checkpoint_obj):
            peft_state, peft_cfg_map = extract_peft_components(checkpoint_obj)
            tuned_sd = materialize_peft_sd_from_adapter(
                peft_state=peft_state,
                base_sd=base_sd,
                build_cfg=build_cfg,
                peft_cfg=get_peft_cfg(peft_cfg_map),
                peft_dense_state=dense_state,
                strict_load=strict_load,
                patched_attn=attn_meta.patched_attn,
                attn_patch_cfg=attn_meta.attn_patch_cfg,
            )
        else:
            tuned_sd = load_ckpt(checkpoint_path)

        aligned = align_to_base_keys(tuned_sd, base_sd)
        if not aligned:
            raise ValueError(
                f"No tensors from checkpoint for task '{task}' aligned to the configured model: {checkpoint_path}"
            )
        load_into_model(clf.model, base_sd, strict=strict_load)
        missing, unexpected = load_into_model(clf.model, aligned, strict=strict_load)

        checkpoint_forward_mode = (
            str(checkpoint_obj.get("forward_mode"))
            if isinstance(checkpoint_obj, dict) and checkpoint_obj.get("forward_mode") is not None
            else "standard"
        )
        forward_mode_name = checkpoint_forward_mode if requested_forward_mode == "auto" else requested_forward_mode
        forward_mode_params = normalize_forward_mode_params(
            forward_mode_name,
            checkpoint_obj.get("forward_mode_params") if isinstance(checkpoint_obj, dict) else None,
        )
        get_forward_mode(forward_mode_name).bind(
            clf=clf,
            base_sd=base_sd,
            strict_load=strict_load,
            params=forward_mode_params,
        )

        hf_path, hf_config, split_map = suite.resolver(task)
        hf_ds = load_hf_splits(
            hf_path,
            config=hf_config,
            requested_splits=tuple(dict.fromkeys(split_map.values())),
        )
        loaders = build_vision_loaders(
            hf_ds=hf_ds,
            hf_path=hf_path,
            preprocess=clf.preprocess,
            train_preprocess=None,
            ft_epochs=1,
            split_map=split_map,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 6)),
            pin_memory=True,
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            seed=int(cfg.get("seed", 0)),
        )
        classnames = list(loaders.classnames)
        if use_humanized_classnames:
            classnames = [humanize(name) for name in classnames]
        templates = get_templates(task)
        if not templates:
            raise ValueError(f"get_templates('{task}') returned no prompt templates.")
        task_build_cfg = OpenClipBuildConfig(
            loader=build_cfg.loader,
            model_name=build_cfg.model_name,
            pretrained=build_cfg.pretrained,
            device=build_cfg.device,
            dtype=build_cfg.dtype,
            prompt_templates=templates,
        )
        text_features, resolved_text_features_source = clf.resolve_eval_text_features(
            text_features_source=text_features_source,
            classnames=classnames,
            build_cfg=task_build_cfg,
            tuned_text_features=tuned_text_features,
            cache_dir="src/.cache/zs_cache",
            force_rebuild_zeroshot=False,
            task_name=task,
            ckpt_path=checkpoint_path,
            verbose=True,
        )

        print(
            f"\nEvaluating {task}: {checkpoint_path} "
            f"(aligned={len(aligned)}, missing={missing}, unexpected={unexpected}, "
            f"forward={forward_mode_name}, head={resolved_text_features_source})"
        )
        started = time.perf_counter()
        accuracy = eval_task_top1(
            clf=clf,
            loaders=loaders,
            classnames=classnames,
            build_cfg_task=task_build_cfg,
            device=device,
            split=split,
            text_features=text_features,
        )
        elapsed = time.perf_counter() - started
        print(f"{task}: {split}_top1={accuracy:.6f} ({100.0 * accuracy:.2f}%)")
        rows.append(
            {
                "task": task,
                "checkpoint": checkpoint_path,
                "top1": float(accuracy),
                "accuracy_percent": float(100.0 * accuracy),
                "seconds": float(elapsed),
                "forward_mode": forward_mode_name,
                "text_features_source": resolved_text_features_source,
                "aligned_keys": len(aligned),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            }
        )
        # Linearized NTK evaluation attaches a frozen reference copy to the
        # classifier. Restore the ordinary forward and release that copy before
        # loading the next task, keeping peak memory independent of task count.
        get_forward_mode("standard").bind(clf=clf, base_sd=base_sd, strict_load=False)
        for attribute in ("_linearized_module", "_linearized_visual_ref"):
            if hasattr(clf, attribute):
                delattr(clf, attribute)
        del tuned_sd, aligned, checkpoint_obj, loaders, hf_ds
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(_format_results_table(rows, split=split))
    avg_accuracy = sum(float(row["top1"]) for row in rows) / max(1, len(rows))
    print("\nLaTeX rows (task order, then avg):")
    print("tasks : " + " & ".join([*(str(row["task"]) for row in rows), "avg"]) + " \\\\")
    print(
        "top1 : "
        + " & ".join([*(f"{100.0 * float(row['top1']):.2f}" for row in rows), f"{100.0 * avg_accuracy:.2f}"])
        + " \\\\"
    )

    output_json = cfg.get("output_json")
    if output_json is not None:
        payload = {
            "suite": suite_name,
            "split": split,
            "model": {
                "loader": build_cfg.loader,
                "clip_model": build_cfg.model_name,
                "clip_pretrained": build_cfg.pretrained,
                "base_ckpt": base_ckpt,
            },
            "per_task": rows,
            "avg_top1": float(avg_accuracy),
            "avg_accuracy_percent": float(100.0 * avg_accuracy),
        }
        atomic_write_json(str(output_json), payload)
        print(f"\nSaved evaluation summary to {output_json}")


if __name__ == "__main__":
    main()
