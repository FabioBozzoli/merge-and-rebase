"""Few-shot knowledge distillation, as a baseline for the ``steer`` / ``steer_text`` rebasin methods.

Same transfer as the rebase entrypoints: a fine-tuned model A (``tuned_ckpts``) is the *teacher* and the
pretrained model B is the *student*. steer corrects B's features with a map fitted on the few-shot support
set; here B is instead fine-tuned (all parameters) to match A's class logits on that same support set.

Everything that makes the comparison fair is shared with the steer entrypoints and not re-implemented:

* support set    -- ``steer._few_shot`` (same ``few_shot`` per class, same ``seed``), including the
                    50-per-class train subsample steer applies to SVHN/MNIST;
* val/test carve -- ``vision_rebase._VAL_TEST_SPLIT_SEED`` / ``text_rebase._build_task_splits``;
* baseline       -- B untouched, scored exactly as ``target_zeroshot`` (vision) / the nearest-mean head (text);
* summary        -- the ``test_results.per_task_*`` schema of ``vision_rebase`` / ``text_rebase``.

Only class logits are distilled, never features: the two models' embedding widths differ (ViT-B/16 512 vs
ViT-L/14 768; T5-base 768 vs T5-large 1024) and the tokenizers differ too (T5 vs RoBERTa), but both sides
always expose the same C class logits.

Usage::

    python -m merge_and_rebase.eval.kd_rebase vision --config configs/kd/vision8_kd_vitb_vitb.json --tasks SVHN
    python -m merge_and_rebase.eval.kd_rebase text   --config configs/kd/text_kd_t5base_t5large.json

Config = the matching steer config (same model/ckpt/data keys) plus an optional ``kd`` block::

    "kd": {"few_shot": 10, "seed": 33, "epochs": 20, "lr": 1e-5, "weight_decay": 0.0, "temperature": 1.0,
           "ce_weight": 0.0, "batch_size": 32, "grad_clip": 1.0, "amp": null, "train_mode": "eval",
           "epoch_selection": "last"}

``few_shot`` / ``seed`` fall back to ``method_params.few_shot`` / ``method_params.seed`` of a steer config.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_suite_arg,
    add_tasks_arg,
    build_logging_overrides,
    merge_non_none,
)
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .rebase_metrics import normalized_accuracy_ratio

_KD_DEFAULTS: dict[str, Any] = {
    "epochs": 20,
    "lr": 1.0e-5,
    "weight_decay": 0.0,
    "temperature": 1.0,
    "ce_weight": 0.0,
    "batch_size": 32,
    "grad_clip": 1.0,
    "amp": None,  # None (model dtype) | "bf16"
    "train_mode": "eval",  # "eval" | "train" (dropout on)
    # "last": always evaluate the model after the final epoch (fixed epoch, no selection on val);
    # "val": keep the epoch with the highest val accuracy (epoch 0 = untouched student is a candidate).
    "epoch_selection": "last",
}


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


def resolve_kd_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """``cfg["kd"]`` over defaults; ``few_shot`` / ``seed`` fall back to the steer ``method_params``."""
    method_params = cfg.get("method_params") or {}
    kd = {**_KD_DEFAULTS, **(cfg.get("kd") or {})}
    if kd.get("few_shot") is None:
        kd["few_shot"] = cfg.get("few_shot", method_params.get("few_shot"))
    if kd.get("few_shot") is None:
        raise ValueError("Set kd.few_shot (or method_params.few_shot of a steer config): shots per class.")
    kd["few_shot"] = int(kd["few_shot"])
    if kd.get("seed") is None:
        kd["seed"] = cfg.get("seed", method_params.get("seed", 42))
    kd["seed"] = int(kd["seed"])
    for key in ("epochs", "batch_size"):
        kd[key] = int(kd[key])
        if kd[key] <= 0:
            raise ValueError(f"kd.{key} must be > 0.")
    for key in ("lr", "weight_decay", "temperature", "ce_weight", "grad_clip"):
        kd[key] = float(kd[key])
    if kd["temperature"] <= 0.0:
        raise ValueError("kd.temperature must be > 0.")
    if kd["amp"] not in (None, "bf16"):
        raise ValueError("kd.amp must be null or 'bf16'.")
    if kd["epoch_selection"] not in {"last", "val"}:
        raise ValueError("kd.epoch_selection must be 'last' or 'val'.")
    if kd["train_mode"] not in {"eval", "train"}:
        raise ValueError("kd.train_mode must be 'eval' or 'train'.")
    return kd


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    labels: torch.Tensor | None = None,
    ce_weight: float = 0.0,
) -> torch.Tensor:
    """``T^2 * KL(softmax(teacher/T) || softmax(student/T))`` (+ ``ce_weight * CE(student, labels)``).

    Same distillation term as ``finetune/regularizers/_distill_runtime.compute_distillation_loss``
    (``kl_div``), restated on plain tensors so this module does not depend on the regularizer stack.
    ``labels`` index the logit columns, i.e. they must already be in the same class space.
    """
    t = float(temperature)
    loss = F.kl_div(
        F.log_softmax(student_logits.float() / t, dim=-1),
        F.softmax(teacher_logits.float() / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)
    if ce_weight > 0.0:
        if labels is None:
            raise ValueError("ce_weight > 0 needs labels.")
        loss = loss + float(ce_weight) * F.cross_entropy(student_logits.float(), labels)
    return loss


def _support_batches(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def _norm_acc(result_acc: float, baseline_acc: float) -> float:
    return normalized_accuracy_ratio(result_acc, baseline_acc)


def _mean(values: list[float]) -> float:
    defined = [float(v) for v in values if float(v) == float(v)]
    return sum(defined) / len(defined) if defined else float("nan")


def _build_summary(
    *,
    cfg: dict[str, Any],
    kd: dict[str, Any],
    suite_name: str,
    tasks: list[str],
    baseline_accs: list[float],
    kd_accs: list[float],
    norm_accs: list[float],
    task_diagnostics: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "tasks": tasks,
        "method": "kd",
        "method_label": f"kd(few_shot={kd['few_shot']}, epochs={kd['epochs']}, lr={kd['lr']:g}, T={kd['temperature']:g})",
        **extra,
        "baseline_label": "target_zeroshot",
        "metric_definitions": {
            "absolute_accuracy": "top-1 accuracy in [0, 1] of the distilled student (epoch picked on the val split)",
            "baseline_accuracy": "top-1 accuracy of the untouched target model (zero-shot / nearest-mean head)",
            "normalized_accuracy_ratio": "absolute_accuracy / baseline_accuracy",
            "normalized_accuracy_ratio_display": (
                "ratio (decimal, not a percentage); values above 1.0 indicate the distilled student exceeds "
                "the untouched target; report as a decimal ratio, never multiplied by 100"
            ),
        },
        "test_results": {
            "per_task_baseline_accuracy": dict(zip(tasks, map(float, baseline_accs), strict=True)),
            "per_task_absolute_accuracy": dict(zip(tasks, map(float, kd_accs), strict=True)),
            "per_task_normalized_accuracy_ratio": dict(zip(tasks, map(float, norm_accs), strict=True)),
            "per_task_baseline": dict(zip(tasks, map(float, baseline_accs), strict=True)),
            "per_task_rebased": dict(zip(tasks, map(float, kd_accs), strict=True)),
            "per_task_norm": dict(zip(tasks, map(float, norm_accs), strict=True)),
            "avg_rebased": _mean(kd_accs),
            "avg_norm": _mean(norm_accs),
        },
        "kd_config": kd,
        "kd_diagnostics": task_diagnostics,
    }


def _common_parser(name: str, help_text: str, sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=help_text)
    add_config_arg(p)
    add_suite_arg(p)
    add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")
    add_device_dtype_args(p, device_default=None, dtype_default=None)
    add_logging_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Eval batch size (training uses kd.batch_size).")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--tuned-ckpts", type=str, default=None)
    p.add_argument("--kd", type=str, default=None, help="JSON object overriding the config's 'kd' block.")
    return p


def _load_cfg(args: argparse.Namespace, cli: dict[str, Any]) -> dict[str, Any]:
    import json

    cfg: dict[str, Any] = load_json(args.config) if args.config is not None else {}
    if args.kd is not None:
        cfg["kd"] = {**(cfg.get("kd") or {}), **json.loads(args.kd)}
    if args.tuned_ckpts is not None:
        cli["tuned_ckpts"] = json.loads(args.tuned_ckpts)
    cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
    cfg["logging"] = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
    return cfg


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------


def _vision_support_indices(train_dataset: Any, task: str, few_shot: int, seed: int) -> list[int]:
    """Indices into the full train dataset of exactly steer's support set.

    steer first keeps ``_TRAIN_FEATURES_PER_CLASS`` examples per class (SVHN/MNIST, ``seed=0``) and
    draws ``few_shot`` per class *from that subset* (``seed``); the other tasks draw from the full split.
    """
    from ..rebase.methods import steer as steer_mod

    labels = steer_mod._dataset_labels(train_dataset)
    pool = torch.arange(len(labels))
    per_class = steer_mod._TRAIN_FEATURES_PER_CLASS.get(task)
    if per_class is not None:
        keep = steer_mod._few_shot(labels, per_class, seed=0)
        pool, labels = pool[keep], labels[keep]
    return pool[steer_mod._few_shot(labels, few_shot, seed)].tolist()


def _load_finetuned_source_vision(clf_source: Any, ckpt_path: str, task: str) -> Any:
    """A copy of ``clf_source`` carrying the task's fine-tuned weights (same checks as vision_rebase's steer path)."""
    from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
    from ..rebase.methods.steer import is_linearized_checkpoint, reconstruct_linearized_checkpoint

    teacher = deepcopy(clf_source)
    model_sd = teacher.model.state_dict()
    raw = load_ckpt(ckpt_path)
    aligned = align_to_base_keys(raw, model_sd)
    if not aligned and is_linearized_checkpoint(raw):
        aligned = reconstruct_linearized_checkpoint(raw, teacher.model)
    if not aligned:
        raise ValueError(f"No tensors from tuned checkpoint aligned to source model keys for '{task}': {ckpt_path}.")
    changed = sum(
        1
        for k, v in aligned.items()
        if not torch.equal(v.detach().cpu().to(model_sd[k].dtype), model_sd[k].detach().cpu())
    )
    if changed == 0:
        raise ValueError(f"Tuned checkpoint for '{task}' is identical to the source base model: {ckpt_path}.")
    load_into_model(teacher.model, aligned, strict=False)
    print(f"  {task}: teacher loaded ({len(aligned)} keys, {changed} differ from base)")
    return teacher


@torch.no_grad()
def _vision_logits(clf: Any, images: torch.Tensor, dev: torch.device) -> torch.Tensor:
    feats = F.normalize(clf.model.encode_image(images.to(dev)).float(), dim=-1)
    return float(clf.logit_scale) * feats @ clf._zs_text_features.float().to(dev).t()


def run_vision(args: argparse.Namespace) -> None:
    run_logger = None
    try:
        from ..data.templates import get_templates
        from ..data.vision_loaders import build_vision_loaders, load_hf_splits
        from ..io.ckpt import resolve_ckpt_path
        from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
        from .datasets.vision8_14_20 import SUITES
        from .print_utils import pretty_print_task_accuracies
        from .utils import eval_task_top1, humanize
        from .vision_rebase import _VAL_TEST_SPLIT_SEED

        cli = {
            "source_clip_model": args.source_clip_model,
            "source_clip_pretrained": args.source_clip_pretrained,
            "target_clip_model": args.target_clip_model,
            "target_clip_pretrained": args.target_clip_pretrained,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "suite": args.suite,
            "tasks": args.tasks,
            "device": args.device,
            "dtype": args.dtype,
        }
        cfg = _load_cfg(args, cli)
        kd = resolve_kd_config(cfg)
        device = str(cfg.get("device", "cuda"))
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        suite_name = cfg.get("suite", "vision8")
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
        suite = SUITES[suite_name]
        tasks_arg = cfg.get("tasks", "all")
        tasks = list(suite.tasks) if tasks_arg == "all" else parse_csv(tasks_arg)
        bad = [t for t in tasks if t not in suite.tasks]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")

        summary_path = default_summary_path(
            entrypoint="eval.kd_rebase", logging_cfg=cfg["logging"], default_parent=None
        )
        run_logger = start_run(
            entrypoint="eval.kd_rebase",
            logging_cfg=cfg["logging"],
            summary_path=summary_path,
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": suite_name,
                "tasks": tasks,
                "summary_path": str(summary_path),
            },
        )

        tuned_by_task = {t: resolve_ckpt_path(str(p)) for t, p in (cfg.get("tuned_ckpts") or {}).items()}
        missing = [t for t in tasks if t not in tuned_by_task]
        if missing:
            raise ValueError(f"tuned_ckpts is missing tasks: {missing}.")

        source_cfg = OpenClipBuildConfig(
            model_name=cfg.get("source_clip_model", "ViT-B-16"),
            pretrained=cfg.get("source_clip_pretrained", "datacomp_xl_s13b_b90k"),
            device=device,
            dtype=cfg.get("dtype", None),
        )
        target_cfg = OpenClipBuildConfig(
            model_name=cfg.get("target_clip_model", "ViT-B-16"),
            pretrained=cfg.get("target_clip_pretrained", "laion2b_s34b_b88k"),
            device=device,
            dtype=cfg.get("dtype", None),
        )
        print(f"Teacher (A): {source_cfg.model_name} / {source_cfg.pretrained}")
        print(f"Student (B): {target_cfg.model_name} / {target_cfg.pretrained}")
        clf_source = OpenClipClassifier.build(source_cfg)
        clf_target = OpenClipClassifier.build(target_cfg)
        student_init = {k: v.detach().cpu().clone() for k, v in clf_target.model.state_dict().items()}

        use_humanized = not bool(cfg.get("no_humanize", True))
        eval_bs = int(cfg.get("batch_size", 128))
        n_workers = int(cfg.get("num_workers", 6))
        val_fraction = float(cfg.get("val_fraction", 0.1))
        generator = torch.Generator().manual_seed(kd["seed"])

        per_task: list[dict[str, Any]] = []
        baseline_accs: list[float] = []
        kd_accs: list[float] = []
        diagnostics: dict[str, Any] = {}

        for task in tasks:
            started = time.perf_counter()
            hf_path, hf_config, split_map = suite.resolver(task)
            hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))

            def _loaders(
                preprocess: Any, hf_ds: Any = hf_ds, hf_path: Any = hf_path, split_map: Any = split_map
            ) -> Any:
                return build_vision_loaders(
                    hf_ds=hf_ds,
                    hf_path=hf_path,
                    preprocess=preprocess,
                    ft_epochs=1,
                    split_map=split_map,
                    batch_size=eval_bs,
                    num_workers=n_workers,
                    pin_memory=True,
                    val_fraction=val_fraction,
                    seed=_VAL_TEST_SPLIT_SEED,
                )

            loaders = _loaders(clf_target.preprocess)
            source_loaders = _loaders(clf_source.preprocess)
            classnames = list(loaders.classnames)
            if use_humanized:
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
            source_build_cfg_task = OpenClipBuildConfig(
                model_name=source_cfg.model_name,
                pretrained=source_cfg.pretrained,
                device=source_cfg.device,
                dtype=source_cfg.dtype,
                prompt_templates=templates,
            )
            item = {"task": task, "loaders": loaders, "classnames": classnames, "build_cfg_task": build_cfg_task}
            per_task.append(item)

            # Support set: the exact examples steer fits on, pre-processed once for each model.
            source_ds, target_ds = source_loaders.train.dataset, loaders.train.dataset
            if len(source_ds) != len(target_ds):
                raise ValueError(f"Teacher/student train splits differ in length for '{task}'.")
            support = _vision_support_indices(source_ds, task, kd["few_shot"], kd["seed"])
            xs_teacher, xs_student, ys = [], [], []
            for i in support:
                x_t, y_t = source_ds[i]
                x_s, y_s = target_ds[i]
                if int(y_t) != int(y_s):
                    raise RuntimeError(f"Teacher/student label mismatch at train index {i} for '{task}'.")
                xs_teacher.append(x_t)
                xs_student.append(x_s)
                ys.append(int(y_s))
            xs_teacher, xs_student = torch.stack(xs_teacher), torch.stack(xs_student)
            ys_t = torch.tensor(ys, dtype=torch.long)
            n_support = len(support)
            print(
                f"  {task}: support set = {n_support} examples ({kd['few_shot']} per class), classes={len(classnames)}"
            )

            # Teacher logits, once: the teacher is frozen and only needed on the support set.
            teacher = _load_finetuned_source_vision(clf_source, str(tuned_by_task[task]), task)
            teacher.to(dev).eval()
            teacher.build_zeroshot_text_features(classnames, source_build_cfg_task, cache_dir=None)
            teacher_logits = torch.cat(
                [_vision_logits(teacher, xs_teacher[i : i + eval_bs], dev).cpu() for i in range(0, n_support, eval_bs)]
            )
            teacher_support_acc = float((teacher_logits.argmax(-1) == ys_t).float().mean())
            del teacher, xs_teacher
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            # Student: untouched B, only the image tower trains; the zero-shot text head stays fixed.
            clf_target.model.load_state_dict(student_init, strict=True)
            clf_target.to(dev)
            clf_target.build_zeroshot_text_features(classnames, build_cfg_task, cache_dir=None)
            w_b = clf_target._zs_text_features.detach().float()
            if w_b.shape[0] != teacher_logits.shape[1]:
                raise ValueError(f"Teacher/student class count mismatch for '{task}'.")
            for p in clf_target.model.parameters():
                p.requires_grad_(False)
            visual_params = list(clf_target.model.visual.parameters())
            for p in visual_params:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(visual_params, lr=kd["lr"], weight_decay=kd["weight_decay"])
            steps_per_epoch = -(-n_support // kd["batch_size"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, kd["epochs"] * steps_per_epoch))

            def _snapshot(model: torch.nn.Module = clf_target.model) -> dict[str, torch.Tensor]:
                return {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}

            baseline_val = eval_task_top1(
                clf=clf_target, loaders=loaders, classnames=classnames, build_cfg_task=build_cfg_task,
                device=device, split="val",
            )  # fmt: skip
            baseline_test = eval_task_top1(
                clf=clf_target, loaders=loaders, classnames=classnames, build_cfg_task=build_cfg_task,
                device=device, split="test",
            )  # fmt: skip
            best_val, best_epoch = baseline_val, 0
            best_state = _snapshot() if kd["epoch_selection"] == "val" else None
            history: list[dict[str, float]] = [{"epoch": 0, "loss": float("nan"), "val": float(baseline_val)}]
            print(f"  {task}: teacher support acc={teacher_support_acc:.4f} | student zero-shot val={baseline_val:.4f}")

            for epoch in range(1, kd["epochs"] + 1):
                clf_target.model.train(kd["train_mode"] == "train")
                total_loss = 0.0
                for idx in _support_batches(n_support, kd["batch_size"], generator):
                    x = xs_student[idx].to(dev)
                    amp = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=kd["amp"] == "bf16")
                    with amp:
                        feats = F.normalize(clf_target.model.encode_image(x).float(), dim=-1)
                    s_logits = float(clf_target.logit_scale) * feats @ w_b.t()
                    loss = kd_loss(
                        s_logits,
                        teacher_logits[idx].to(dev),
                        temperature=kd["temperature"],
                        labels=ys_t[idx].to(dev),
                        ce_weight=kd["ce_weight"],
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if kd["grad_clip"] > 0.0:
                        torch.nn.utils.clip_grad_norm_(visual_params, kd["grad_clip"])
                    opt.step()
                    sched.step()
                    total_loss += float(loss.detach()) * len(idx)
                clf_target.model.eval()
                val_acc = eval_task_top1(
                    clf=clf_target, loaders=loaders, classnames=classnames, build_cfg_task=build_cfg_task,
                    device=device, split="val",
                )  # fmt: skip
                history.append({"epoch": epoch, "loss": total_loss / n_support, "val": float(val_acc)})
                run_logger.log_event(
                    "kd_epoch",
                    metrics={f"kd/{task}/loss": total_loss / n_support, f"kd/{task}/val_acc": float(val_acc)},
                    context={"task": task, "epoch": epoch},
                )
                print(f"    epoch {epoch:3d}  loss={total_loss / n_support:.5f}  val={val_acc:.4f}")
                if kd["epoch_selection"] == "val" and val_acc > best_val:  # ties keep the earlier epoch
                    best_val, best_epoch, best_state = val_acc, epoch, _snapshot()

            if kd["epoch_selection"] == "val":
                clf_target.model.visual.load_state_dict(best_state)
            else:
                best_epoch, best_val = kd["epochs"], history[-1]["val"]
            kd_test = eval_task_top1(
                clf=clf_target, loaders=loaders, classnames=classnames, build_cfg_task=build_cfg_task,
                device=device, split="test",
            )  # fmt: skip
            baseline_accs.append(float(baseline_test))
            kd_accs.append(float(kd_test))
            diagnostics[task] = {
                "n_support": n_support,
                "teacher_support_accuracy": teacher_support_acc,
                "epoch_selection": kd["epoch_selection"],
                "best_epoch": best_epoch,
                "best_val": float(best_val),
                "baseline_val": float(baseline_val),
                "history": history,
                "seconds": time.perf_counter() - started,
            }
            print(
                f"  {task}: epoch={best_epoch} ({kd['epoch_selection']}) | test baseline={baseline_test:.4f} kd={kd_test:.4f}"
            )
            del opt, sched, best_state, xs_student, teacher_logits
            for p in clf_target.model.parameters():
                p.requires_grad_(False)

        norm_accs = [_norm_acc(r, b) for r, b in zip(kd_accs, baseline_accs, strict=True)]
        pretty_print_task_accuracies(
            suite_name,
            "kd",
            f"A={source_cfg.pretrained} → B={target_cfg.pretrained}",
            per_task,
            kd_accs,
            norm_accs,
            single_accs=baseline_accs,
            baseline_label="target_zeroshot",
            result_label="kd",
        )
        run_logger.log_summary(
            _build_summary(
                cfg=cfg,
                kd=kd,
                suite_name=suite_name,
                tasks=tasks,
                baseline_accs=baseline_accs,
                kd_accs=kd_accs,
                norm_accs=norm_accs,
                task_diagnostics=diagnostics,
                extra={
                    "source_model": f"{source_cfg.model_name}/{source_cfg.pretrained}",
                    "target_model": f"{target_cfg.model_name}/{target_cfg.pretrained}",
                },
            )
        )
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def _check_hf_checkpoint_dir(ref: str) -> None:
    """Refuse an HF checkpoint directory this checkout cannot load faithfully.

    ``AutoModelForSequenceClassification`` on a ``T5EncoderForSequenceClassification`` directory (the
    ``checkpoints_t5_converted/`` layout, from another code version) would silently random-initialise the
    decoder and the head's ``dense`` layer, and the teacher logits would be noise.
    """
    import json

    config_path = Path(ref) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"'{ref}' is a directory without config.json: not an HF checkpoint.")
    architectures = json.loads(config_path.read_text()).get("architectures") or []
    arch = str(architectures[0]) if architectures else ""
    if not arch.endswith("ForSequenceClassification") or "Encoder" in arch:
        raise ValueError(
            f"'{ref}' is a '{arch or 'unknown'}' checkpoint; this entrypoint builds the teacher with "
            "AutoModelForSequenceClassification (e.g. T5ForSequenceClassification), whose decoder and head would "
            "be randomly initialised from it. Use a local .pt full checkpoint or a Hub id such as "
            "varun-v-rao/t5-base-snli."
        )


def _load_teacher_state_dict_text(ref: str, *, base_sd: dict[str, torch.Tensor], source_cfg: Any, model_kind: str):
    """``text_rebase._load_tuned_source_state_dict`` plus HF-format checkpoint *directories*.

    The original helper sends any existing path to ``load_ckpt``, which fails on a directory, so those are
    read with ``from_pretrained`` (after ``_check_hf_checkpoint_dir``).
    """
    from ..io.ckpt import align_to_base_keys
    from ..models.text_lm import TextBuildConfig, TextLM
    from .text_rebase import _load_tuned_source_state_dict

    if not Path(ref).is_dir():
        return _load_tuned_source_state_dict(ref, base_sd=base_sd, source_cfg=source_cfg, model_kind=model_kind)
    _check_hf_checkpoint_dir(ref)
    llm = TextLM.build(
        TextBuildConfig(
            model_name_or_path=ref,
            model_arch=source_cfg.model_arch,
            device="cpu",
            dtype=source_cfg.dtype,
            model_kind=model_kind,
            num_labels=source_cfg.num_labels,
            trust_remote_code=source_cfg.trust_remote_code,
            use_fast_tokenizer=source_cfg.use_fast_tokenizer,
        )
    )
    aligned = align_to_base_keys(dict(llm.model.state_dict()), base_sd)
    if not aligned:
        raise ValueError(f"No tensors from '{ref}' aligned to the source model's keys.")
    return aligned


def _text_batch_logits(model: torch.nn.Module, batch: dict[str, Any], dev: str, mask: torch.Tensor) -> torch.Tensor:
    ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev) if batch.get("attention_mask") is not None else None
    return model(input_ids=ids, attention_mask=attn).logits.index_select(1, mask.to(dev))


def run_text(args: argparse.Namespace) -> None:
    run_logger = None
    try:
        from torch.utils.data import DataLoader, Dataset

        from ..io.ckpt import load_into_model, resolve_ckpt_path
        from ..rebase.methods.steer import _few_shot
        from ..rebase.text.adapters import subset_loader
        from .llm_merge import _head_class_ids_for_task, _inject_task_head, _load_task_heads
        from .print_utils import pretty_print_task_accuracies
        from .text_rebase import (
            SUITES,
            _build_llm,
            _build_task_splits,
            _tokenize_splits,
        )
        from .utils import to_cpu_fp32

        cli = {
            "source_model_name_or_path": args.source_model_name_or_path,
            "target_model_name_or_path": args.target_model_name_or_path,
            "source_model_arch": args.source_model_arch,
            "target_model_arch": args.target_model_arch,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "suite": args.suite,
            "tasks": args.tasks,
            "device": args.device,
            "dtype": args.dtype,
            "target_task_heads": args.target_task_heads,
            "source_task_heads": args.source_task_heads,
        }
        cfg = _load_cfg(args, cli)
        kd = resolve_kd_config(cfg)
        device = str(cfg.get("device", "cuda"))
        model_kind = "sequence_classification"
        if str(cfg.get("eval_mode", "head_logits")) != "head_logits":
            raise ValueError("kd_rebase text only supports eval_mode='head_logits' (sequence-classification heads).")

        suite_name = str(cfg.get("suite", "nli6"))
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
        tasks_arg = cfg.get("tasks", "all")
        if tasks_arg == "all":
            tasks = list(SUITES[suite_name])
        else:
            tasks = [
                t.strip().lower() for t in (parse_csv(tasks_arg) if isinstance(tasks_arg, str) else list(tasks_arg))
            ]
        bad = [t for t in tasks if t not in SUITES[suite_name]]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(SUITES[suite_name])}")

        heads_path = cfg.get("target_task_heads", cfg.get("task_heads"))
        if not heads_path:
            raise ValueError(
                "kd_rebase text needs config['target_task_heads'] (e.g. a scripts/build_nearest_mean_head.py "
                "file): it is the student's starting head and defines the baseline, exactly as for steer_text."
            )
        target_heads = _load_task_heads(str(heads_path))
        source_heads = _load_task_heads(str(cfg["source_task_heads"])) if cfg.get("source_task_heads") else None
        head_key_pattern = str(cfg.get("head_key_pattern", "classification_head"))

        summary_path = default_summary_path(
            entrypoint="eval.kd_rebase", logging_cfg=cfg["logging"], default_parent=None
        )
        run_logger = start_run(
            entrypoint="eval.kd_rebase",
            logging_cfg=cfg["logging"],
            summary_path=summary_path,
            metadata={
                "config_path": args.config,
                "resolved_config": cfg,
                "suite": suite_name,
                "tasks": tasks,
                "summary_path": str(summary_path),
            },
        )

        tuned = {str(t).strip().lower(): resolve_ckpt_path(str(v)) for t, v in (cfg.get("tuned_ckpts") or {}).items()}
        missing = [t for t in tasks if t not in tuned]
        if missing:
            raise ValueError(f"tuned_ckpts is missing tasks: {missing}.")

        llm_source, source_cfg = _build_llm(cfg, role="source", model_kind=model_kind, device=device)
        llm_target, target_cfg = _build_llm(cfg, role="target", model_kind=model_kind, device=device)
        print(f"Teacher (A): {source_cfg.model_name_or_path} ({source_cfg.model_arch})")
        print(f"Student (B): {target_cfg.model_name_or_path} ({target_cfg.model_arch})")
        # Snapshot before any head injection, so every task restarts from the pristine pretrained student.
        target_base_sd = to_cpu_fp32({k: v for k, v in llm_target.model.state_dict().items()})
        target_dtypes = {k: v.dtype for k, v in llm_target.model.state_dict().items()}

        eval_split = str(cfg.get("split", "test"))
        val_fraction = float(cfg.get("val_fraction", 0.1))
        eval_bs = int(cfg.get("batch_size", 8))
        n_workers = int(cfg.get("num_workers", 0))
        max_length = int(cfg.get("max_length", 512))
        max_eval = cfg.get("max_samples_per_task")
        max_train = cfg.get("max_train_samples")
        generator = torch.Generator().manual_seed(kd["seed"])

        per_task: list[dict[str, Any]] = []
        baseline_accs: list[float] = []
        kd_accs: list[float] = []
        diagnostics: dict[str, Any] = {}

        class _Indexed(Dataset):
            """Support examples tagged with their position, so shuffled batches find their teacher logits."""

            def __init__(self, base: Any) -> None:
                self.base = base

            def __len__(self) -> int:
                return len(self.base)

            def __getitem__(self, i: int) -> tuple[Any, int]:
                return self.base[i], i

        for task in tasks:
            started = time.perf_counter()
            splits = _build_task_splits(
                task=task,
                eval_split=eval_split,
                val_fraction=val_fraction,
                max_train_samples=int(max_train) if max_train is not None else None,
                max_eval_samples=int(max_eval) if max_eval is not None else None,
            )
            head_class_ids = _head_class_ids_for_task(
                task=task,
                task_num_labels=len(splits["test"].labels),
                head_num_labels=int(cfg.get("num_labels", 3)),
                masked_class=(cfg.get("task_mask_class", {}) or {}).get(task, None),
            )

            def _tok(tokenizer: Any, template: Any, splits: Any = splits, head_class_ids: Any = head_class_ids) -> Any:
                return _tokenize_splits(
                    splits=splits,
                    tokenizer=tokenizer,
                    batch_size=eval_bs,
                    num_workers=n_workers,
                    max_length=max_length,
                    head_class_ids=head_class_ids,
                    premise_hypothesis_template=template,
                )

            loaders = _tok(llm_target.tokenizer, cfg.get("target_input_template"))
            source_loaders = _tok(llm_source.tokenizer, cfg.get("source_input_template"))
            mask = torch.tensor(loaders.mask_class, dtype=torch.long)
            per_task.append({"task": task, "loaders": loaders})

            # Support set: steer_text's draw (same helper, same pool, same seed).
            local_labels = torch.tensor(loaders.local_labels["train"], dtype=torch.long)
            support = _few_shot(local_labels, kd["few_shot"], kd["seed"]).tolist()
            n_support = len(support)
            print(f"  {task}: support set = {n_support} examples ({kd['few_shot']} per class)")

            # Teacher logits, once, in the teacher's own tokenization. Same order as `support`.
            teacher = deepcopy(llm_source)
            aligned = _load_teacher_state_dict_text(
                str(tuned[task]), base_sd=teacher.model.state_dict(), source_cfg=source_cfg, model_kind=model_kind
            )
            load_into_model(teacher.model, aligned, strict=False)
            if source_heads is not None:
                _inject_task_head(
                    model=teacher.model, task=task, task_heads=source_heads,
                    head_key_pattern=head_key_pattern, head_class_ids=head_class_ids,
                )  # fmt: skip
            teacher.model.to(device).eval()
            t_chunks = []
            with torch.no_grad():
                for batch in subset_loader(source_loaders.train, support, batch_size=eval_bs):
                    t_chunks.append(_text_batch_logits(teacher.model, batch, device, mask).float().cpu())
            teacher_logits = torch.cat(t_chunks)
            teacher_support_acc = float(
                (
                    mask[teacher_logits.argmax(-1)]
                    == torch.tensor([int(loaders.train.dataset[i]["labels"]) for i in support])
                )
                .float()
                .mean()
            )
            del teacher, aligned
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

            # Student: pristine B + the task's starting head (nearest-mean), then all parameters train.
            load_into_model(llm_target.model, target_base_sd, strict=False)
            _inject_task_head(
                model=llm_target.model, task=task, task_heads=target_heads,
                head_key_pattern=head_key_pattern, head_class_ids=head_class_ids,
            )  # fmt: skip
            model = llm_target.model.to(device)

            def _eval(split: str, loaders: Any = loaders) -> float:
                return float(
                    llm_target.sequence_classification_accuracy(
                        loaders.val if split == "val" else loaders.test, device=device, mask_class=loaders.mask_class
                    )
                )

            def _snapshot(model: torch.nn.Module = model) -> dict[str, torch.Tensor]:
                return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            baseline_val, baseline_test = _eval("val"), _eval("test")
            best_val, best_epoch = baseline_val, 0
            best_state = _snapshot() if kd["epoch_selection"] == "val" else None
            history: list[dict[str, float]] = [{"epoch": 0, "loss": float("nan"), "val": float(baseline_val)}]
            print(f"  {task}: teacher support acc={teacher_support_acc:.4f} | student baseline val={baseline_val:.4f}")

            for p in model.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=kd["lr"], weight_decay=kd["weight_decay"]
            )
            steps_per_epoch = -(-n_support // kd["batch_size"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, kd["epochs"] * steps_per_epoch))
            student_train_ds = _Indexed(torch.utils.data.Subset(loaders.train.dataset, support))
            inner_collate = loaders.train.collate_fn

            def _collate(batch: list[Any], inner_collate: Any = inner_collate) -> Any:
                out = inner_collate([b[0] for b in batch])
                out["support_index"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
                return out

            train_loader = DataLoader(
                student_train_ds, batch_size=kd["batch_size"], shuffle=True, collate_fn=_collate,
                generator=generator, num_workers=0,
            )  # fmt: skip

            for epoch in range(1, kd["epochs"] + 1):
                model.train(kd["train_mode"] == "train")
                total_loss = 0.0
                for batch in train_loader:
                    sidx = batch["support_index"]
                    amp = torch.autocast(
                        device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=kd["amp"] == "bf16"
                    )
                    with amp:
                        s_logits = _text_batch_logits(model, batch, device, mask)
                    labels_local = torch.tensor(
                        [loaders.mask_class.index(int(y)) for y in batch["labels"]], dtype=torch.long, device=device
                    )
                    loss = kd_loss(
                        s_logits,
                        teacher_logits[sidx].to(device),
                        temperature=kd["temperature"],
                        labels=labels_local,
                        ce_weight=kd["ce_weight"],
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if kd["grad_clip"] > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), kd["grad_clip"])
                    opt.step()
                    sched.step()
                    total_loss += float(loss.detach()) * len(sidx)
                val_acc = _eval("val")  # also switches the model to eval()
                history.append({"epoch": epoch, "loss": total_loss / n_support, "val": float(val_acc)})
                run_logger.log_event(
                    "kd_epoch",
                    metrics={f"kd/{task}/loss": total_loss / n_support, f"kd/{task}/val_acc": float(val_acc)},
                    context={"task": task, "epoch": epoch},
                )
                print(f"    epoch {epoch:3d}  loss={total_loss / n_support:.5f}  val={val_acc:.4f}")
                if kd["epoch_selection"] == "val" and val_acc > best_val:  # ties keep the earlier epoch
                    best_val, best_epoch, best_state = val_acc, epoch, _snapshot()

            if kd["epoch_selection"] == "val":
                model.load_state_dict({k: v.to(target_dtypes[k]) for k, v in best_state.items()})
            else:
                best_epoch, best_val = kd["epochs"], history[-1]["val"]
            kd_test = _eval("test")
            baseline_accs.append(float(baseline_test))
            kd_accs.append(float(kd_test))
            diagnostics[task] = {
                "n_support": n_support,
                "teacher_support_accuracy": teacher_support_acc,
                "epoch_selection": kd["epoch_selection"],
                "best_epoch": best_epoch,
                "best_val": float(best_val),
                "baseline_val": float(baseline_val),
                "history": history,
                "seconds": time.perf_counter() - started,
            }
            print(
                f"  {task}: epoch={best_epoch} ({kd['epoch_selection']}) | test baseline={baseline_test:.4f} kd={kd_test:.4f}"
            )
            del opt, sched, best_state, teacher_logits
            for p in model.parameters():
                p.requires_grad_(False)

        norm_accs = [_norm_acc(r, b) for r, b in zip(kd_accs, baseline_accs, strict=True)]
        pretty_print_task_accuracies(
            suite_name,
            "kd",
            f"A={source_cfg.model_name_or_path} → B={target_cfg.model_name_or_path}",
            per_task,
            kd_accs,
            norm_accs,
            single_accs=baseline_accs,
            baseline_label="target_zeroshot",
            result_label="kd",
        )
        run_logger.log_summary(
            _build_summary(
                cfg=cfg,
                kd=kd,
                suite_name=suite_name,
                tasks=tasks,
                baseline_accs=baseline_accs,
                kd_accs=kd_accs,
                norm_accs=norm_accs,
                task_diagnostics=diagnostics,
                extra={
                    "eval_mode": "head_logits",
                    "source_model": source_cfg.model_name_or_path,
                    "target_model": target_cfg.model_name_or_path,
                },
            )
        )
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot knowledge-distillation baseline for steer / steer_text.")
    sub = parser.add_subparsers(dest="modality", required=True)

    pv = _common_parser("vision", "OpenCLIP teacher -> OpenCLIP student.", sub)
    pv.add_argument("--source-clip-model", type=str, default=None)
    pv.add_argument("--source-clip-pretrained", type=str, default=None)
    pv.add_argument("--target-clip-model", type=str, default=None)
    pv.add_argument("--target-clip-pretrained", type=str, default=None)
    pv.set_defaults(func=run_vision)

    pt = _common_parser(
        "text", "HF sequence-classification teacher -> student (e.g. T5-base -> T5-large / RoBERTa).", sub
    )
    pt.add_argument("--source-model-name-or-path", type=str, default=None)
    pt.add_argument("--target-model-name-or-path", type=str, default=None)
    pt.add_argument("--source-model-arch", type=str, default=None)
    pt.add_argument("--target-model-arch", type=str, default=None)
    pt.add_argument("--target-task-heads", type=str, default=None)
    pt.add_argument("--source-task-heads", type=str, default=None)
    pt.set_defaults(func=run_text)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
