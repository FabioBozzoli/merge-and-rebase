"""
Few-shot linear probing of a single CLIP model -- no rebasin, no task vector.

This is the control the rebasin+probe runs are measured against. It builds
exactly one model, the ``clip_model``/``clip_pretrained`` pair given in the
config or on the command line, and uses only that pretrain's own backbone and
its own zero-shot heads. There is no source model, no fine-tuned checkpoint and
no delta anywhere in this file: the number it reports is what a few-shot linear
probe gets on the untouched pretrained model.

For each task it reports the zero-shot accuracy and the probed accuracy on the
same split, so the delta between the two columns is what probing added.

Probing itself is ``eval/linear_probe.py``'s ``train_zeroshot_head_probe``: the
backbone stays frozen and only the zero-shot head (``[C, D]``) is trained,
starting from that head rather than from a random draw -- identical to what
``eval/vision_rebase.py`` does after transport, so the two are comparable.

Usage:
    python -m merge_and_rebase.eval.vision_linear_probe \\
        --config configs/vision8_zeroshot_linear_probe.json
"""

from __future__ import annotations

import argparse
from typing import Any

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
from ..data.templates import get_templates
from ..data.vision_loaders import build_vision_loaders, load_hf_splits
from ..eval.utils import build_grad_dataloader, eval_task_top1, humanize
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from .datasets.vision8_14_20 import SUITES
from .linear_probe import train_zeroshot_head_probe

# Same constant, and same reason, as eval/vision_rebase.py: the val/test carve
# must not move when cfg["seed"] changes, so a seed sweep only redraws the
# support set and not the held-out data.
_VAL_TEST_SPLIT_SEED = 0


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser(description="Few-shot linear probing of a single CLIP model (no rebasin).")
        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")
        add_device_dtype_args(p, device_default=None, dtype_default=None)
        p.add_argument("--clip-model", type=str, default=None)
        p.add_argument("--clip-pretrained", type=str, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--val-fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--split", type=str, default=None, choices=["val", "test"])
        p.add_argument("--no-humanize", action="store_true", default=None)
        p.add_argument("--linear-probe-shots-per-class", type=int, default=None)
        p.add_argument("--linear-probe-epochs", type=int, default=None)
        p.add_argument("--linear-probe-lr", type=float, default=None)
        p.add_argument("--linear-probe-log-every", type=int, default=None)
        add_logging_args(p)

        args = p.parse_args()
        cfg: dict[str, Any] = load_json(args.config) if args.config is not None else {}
        cli = {
            "suite": getattr(args, "suite", None),
            "tasks": getattr(args, "tasks", None),
            "device": args.device,
            "dtype": args.dtype,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "split": args.split,
            "no_humanize": args.no_humanize,
            "linear_probe_shots_per_class": args.linear_probe_shots_per_class,
            "linear_probe_epochs": args.linear_probe_epochs,
            "linear_probe_lr": args.linear_probe_lr,
            "linear_probe_log_every": args.linear_probe_log_every,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging"), build_logging_overrides(args))

        suite_name = str(cfg.get("suite", "vision8"))
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Allowed: {sorted(SUITES)}")
        suite = SUITES[suite_name]
        tasks_arg = cfg.get("tasks", "all")
        tasks = (
            list(suite.tasks)
            if tasks_arg == "all"
            else [t.strip() for t in (parse_csv(tasks_arg) if isinstance(tasks_arg, str) else list(tasks_arg))]
        )
        bad = [t for t in tasks if t not in suite.tasks]
        if bad:
            raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite.tasks)}")

        # The one and only model this entrypoint knows about. Accepts the
        # rebasin configs' "target_clip_*" spelling too, so a zero-shot control
        # can be pointed at the same B a transport run used without editing keys.
        model_name = cfg.get("clip_model", cfg.get("target_clip_model", None))
        pretrained = cfg.get("clip_pretrained", cfg.get("target_clip_pretrained", None))
        if model_name is None or pretrained is None:
            raise ValueError("Set clip_model and clip_pretrained (or target_clip_model/target_clip_pretrained).")

        device = str(cfg.get("device", "cuda"))
        split = str(cfg.get("split", "test"))
        seed = int(cfg.get("seed", 42))
        shots = cfg.get("linear_probe_shots_per_class", None)
        if shots is None:
            raise ValueError("linear_probe_shots_per_class is required: it is the probe's support-set size.")
        shots = int(shots)
        epochs = int(cfg.get("linear_probe_epochs", 50))
        lr = float(cfg.get("linear_probe_lr", 1e-3))
        log_every = cfg.get("linear_probe_log_every", None)
        log_every = int(log_every) if log_every is not None else None
        use_humanized_classnames = not bool(cfg.get("no_humanize", True))

        run_summary_path = default_summary_path(
            entrypoint="eval.vision_linear_probe", logging_cfg=logging_cfg, default_parent=None
        )
        run_logger = start_run(
            entrypoint="eval.vision_linear_probe",
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

        build_cfg = OpenClipBuildConfig(
            model_name=str(model_name),
            pretrained=str(pretrained),
            device=device,
            dtype=cfg.get("dtype", None),
        )
        print(f"Model: {build_cfg.model_name} / {build_cfg.pretrained}  (no rebasin, no task vector)")
        clf = OpenClipClassifier.build(build_cfg)

        zeroshot_by_task: dict[str, float] = {}
        probed_by_task: dict[str, float] = {}

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
                seed=_VAL_TEST_SPLIT_SEED,
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

            print(f"\n--- {task} ---")
            zeroshot_acc = float(
                eval_task_top1(
                    clf=clf,
                    loaders=loaders,
                    classnames=classnames,
                    build_cfg_task=build_cfg_task,
                    device=device,
                    split=split,
                )
            )
            zeroshot_by_task[task] = zeroshot_acc
            print(f"  {task}: zero-shot {split} accuracy = {zeroshot_acc:.4f}")

            # eval_task_top1 has just built this task's zero-shot head into clf;
            # the probe starts from it, so its epoch-0 line reproduces the number
            # above and every later line is what probing added.
            probe_loader = build_grad_dataloader(
                loaders.train,
                loaders.train.dataset,
                grad_imgs_per_class=shots,
                num_workers=int(cfg.get("num_workers", 6)),
                seed=seed,
            )
            print(f"  {task}: linear-probing on {len(probe_loader.dataset)} support images ({shots}/class)")
            head = train_zeroshot_head_probe(
                clf,
                probe_loader,
                device=device,
                lr=lr,
                steps=epochs,
                eval_loaders={"support": probe_loader, "val": loaders.val, "test": loaders.test},
                log_every=log_every,
                log_prefix=f"  [probe:{task}]",
            )
            probed_acc = float(
                eval_task_top1(
                    clf=clf,
                    loaders=loaders,
                    classnames=classnames,
                    build_cfg_task=build_cfg_task,
                    device=device,
                    split=split,
                    text_features=head,
                )
            )
            probed_by_task[task] = probed_acc
            print(f"  {task}: probed {split} accuracy = {probed_acc:.4f}")
            run_logger.log_event(
                "linear_probe_task_end",
                metrics={
                    f"probe/{task}/zeroshot_{split}_acc": zeroshot_acc,
                    f"probe/{task}/probed_{split}_acc": probed_acc,
                },
                context={"task": task, "shots_per_class": shots, "epochs": epochs, "lr": lr},
            )

        print(
            f"\nLinear probe ({shots}/class, {epochs} epochs) - "
            f"{build_cfg.model_name} / {build_cfg.pretrained} - split: {split}"
        )
        task_col = max(max((len(t) for t in tasks), default=4), len("task"), len("avg"))
        print(f"{'task':>{task_col}}  {'zero_shot':>12}  {'probed':>12}  {'delta':>12}")
        print(f"{'-' * task_col}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
        for t in tasks:
            zs, pr = zeroshot_by_task[t], probed_by_task[t]
            print(f"{t:>{task_col}}  {zs:>12.6f}  {pr:>12.6f}  {pr - zs:>+12.6f}")
        avg_zs = sum(zeroshot_by_task.values()) / max(1, len(zeroshot_by_task))
        avg_pr = sum(probed_by_task.values()) / max(1, len(probed_by_task))
        print(f"{'-' * task_col}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
        print(f"{'avg':>{task_col}}  {avg_zs:>12.6f}  {avg_pr:>12.6f}  {avg_pr - avg_zs:>+12.6f}")

        run_logger.log_summary(
            {
                "suite": suite_name,
                "tasks": tasks,
                "entrypoint": "eval.vision_linear_probe",
                "clip_model": build_cfg.model_name,
                "clip_pretrained": build_cfg.pretrained,
                "split": split,
                "linear_probe_shots_per_class": shots,
                "linear_probe_epochs": epochs,
                "linear_probe_lr": lr,
                "per_task_zeroshot_accuracy": zeroshot_by_task,
                "per_task_probed_accuracy": probed_by_task,
                "avg_zeroshot_accuracy": avg_zs,
                "avg_probed_accuracy": avg_pr,
            }
        )
        run_logger.finish("success")
    except Exception as err:  # noqa: BLE001 -- mirrors the other entrypoints
        finish_with_error(run_logger, err)
        raise


if __name__ == "__main__":
    main()
