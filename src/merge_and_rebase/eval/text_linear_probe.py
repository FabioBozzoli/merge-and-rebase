"""
Few-shot linear probing of a single text classifier -- no rebasin, no task vector.

This is the control the steer_text/theseus runs are measured against, and the text
twin of ``eval/vision_linear_probe.py``. It builds exactly one model (the target),
initializes its classification head from nearest-class-mean centroids over a
few-shot support set, trains that head with the backbone frozen, and reports
accuracy before and after training.

Before this file existed the same control ran as ``steer_text`` with ``alpha=0``,
which meant a config had to carry A's checkpoints, a feature cache and a stage-2
strategy purely to have them multiplied by zero. None of that is here: there is no
source model, no delta, no alpha and no method.

The split carve, the support draw, the head init and the probe loop are *imported*
from the modules the rebasin runs use, so the control cannot drift away from what
it is a control for.

Usage:
    python -m merge_and_rebase.eval.text_linear_probe \\
        --config configs/text_probe_t5large_nearest_mean.json
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
from ..data.text_loaders import default_head_class_ids_for_task
from ..rebase.text.adapters import balanced_indices, subset_loader, train_linear_probe_head
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run

# Imported, not reimplemented: these are the exact split carve, model build, head
# init and loader resolution every rebasin run uses. A copy here would be free to
# drift -- a different val/test carve or head-class mapping would make this control
# incomparable to the runs it exists to bound, and nothing would fail loudly.
from .text_rebase import (
    SUITES,
    _build_llm,
    _build_task_splits,
    _nearest_mean_init_head,
    _resolve_eval_loader,
    _tokenize_splits,
)

_ENTRYPOINT = "eval.text_linear_probe"


def probe_task(
    *,
    llm: Any,
    task: str,
    cfg: dict[str, Any],
    device: str,
    seed: int,
    shots: int,
    epochs: int,
    lr: float,
    dropout: bool,
    log_every: int | None,
) -> dict[str, Any]:
    """Probe one task and return its row of the summary.

    The head is rebuilt from the support set for every task, so tasks are
    independent and their order does not matter.
    """
    num_labels = int(cfg.get("num_labels", 3))
    splits = _build_task_splits(
        task=task,
        eval_split=str(cfg.get("split", "test")),
        val_fraction=float(cfg.get("val_fraction", 0.1)),
        max_train_samples=(
            int(cfg["max_train_samples"]) if cfg.get("max_train_samples") is not None else None
        ),
        max_eval_samples=(
            int(cfg["max_samples_per_task"]) if cfg.get("max_samples_per_task") is not None else None
        ),
    )
    loaders = _tokenize_splits(
        splits=splits,
        tokenizer=llm.tokenizer,
        batch_size=int(cfg.get("batch_size", 16)),
        num_workers=int(cfg.get("num_workers", 0)),
        max_length=int(cfg.get("max_length", 256)),
        head_class_ids=default_head_class_ids_for_task(task, num_labels),
        premise_hypothesis_template=cfg.get("premise_hypothesis_template", None),
    )

    support_indices = balanced_indices(loaders.local_labels["train"], shots, seed=seed)
    support_loader = subset_loader(loaders.train, support_indices, batch_size=int(cfg.get("batch_size", 16)))
    support_labels = [int(loaders.local_labels["train"][i]) for i in support_indices]
    print(f"  {task}: {len(support_indices)} support examples ({shots}/class)")

    _nearest_mean_init_head(
        model=llm.model,
        loader=support_loader,
        local_labels=support_labels,
        mask_class=loaders.mask_class,
        device=device,
    )

    def _accuracy(split: str) -> float:
        return float(
            llm.sequence_classification_accuracy(
                _resolve_eval_loader(loaders, split), device=device, mask_class=loaders.mask_class
            )
        )

    # The head as nearest-mean centroids alone, before any gradient step: the number
    # the probe has to improve on, and the same quantity the rebasin runs report as
    # the target's zero-shot column (they load an equivalent head from a file).
    init_test = _accuracy("test")
    print(f"  {task}: nearest-mean init test accuracy = {init_test:.4f}")

    train_linear_probe_head(
        llm.model,
        support_loader,
        device=device,
        mask_class=loaders.mask_class,
        lr=lr,
        steps=epochs,
        eval_loaders={"support": support_loader, "val": loaders.val, "test": loaders.test},
        log_every=log_every,
        log_prefix=f"  [probe:{task}]",
        dropout=dropout,
    )

    probed_test, probed_val = _accuracy("test"), _accuracy("val")
    print(f"  {task}: probed test accuracy = {probed_test:.4f}  (val {probed_val:.4f})")
    return {
        "task": task,
        "support_size": len(support_indices),
        "init_test_accuracy": init_test,
        "probed_test_accuracy": probed_test,
        "probed_val_accuracy": probed_val,
    }


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser(description="Few-shot linear probing of a single text classifier (no rebasin).")
        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")
        add_device_dtype_args(p, device_default=None, dtype_default=None)
        p.add_argument("--target-model-name-or-path", type=str, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--split", type=str, default=None, choices=["val", "test"])
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
            "target_model_name_or_path": args.target_model_name_or_path,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "split": args.split,
            "linear_probe_shots_per_class": args.linear_probe_shots_per_class,
            "linear_probe_epochs": args.linear_probe_epochs,
            "linear_probe_lr": args.linear_probe_lr,
            "linear_probe_log_every": args.linear_probe_log_every,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging"), build_logging_overrides(args))

        suite_name = str(cfg.get("suite", "nli6"))
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
        suite_tasks = SUITES[suite_name]
        tasks_arg = cfg.get("tasks", "all")
        if tasks_arg == "all":
            tasks = list(suite_tasks)
        else:
            raw = parse_csv(tasks_arg) if isinstance(tasks_arg, str) else list(tasks_arg)
            tasks = [str(t).strip().lower() for t in raw]
            bad = [t for t in tasks if t not in suite_tasks]
            if bad:
                raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite_tasks)}")

        shots = cfg.get("linear_probe_shots_per_class", None)
        if shots is None:
            raise ValueError("linear_probe_shots_per_class is required: it is the probe's support-set size.")
        shots = int(shots)
        epochs = int(cfg.get("linear_probe_epochs", 200))
        lr = float(cfg.get("linear_probe_lr", 1e-4))
        # False keeps the frozen backbone in eval(), which also lets the probe loop
        # cache the head's inputs once instead of recomputing them every epoch.
        dropout = bool(cfg.get("linear_probe_dropout", False))
        log_every = cfg.get("linear_probe_log_every", None)
        log_every = int(log_every) if log_every is not None else None
        init = str(cfg.get("linear_probe_init", "nearest_mean")).strip().lower()
        if init != "nearest_mean":
            raise ValueError(f"linear_probe_init must be 'nearest_mean' here, got '{init}'.")
        device = str(cfg.get("device", "cuda"))
        seed = int(cfg.get("seed", 42))
        split = str(cfg.get("split", "test"))

        run_summary_path = default_summary_path(
            entrypoint=_ENTRYPOINT, logging_cfg=logging_cfg, default_parent=None
        )
        run_logger = start_run(
            entrypoint=_ENTRYPOINT,
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

        model_kind = str(cfg.get("model_kind", "encoder_classification"))
        llm, build_cfg = _build_llm(cfg, role="target", model_kind=model_kind, device=device)
        print(f"Model: {build_cfg.model_name_or_path} ({build_cfg.model_arch}, {model_kind})")
        print(f"Probe: {shots}/class, {epochs} epochs, lr {lr}, nearest-mean init, dropout {'on' if dropout else 'off'}")

        rows = [
            probe_task(
                llm=llm, task=task, cfg=cfg, device=device, seed=seed, shots=shots,
                epochs=epochs, lr=lr, dropout=dropout, log_every=log_every,
            )
            for task in tasks
        ]
        for row in rows:
            run_logger.log_event(
                "linear_probe_task_end",
                metrics={
                    f"probe/{row['task']}/init_test_acc": row["init_test_accuracy"],
                    f"probe/{row['task']}/probed_test_acc": row["probed_test_accuracy"],
                },
                context={"task": row["task"], "shots_per_class": shots, "epochs": epochs, "lr": lr},
            )

        init_by_task = {r["task"]: r["init_test_accuracy"] for r in rows}
        probed_by_task = {r["task"]: r["probed_test_accuracy"] for r in rows}
        # probed / init, i.e. what probing added over the head it started from. The
        # rebasin summaries call this "norm" against their own baseline column, and
        # the collector reads these key names from both.
        norm_by_task = {
            t: (probed_by_task[t] / init_by_task[t] if init_by_task[t] else float("nan")) for t in probed_by_task
        }

        task_col = max(max((len(t) for t in tasks), default=4), len("task"), len("avg"))
        print(f"\nLinear probe ({shots}/class, {epochs} epochs) - {build_cfg.model_name_or_path} - split: {split}")
        print(f"{'task':>{task_col}}  {'init':>12}  {'probed':>12}  {'delta':>12}")
        print(f"{'-' * task_col}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
        for t in tasks:
            print(f"{t:>{task_col}}  {init_by_task[t]:>12.6f}  {probed_by_task[t]:>12.6f}  "
                  f"{probed_by_task[t] - init_by_task[t]:>+12.6f}")
        avg_init = sum(init_by_task.values()) / max(1, len(init_by_task))
        avg_probed = sum(probed_by_task.values()) / max(1, len(probed_by_task))
        print(f"{'-' * task_col}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
        print(f"{'avg':>{task_col}}  {avg_init:>12.6f}  {avg_probed:>12.6f}  {avg_probed - avg_init:>+12.6f}")

        run_logger.log_summary(
            {
                "suite": suite_name,
                "tasks": tasks,
                "entrypoint": _ENTRYPOINT,
                "target_model": build_cfg.model_name_or_path,
                "model_kind": model_kind,
                "split": split,
                "seed": seed,
                "linear_probe_shots_per_class": shots,
                "linear_probe_epochs": epochs,
                "linear_probe_lr": lr,
                "linear_probe_init": init,
                "linear_probe_dropout": dropout,
                "per_task_support_size": {r["task"]: r["support_size"] for r in rows},
                "per_task_val_accuracy": {r["task"]: r["probed_val_accuracy"] for r in rows},
                # Same key names the rebasin entrypoints write, so grid.py's state
                # check and collect_t5enc_results.py read this summary unchanged.
                # "baseline" is the nearest-mean head before training, "rebased" the
                # probed head -- the two columns this control is meant to compare.
                "test_results": {
                    "per_task_baseline": init_by_task,
                    "per_task_rebased": probed_by_task,
                    "per_task_norm": norm_by_task,
                    "per_task_baseline_accuracy": init_by_task,
                    "per_task_absolute_accuracy": probed_by_task,
                    "per_task_normalized_accuracy_ratio": norm_by_task,
                    "avg_rebased": avg_probed,
                    "avg_norm": sum(norm_by_task.values()) / max(1, len(norm_by_task)),
                },
            }
        )
        run_logger.finish("success")
    except Exception as err:  # noqa: BLE001 -- mirrors the other entrypoints
        finish_with_error(run_logger, err)
        raise


if __name__ == "__main__":
    main()
