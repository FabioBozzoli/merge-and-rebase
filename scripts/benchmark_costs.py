#!/usr/bin/env python3
"""Isolated, instrumented cost experiment for the repository's vision baselines."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import importlib.abc
import importlib.machinery
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
    }


def rss_tree():
    """Linux resident bytes, summed over this process and DataLoader descendants."""
    pending, total = [os.getpid()], 0
    while pending:
        pid = pending.pop()
        try:
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
        except (OSError, ValueError, IndexError):
            pass
    return total


class Meter:
    def __init__(self, torch, warmup=3, device="cpu"):
        self.torch, self.warmup = torch, warmup
        self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.started = time.perf_counter()
        self.stack, self.records, self.batches, self.curve = [], [], [], []
        self.peak_allocated = self.peak_reserved = self.peak_rss = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        while not self.stop.is_set():
            rss = rss_tree()
            with self.lock:
                self.peak_rss = max(self.peak_rss, rss)
                for frame in self.stack:
                    frame["peak_rss_bytes"] = max(frame["peak_rss_bytes"], rss)
            self.stop.wait(0.02)

    def boundary(self):
        if not self.cuda:
            return
        self.torch.cuda.synchronize()
        allocated = self.torch.cuda.max_memory_allocated()
        reserved = self.torch.cuda.max_memory_reserved()
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        with self.lock:
            for frame in self.stack:
                frame["peak_allocated_bytes"] = max(frame["peak_allocated_bytes"], allocated)
                frame["peak_reserved_bytes"] = max(frame["peak_reserved_bytes"], reserved)
        self.torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def phase(self, name):
        self.boundary()
        frame = {
            "name": name,
            "seconds": 0.0,
            "child_seconds": 0.0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "peak_rss_bytes": rss_tree(),
        }
        with self.lock:
            self.stack.append(frame)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.boundary()
            frame["seconds"] = time.perf_counter() - start
            frame["exclusive_seconds"] = frame["seconds"] - frame["child_seconds"]
            with self.lock:
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["child_seconds"] += frame["seconds"]
            self.records.append(frame)

    @contextlib.contextmanager
    def batch(self, name, size):
        # Events are resolved after the run; no per-batch synchronization is introduced.
        start = time.perf_counter()
        events = None
        if self.cuda:
            events = (self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True))
            events[0].record()
        try:
            yield
        finally:
            if events:
                events[1].record()
            self.batches.append(
                {"name": name, "size": int(size), "host_seconds": time.perf_counter() - start, "events": events}
            )

    def finish(self):
        self.boundary()
        self.stop.set()
        self.sampler.join()
        groups = defaultdict(list)
        for b in self.batches:
            groups[(b["name"], b["size"])].append(b)
        batch_stats = {}
        for (name, size), items in groups.items():
            steady = items[self.warmup :]
            batch_stats[f"{name}/batch_size={size}"] = {
                "calls": len(items),
                "images_processed": sum(b["size"] for b in items),
                "warmup_discarded": min(len(items), self.warmup),
                "host_dispatch_seconds": stats([b["host_seconds"] for b in steady]),
                "cuda_stream_span_seconds": stats(
                    [b["events"][0].elapsed_time(b["events"][1]) / 1000 for b in steady if b["events"]]
                ),
            }
        grouped = defaultdict(list)
        for record in self.records:
            grouped[record["name"]].append(record)
        phases = {
            name: {
                "calls": len(items),
                "inclusive_seconds": sum(i["seconds"] for i in items),
                "exclusive_seconds": sum(i["exclusive_seconds"] for i in items),
                "peak_allocated_bytes": max(i["peak_allocated_bytes"] for i in items),
                "peak_reserved_bytes": max(i["peak_reserved_bytes"] for i in items),
                "peak_rss_bytes": max(i["peak_rss_bytes"] for i in items),
            }
            for name, items in grouped.items()
        }
        return {
            "phases": phases,
            "batches": batch_stats,
            "curve": self.curve,
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
            "peak_rss_tree_bytes": self.peak_rss,
            "phase_records": self.records,
        }


# Instrument only these repository modules, in memory. Source files are never rewritten.
MODULE_PHASES = {
    "merge_and_rebase.eval.linear_probe": {
        "train_zeroshot_head_probe": "probe_pipeline",
        "_probe_accuracy": "evaluation",
    },
    "merge_and_rebase.eval.kd_rebase": {
        "_load_finetuned_source_vision": "teacher_loading",
        "_vision_logits": "teacher_forward",
    },
    "merge_and_rebase.rebase.methods.steer": {
        "prepare": "refract_prepare",
        "transport": "refract_transport",
        "_collect_linear_split": "extraction",
        "_collect_standard_split": "extraction",
        "_collect_target_blocks_split": "extraction",
        "_load_cached_split": "cache_read",
        "_save_split_cache": "cache_write",
        "_stage1_projection": "stage1_fit",
        "_stage1_target_corrections": "stage1_targets",
        "_fit_block_ridge": "stage2_fit",
        "_fit_global_mlp": "stage2_fit",
    },
}


class Instrument(ast.NodeTransformer):
    def __init__(self, module):
        self.module, self.function = module, None
        self.counts = defaultdict(int)

    def context(self, kind, name, body, size=None):
        args = [ast.Constant(name)] + ([ast.parse(size, mode="eval").body] if size else [])
        node = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="_cost_meter", ctx=ast.Load()), attr=kind, ctx=ast.Load()),
                        args=args,
                        keywords=[],
                    )
                )
            ],
            body=body,
        )
        self.counts[f"{kind}:{name}"] += 1
        return node

    def visit_FunctionDef(self, node):
        old, self.function = self.function, node.name
        node = self.generic_visit(node)
        name = MODULE_PHASES[self.module].get(node.name)
        if node.name == "prepare":
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "f_a" for t in statement.targets
                ):
                    node.body[index:] = [self.context("phase", "fitting_and_diagnostics", node.body[index:])]
                    break
        if node.name == "_vision_logits":
            node.body = [self.context("batch", "teacher_forward", node.body, "len(images)")]
        if name:
            # Retain docstrings as the first statement.
            offset = int(
                bool(
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
            )
            node.body[offset:] = [self.context("phase", name, node.body[offset:])]
        self.function = old
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if (
            self.function == "train_zeroshot_head_probe"
            and isinstance(node.target, ast.Tuple)
            and any(isinstance(n, ast.Name) and n.id == "images" for n in ast.walk(node.target))
        ):
            node.body = [self.context("batch", "train", node.body, "len(images)")]
            return self.context("phase", "training", [node])
        if self.function == "run_vision" and isinstance(node.target, ast.Name) and node.target.id == "idx":
            node.body = [self.context("batch", "train", node.body, "len(idx)")]
            return self.context("phase", "training", [node])
        if self.function in {"_collect_linear_split", "_collect_standard_split"} and isinstance(node.target, ast.Tuple):
            # Only the paired source/target image loop, never per-block loops.
            if any(isinstance(n, ast.Name) and n.id == "x_a" for n in ast.walk(node.target)):
                node.body = [self.context("batch", "extraction", node.body, "len(x_a)")]
        return node

    def visit_Assign(self, node):
        if self.function in {"_collect_linear_split", "_collect_standard_split"}:
            categories = {
                "f0": "source_forward_pretrained",
                "f_a_pre": "source_forward_pretrained",
                "f_a_ft": "source_forward_finetuned",
                "full_out": "source_jvp_full",
                "block_out": "source_jvp_block",
            }
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in categories:
                    return self.context("batch", categories[target.id], [node], "len(x_a)")
        if self.function == "run_vision" and any(
            isinstance(t, ast.Name) and t.id == "teacher_logits" for t in node.targets
        ):
            return self.context("phase", "teacher_precompute", [node])
        return self.generic_visit(node)


class InstrumentLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        tree = ast.parse(self.get_source(self.name), filename=self.path)
        transform = Instrument(self.name)
        tree = transform.visit(tree)
        module._cost_meter = self.meter
        self.counts[self.name] = dict(transform.counts)
        exec(compile(ast.fix_missing_locations(tree), self.path, "exec"), module.__dict__)


class InstrumentFinder(importlib.abc.MetaPathFinder):
    def __init__(self, meter):
        self.meter, self.counts = meter, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULE_PHASES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            loader = InstrumentLoader(fullname, spec.origin)
            loader.meter, loader.counts = self.meter, self.counts
            spec.loader = loader
        return spec


def patch_runtime(meter, cfg, method):
    from merge_and_rebase.eval import utils
    from merge_and_rebase.models.openclip_classifier import OpenClipClassifier

    original_eval = utils.eval_task_top1

    def evaluate(*args, **kwargs):
        with meter.phase("evaluation"):
            value = original_eval(*args, **kwargs)
        meter.curve.append(
            {
                "elapsed_seconds": time.perf_counter() - meter.started,
                "training_seconds": sum(r["seconds"] for r in meter.records if r["name"] == "training"),
                "split": kwargs.get("split"),
                "accuracy": float(value),
            }
        )
        return value

    utils.eval_task_top1 = evaluate
    original_build = OpenClipClassifier.build

    def build(build_cfg):
        with meter.phase("model_loading"):
            clf = original_build(build_cfg)
        return clf

    OpenClipClassifier.build = staticmethod(build)
    original_text = OpenClipClassifier.build_zeroshot_text_features

    def text_features(self, *args, **kwargs):
        # Keep evaluator text-head caches inside this run as well.
        if kwargs.get("cache_dir") is not None:
            kwargs["cache_dir"] = str(Path(cfg["logging"]["local_log_dir"]).parent / "text_cache")
        with meter.phase("text_head"):
            return original_text(self, *args, **kwargs)

    OpenClipClassifier.build_zeroshot_text_features = text_features
    # Keep benchmark LP's support examples identical to KD/REFRACT (including MNIST/SVHN caps).
    if method == "linear_probe":
        from torch.utils.data import DataLoader, Subset

        from merge_and_rebase.eval import vision_linear_probe as lp
        from merge_and_rebase.eval.kd_rebase import _vision_support_indices

        def support_loader(_loader, dataset, **kwargs):
            indices = _vision_support_indices(dataset, cfg["tasks"], cfg["linear_probe_shots_per_class"], cfg["seed"])
            return DataLoader(
                Subset(dataset, indices),
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg["num_workers"],
                pin_memory=True,
            )

        lp.build_grad_dataloader = support_loader
        from merge_and_rebase.eval import linear_probe

        original_accuracy = linear_probe._probe_accuracy

        def accuracy(*args, **kwargs):
            value = original_accuracy(*args, **kwargs)
            meter.curve.append(
                {
                    "elapsed_seconds": time.perf_counter() - meter.started,
                    "training_seconds": sum(r["seconds"] for r in meter.records if r["name"] == "training"),
                    "split": "val",
                    "accuracy": float(value),
                }
            )
            return value

        linear_probe._probe_accuracy = accuracy
        # This run uses validation each epoch, and test only before/after training.
        # Rewrite the caller's dict via a wrapper; avoid changing repository source.
        original_probe = lp.train_zeroshot_head_probe

        def probe(*args, **kwargs):
            kwargs["eval_loaders"] = {"val": kwargs["eval_loaders"]["val"]}
            kwargs["log_every"] = 1
            return original_probe(*args, **kwargs)

        lp.train_zeroshot_head_probe = probe


def forward_benchmark(torch, meter, cfg, iterations):
    from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

    clf = OpenClipClassifier.build(
        OpenClipBuildConfig(
            model_name=cfg["source_clip_model"],
            pretrained=cfg["source_clip_pretrained"],
            device=cfg["device"],
            dtype=cfg["dtype"],
        )
    )
    visual = clf.model.visual.eval()
    size = visual.image_size
    size = (size, size) if isinstance(size, int) else tuple(size)
    parameter = next(visual.parameters())
    images = torch.randn(cfg["batch_size"], 3, *size, device=cfg["device"], dtype=parameter.dtype)
    with torch.no_grad():
        with meter.phase("source_forward_benchmark"):
            for _ in range(iterations + meter.warmup):
                with meter.batch("source_forward", len(images)):
                    visual(images)
    return {"input_shape": list(images.shape), "synthetic_input": True, "includes_preprocessing_or_transfer": False}


def worker(args):
    import numpy as np
    import torch

    cfg = json.loads(Path(args.worker_config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if str(cfg["device"]).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; run on a GPU node or pass --device cpu for a smoke test."
            )
        torch.cuda.set_device(torch.device(cfg["device"]))
        torch.cuda.init()
    meter = Meter(torch, warmup=args.warmup, device=cfg["device"])
    finder = InstrumentFinder(meter)
    sys.meta_path.insert(0, finder)
    result, error = {}, None
    started = time.perf_counter()
    try:
        patch_runtime(meter, cfg, args.worker_method)
        with meter.phase("pipeline"):
            if args.worker_method == "source_forward":
                result["forward_protocol"] = forward_benchmark(torch, meter, cfg, args.forward_batches)
            else:
                if args.worker_method == "linear_probe":
                    from merge_and_rebase.eval.vision_linear_probe import main
                elif args.worker_method == "kd":
                    from merge_and_rebase.eval.kd_rebase import main
                else:
                    from merge_and_rebase.eval.vision_rebase import main
                sys.argv = ["cost-benchmark"] + (["vision"] if args.worker_method == "kd" else [])
                sys.argv += ["--config", args.worker_config]
                main()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result.update(meter.finish())
        seconds = time.perf_counter() - started
        result.update(
            {
                "method": args.worker_method,
                "status": "error" if error else "success",
                "error": error,
                "pipeline_seconds": result.get("phases", {}).get("pipeline", {}).get("inclusive_seconds"),
                "worker_seconds": seconds,
                "gpu_hours": seconds / 3600 if meter.cuda else 0.0,
                "resolved_config": cfg,
                "instrumentation": finder.counts,
                "hardware": {
                    "gpu": torch.cuda.get_device_name() if meter.cuda else None,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cpu_threads": torch.get_num_threads(),
                },
            }
        )
        Path(args.worker_output).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


def make_config(base, method, task, seed, args, out, cache):
    cfg = json.loads(json.dumps(base))
    cfg.update(
        tasks=task,
        seed=seed,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_search=False,
        linear_probe_head=False,
    )
    cfg["logging"] = {"use_wandb": False, "mode": "offline", "local_log_dir": str(out / "run_logs")}
    cfg["method"] = "steer"
    params = cfg.setdefault("method_params", {})
    params.update(few_shot=args.shots, seed=seed, feature_cache_dir=str(cache), force_recompute_features=False)
    if method == "linear_probe":
        cfg.update(
            clip_model=cfg["target_clip_model"],
            clip_pretrained=cfg["target_clip_pretrained"],
            linear_probe_shots_per_class=args.shots,
            linear_probe_epochs=args.epochs,
            linear_probe_lr=args.probe_lr,
            linear_probe_log_every=1,
            split="test",
        )
    elif method == "kd":
        cfg.setdefault("kd", {}).update(
            few_shot=args.shots,
            seed=seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.kd_lr,
            amp=None,
            epoch_selection="last",
        )
    return cfg


def accuracy_from_summary(path):
    files = sorted((path / "run_logs").rglob("*.json"))
    for file in reversed(files):
        obj = json.loads(file.read_text())
        if "avg_probed_accuracy" in obj:
            return obj["avg_probed_accuracy"]
        if "test_results" in obj:
            return obj["test_results"].get("avg_rebased")
    return None


def launch(args):
    base = json.loads(Path(args.config).read_text())
    if not all(
        k in base
        for k in ("source_clip_model", "source_clip_pretrained", "target_clip_model", "target_clip_pretrained")
    ):
        raise ValueError("Use a vision REFRACT/STEER config with source and target model keys.")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    methods = args.methods.split(",")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate methods are not allowed.")
    if "refract_warm" in methods and "refract_cold" not in methods:
        raise ValueError("refract_warm requires refract_cold in the same experiment.")
    for task in args.tasks.split(","):
        for seed in map(int, args.seeds.split(",")):
            for repeat in range(args.repeats):
                cache = out / task / f"seed{seed}" / f"repeat{repeat}" / "feature_cache"
                # Cold must precede warm; other methods are shuffled to reduce order effects.
                order = [m for m in methods if m != "refract_warm"]
                random.Random(seed + repeat).shuffle(order)
                if "refract_warm" in methods:
                    order.insert(order.index("refract_cold") + 1, "refract_warm")
                for label in order:
                    method = "refract" if label.startswith("refract_") else label
                    run = cache.parent / label
                    run.mkdir(parents=True)
                    cfg = make_config(base, method, task, seed, args, run, cache)
                    config_path = run / "config.json"
                    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-method",
                        method,
                        "--worker-config",
                        str(config_path),
                        "--worker-output",
                        str(run / "metrics.json"),
                        "--warmup",
                        str(args.warmup),
                        "--forward-batches",
                        str(args.forward_batches),
                    ]
                    print(f"{task} seed={seed} repeat={repeat} {label}", flush=True)
                    started = time.perf_counter()
                    with (run / "console.log").open("w") as log:
                        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    wall = time.perf_counter() - started
                    if proc.returncode:
                        raise RuntimeError(f"{label} failed; see {run / 'console.log'}")
                    metrics = json.loads((run / "metrics.json").read_text())
                    phases = metrics["phases"]
                    if label == "refract_warm" and phases.get("extraction", {}).get("calls", 0):
                        raise RuntimeError(f"Warm run recomputed features; inspect {run}")
                    counts = {
                        name: sum(v["calls"] for key, v in metrics["batches"].items() if key.startswith(name + "/"))
                        for name in ("train", "extraction", "teacher_forward")
                    }
                    row = {
                        "task": task,
                        "seed": seed,
                        "repeat": repeat,
                        "method": label,
                        "accuracy": accuracy_from_summary(run),
                        "end_to_end_seconds": wall,
                        "allocated_gpu_hours": wall / 3600 if metrics["hardware"]["gpu"] else 0.0,
                        "pipeline_seconds": metrics["pipeline_seconds"],
                        "training_seconds": phases.get("training", {}).get("inclusive_seconds", 0.0),
                        "training_batches": counts["train"],
                        "mean_training_seconds_per_batch": phases.get("training", {}).get("inclusive_seconds", 0.0)
                        / counts["train"]
                        if counts["train"]
                        else None,
                        "extraction_batches": counts["extraction"],
                        "mean_extraction_seconds_per_batch": phases.get("extraction", {}).get("inclusive_seconds", 0.0)
                        / counts["extraction"]
                        if counts["extraction"]
                        else None,
                        "teacher_batches": counts["teacher_forward"],
                        "evaluation_seconds": phases.get("evaluation", {}).get("inclusive_seconds", 0.0),
                        "extraction_seconds": phases.get("extraction", {}).get("inclusive_seconds", 0.0),
                        "teacher_precompute_seconds": phases.get("teacher_precompute", {}).get(
                            "inclusive_seconds", 0.0
                        ),
                        "fitting_and_diagnostics_seconds": phases.get("fitting_and_diagnostics", {}).get(
                            "inclusive_seconds", 0.0
                        ),
                        "fit_kernel_seconds": sum(
                            phases.get(k, {}).get("exclusive_seconds", 0.0) for k in ("stage1_targets", "stage2_fit")
                        )
                        + phases.get("stage1_fit", {}).get("exclusive_seconds", 0.0),
                        "peak_vram_allocated_bytes": metrics["peak_allocated_bytes"],
                        "peak_vram_reserved_bytes": metrics["peak_reserved_bytes"],
                        "peak_rss_tree_bytes": metrics["peak_rss_tree_bytes"],
                        "feature_cache_bytes": sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
                        if method == "refract"
                        else 0,
                    }
                    rows.append(row)
                    with (out / "runs.csv").open("w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["seed"], row["method"])].append(row)
    per_seed = [
        {
            "task": task,
            "seed": seed,
            "method": method,
            "metrics": {
                k: stats([r[k] for r in items if r[k] is not None])
                for k in items[0]
                if k not in {"task", "method", "seed", "repeat"}
            },
        }
        for (task, seed, method), items in grouped.items()
    ]
    (out / "per_seed.json").write_text(json.dumps(per_seed, indent=2) + "\n")
    grouped_seeds = defaultdict(list)
    for row in per_seed:
        grouped_seeds[(row["task"], row["method"])].append(row)
    aggregate = [
        {
            "task": task,
            "method": method,
            "metrics": {
                k: stats([r["metrics"][k]["mean"] for r in items if r["metrics"][k]["n"]]) for k in items[0]["metrics"]
            },
        }
        for (task, method), items in grouped_seeds.items()
    ]
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    paired = []
    lookup = {(r["task"], r["seed"], r["repeat"], r["method"]): r for r in rows}
    for kd in (r for r in rows if r["method"] == "kd"):
        for label in ("refract_cold", "refract_warm"):
            ref = lookup.get((kd["task"], kd["seed"], kd["repeat"], label))
            if ref:
                paired.append(
                    {
                        "task": kd["task"],
                        "seed": kd["seed"],
                        "repeat": kd["repeat"],
                        "denominator": label,
                        "kd_over_refract": {
                            k: kd[k] / ref[k] if ref[k] else None
                            for k in (
                                "end_to_end_seconds",
                                "pipeline_seconds",
                                "peak_vram_allocated_bytes",
                                "peak_vram_reserved_bytes",
                                "peak_rss_tree_bytes",
                            )
                        },
                    }
                )
    (out / "paired_ratios.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(f"Results: {out / 'runs.csv'}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--output")
    p.add_argument("--tasks", default="Cars")
    p.add_argument("--seeds", default="33,42,54")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--kd-lr", type=float, default=1e-5)
    p.add_argument("--methods", default="source_forward,linear_probe,kd,refract_cold,refract_warm")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--forward-batches", type=int, default=30)
    p.add_argument("--worker-method", choices=["source_forward", "linear_probe", "kd", "refract"])
    p.add_argument("--worker-config")
    p.add_argument("--worker-output")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.worker_method:
        worker(args)
    else:
        if not args.config or not args.output:
            p.error("--config and --output are required")
        valid = {"source_forward", "linear_probe", "kd", "refract_cold", "refract_warm"}
        if not set(args.methods.split(",")) <= valid:
            p.error(f"--methods must be a subset of {sorted(valid)}")
        if min(args.repeats, args.shots, args.epochs, args.batch_size, args.forward_batches) <= 0 or args.warmup < 0:
            p.error("counts must be positive; --warmup must be nonnegative")
        launch(args)


if __name__ == "__main__":
    main()
