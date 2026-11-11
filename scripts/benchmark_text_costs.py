#!/usr/bin/env python3
"""Measure source block residuals, target feature extraction and complete ridge fitting."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from merge_and_rebase.data.text_loaders import build_nli_task_data, build_nli_tokenized_loader
from merge_and_rebase.rebase.methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _few_shot,
    _fit_block_ridge,
    _stage1_projection,
)
from merge_and_rebase.rebase.text.adapters import head_linear, neutralize_intermediate_head_layers
from merge_and_rebase.rebase.text.steer_text import (
    _head_as_identity,
    _pooled_features,
    _TextBlockCapture,
    num_residual_blocks,
    text_parameter_blocks,
)
from merge_and_rebase.utils.linearization import LinearizedModule
from scripts.eval_t5_encoder_checkpoints import T5EncoderForSequenceClassification


def timed(device, operation):
    """Synchronized wall time; includes Python dispatch and waits for actual completion."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


class ResidualExtractor:
    """The same forwards/masks as steer_text._collect_linear_split, retaining results on device."""

    def __init__(self, source, tuned):
        self.block_ids, self.num_groups = text_parameter_blocks(source)
        names = [name for name, _ in source.named_parameters()]
        self.linmod = LinearizedModule.from_module(source, copy_module=False, param_names=names)
        self.theta0 = dict(zip(self.linmod.param_names, self.linmod.theta0, strict=True))
        self.tuned = {name: p.detach() for name, p in tuned.named_parameters()}
        if self.tuned.keys() != self.theta0.keys():
            raise ValueError("Source and tuned parameter names differ")
        self.source = source

    @torch.no_grad()
    def __call__(self, batch):
        kwargs = {key: batch[key] for key in ("input_ids", "attention_mask") if key in batch}
        f0 = _pooled_features(self.source, batch, batch["input_ids"].device)
        full = self.linmod.forward(current_params=self.tuned, kwargs=kwargs, output_transform=lambda out: out.logits)
        delta = (full - f0).detach()
        blocks = torch.empty(f0.shape[0], self.num_groups, f0.shape[-1], device=f0.device, dtype=f0.dtype)
        for block_id in range(self.num_groups):
            masked = {
                name: self.tuned[name] if group == block_id else self.theta0[name]
                for name, group in zip(self.linmod.param_names, self.block_ids, strict=True)
            }
            out = self.linmod.forward(current_params=masked, kwargs=kwargs, output_transform=lambda out: out.logits)
            blocks[:, block_id] = out - f0
        return f0.detach(), delta, blocks.detach()


@torch.no_grad()
def target_features(target, batch):
    # One forward produces all intermediate features and the final head-input feature.
    with _TextBlockCapture(target) as capture:
        capture.attention_mask = batch.get("attention_mask")
        out = _pooled_features(target, batch, batch["input_ids"].device)
    blocks = dict(capture.activations)
    blocks[len(blocks)] = out.detach()
    return out.detach(), blocks


def nearest_mean_head(features, labels, num_labels):
    normed = F.normalize(features.double(), dim=-1)
    return torch.stack([F.normalize(normed[labels == c].mean(0), dim=0) for c in range(num_labels)])


def fit_all(data, w_a, w_b, config):
    """CPU float64, matching steer_text.prepare; includes preparation and all fitting stages."""
    start = time.perf_counter()
    f_a, delta, f_b, blocks_a = [data[key].double() for key in ("f_a", "delta", "f_b", "blocks_a")]
    blocks_b = {key: value.double() for key, value in data["blocks_b"].items()}
    selected = torch.arange(f_a.shape[0])
    logit_map, kept_rank, rank = _stage1_projection(
        f_a=f_a,
        delta_a=delta,
        w_a=w_a.double(),
        f_b=f_b,
        w_b=w_b.double(),
        selected=selected,
        regularization=config["stage1_lambda"],
        regularization_mode=config["stage1_regularization_mode"],
    )
    p_b = torch.linalg.pinv(w_b.double())
    targets = blocks_a @ logit_map.T @ p_b.T
    n_source_residual = blocks_a.shape[1] - 1
    n_target_residual = len(blocks_b) - 1
    residual_b = {b: blocks_b[b] for b in range(n_target_residual)}
    grouped = _BLOCK_GROUP_STRATEGIES[config["block_group_strategy"]](residual_b, n_source_residual)
    grouped[n_source_residual] = blocks_b[n_target_residual]
    preparation_s = time.perf_counter() - start
    solve_start = time.perf_counter()
    coefficients = _fit_block_ridge(
        grouped,
        targets,
        selected=selected,
        regularization=config["ridge_lambda"],
        mode=config["block_ridge_mode"],
        rho=config["rho"],
    )
    solver_s = time.perf_counter() - solve_start
    if any(not torch.isfinite(c).all() for c in coefficients):
        raise RuntimeError("Non-finite ridge coefficients")
    return {
        "ridge_total_s": preparation_s + solver_s,
        "ridge_preparation_and_stage1_s": preparation_s,
        "ridge_stage2_solver_s": solver_s,
        "num_ridges": len(coefficients),
        "ridge_input_dimensions": [int(grouped[b].shape[1]) for b in range(len(coefficients))],
        "stage1_rank": rank,
        "stage1_kept_rank": kept_rank,
    }


def collect(extractor, target, batches_a, batches_b, device, *, warmup):
    if len(batches_a) != len(batches_b) or not batches_a:
        raise ValueError("Empty or unaligned loaders")
    # Real batches, separate from measurement. No torch.inference_mode: it disables forward AD.
    for index in range(warmup):
        a = {k: v.to(device) for k, v in batches_a[index % len(batches_a)].items()}
        b = {k: v.to(device) for k, v in batches_b[index % len(batches_b)].items()}
        result, _ = timed(device, lambda a=a: extractor(a))
        del result
        result, _ = timed(device, lambda b=b: target_features(target, b))
        del result, a, b

    chunks = {key: [] for key in ("f_a", "delta", "blocks_a", "f_b")}
    block_chunks = {}
    records = []
    for index, (cpu_a, cpu_b) in enumerate(zip(batches_a, batches_b, strict=True)):
        if not torch.equal(cpu_a["labels"], cpu_b["labels"]):
            raise ValueError("Unaligned labels")
        a = {k: v.to(device) for k, v in cpu_a.items()}
        b = {k: v.to(device) for k, v in cpu_b.items()}
        (f_a, delta, blocks_a), residual_s = timed(device, lambda a=a: extractor(a))
        (f_b, blocks_b), feature_s = timed(device, lambda b=b: target_features(target, b))
        # Copies, correctness checks and dataset assembly are outside the measured regions.
        values = {"f_a": f_a.cpu(), "delta": delta.cpu(), "blocks_a": blocks_a.cpu(), "f_b": f_b.cpu()}
        reconstruction = values["blocks_a"].sum(1)
        torch.testing.assert_close(reconstruction, values["delta"], rtol=1e-3, atol=1e-4)
        for key, value in values.items():
            chunks[key].append(value)
        for key, value in blocks_b.items():
            block_chunks.setdefault(key, []).append(value.cpu())
        records.append(
            {
                "batch": index,
                "n_examples": len(a["input_ids"]),
                "residual_s": residual_s,
                "features_B_s": feature_s,
                "source_tokens": int(cpu_a["attention_mask"].sum()),
                "target_tokens": int(cpu_b["attention_mask"].sum()),
                "source_padded_length": int(cpu_a["input_ids"].shape[1]),
                "target_padded_length": int(cpu_b["input_ids"].shape[1]),
                "decomposition_max_abs_error": float((reconstruction - values["delta"]).abs().max()),
            }
        )
        del f_a, delta, blocks_a, f_b, blocks_b, a, b
        if index % 10 == 0:
            print(f"  measured batch {index + 1}/{len(batches_a)}", flush=True)
    data = {key: torch.cat(values) for key, values in chunks.items()}
    data["blocks_b"] = {key: torch.cat(values) for key, values in block_chunks.items()}
    return data, records


def summarize(rows):
    result = {}
    for pair in dict.fromkeys(row["pair"] for row in rows):
        subset = [row for row in rows if row["pair"] == pair]
        result[pair] = {}
        for metric in ("residual_ms_per_example", "features_B_ms_per_example", "ridge_total_s"):
            values = [row[metric] for row in subset]
            result[pair][metric] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return result


def save_results(output, rows):
    (output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (output / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["| Step | T5 → T5-large | T5 → RoBERTa |", "| :--- | ---: | ---: |"]
    for label, key in (
        ("Residual extraction (ms/example)", "residual_ms_per_example"),
        ("Feature extraction, forward of B (ms/example)", "features_B_ms_per_example"),
        ("Ridge fitting, total (s)", "ridge_total_s"),
    ):
        cells = []
        for pair in ("t5_to_t5_large", "t5_to_roberta"):
            item = summary.get(pair, {}).get(key)
            cells.append(f"{item['mean']:.4f} ± {item['std']:.4f}" if item else "–")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    (output / "table.md").write_text("\n".join(lines) + "\n")


def load_encoder(name, num_labels):
    from transformers import T5Config, T5EncoderModel

    config = T5Config.from_pretrained(name, num_labels=num_labels)
    model = T5EncoderForSequenceClassification(config)
    # Explicitly load the encoder: no ambiguity from the outer classifier's prefix.
    model.transformer = T5EncoderModel.from_pretrained(name, attn_implementation="eager")
    return model.eval()


def main():
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for key in ("shots", "repeats", "batch_size", "max_candidates", "cpu_threads"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["warmup_batches"] < 1:
        raise ValueError("At least one warm-up batch is required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "hostname": platform.node(),
        "device": str(device),
        "dtype": "fp32",
        "fit_device": "cpu",
        "fit_dtype": "float64",
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_attention": "math SDPA forced by LinearizedModule",
        "target_attention": "eager",
        "tf32": False,
    }
    (args.output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pool = build_nli_task_data(task=config["task"], split="train", max_samples=config["max_candidates"])
    labels = torch.tensor([row.label for row in pool.examples])
    indices = _few_shot(labels, config["shots"], config["seed"])
    support = replace(pool, examples=[pool.examples[i] for i in indices.tolist()])
    labels = labels[indices]
    support_info = {
        "dataset": pool.meta,
        "indices_in_candidate_pool": indices.tolist(),
        "class_counts": torch.bincount(labels).tolist(),
        "examples_sha256": hashlib.sha256(
            json.dumps(
                [[row.premise, row.hypothesis, row.label] for row in support.examples], ensure_ascii=False
            ).encode()
        ).hexdigest(),
    }
    (args.output / "support.json").write_text(json.dumps(support_info, indent=2) + "\n")
    checkpoint = Path(config["source_checkpoint"])
    conversion = json.loads((checkpoint / "conversion.json").read_text())
    if conversion["base_model"] != config["source_base"] or conversion["task"] != config["task"]:
        raise ValueError("Checkpoint conversion does not match source/task")
    if conversion["head_class_ids"] != list(range(len(pool.labels))):
        raise ValueError("This benchmark requires contiguous head class IDs")
    (args.output / "source_conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    source_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    def batches(tokenizer):
        return list(
            build_nli_tokenized_loader(
                task_data=support,
                tokenizer=tokenizer,
                batch_size=config["batch_size"],
                num_workers=0,
                max_length=config["max_length"],
                shuffle=False,
                head_class_ids=list(range(len(pool.labels))),
            ).loader
        )

    batches_a = batches(source_tokenizer)
    source = load_encoder(config["source_base"], len(pool.labels)).to(device).eval()
    tuned = T5EncoderForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    w_a = head_linear(tuned)[1].weight.detach().cpu().double()
    rows = []
    with _head_as_identity(source), _head_as_identity(tuned):
        extractor = ResidualExtractor(source, tuned)
        metadata["source_residual_blocks"] = extractor.num_groups - 1
        metadata["source_parameter_groups"] = extractor.num_groups
        metadata["jvps_per_batch"] = extractor.num_groups + 1
        (args.output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for target_spec in config["targets"]:
            pair = target_spec["pair"]
            print(f"=== {pair} ===", flush=True)
            name = target_spec["model"]
            tokenizer_b = AutoTokenizer.from_pretrained(name)
            if target_spec["kind"] == "t5_encoder":
                target = load_encoder(name, len(pool.labels))
            else:
                target = AutoModelForSequenceClassification.from_pretrained(
                    name, num_labels=len(pool.labels), attn_implementation="eager"
                )
                neutralize_intermediate_head_layers(target)
            target = target.to(device).eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
            batches_b = batches(tokenizer_b)
            with _head_as_identity(target):
                for repeat in range(config["repeats"]):
                    print(f"{pair}: repeat {repeat + 1}/{config['repeats']}", flush=True)
                    data, records = collect(
                        extractor, target, batches_a, batches_b, device, warmup=config["warmup_batches"]
                    )
                    # Head construction is an explicit setup step, outside ridge fitting.
                    w_b = nearest_mean_head(data["f_b"], labels, len(pool.labels))
                    if repeat == 0:
                        fit_all(data, w_a, w_b, config)  # Warm up complete fitting on the actual support set.
                    fit = fit_all(data, w_a, w_b, config)
                    n = sum(record["n_examples"] for record in records)
                    row = {
                        "pair": pair,
                        "repeat": repeat,
                        "n_examples": n,
                        "batch_size": config["batch_size"],
                        "jvps_per_batch": extractor.num_groups + 1,
                        "target_residual_blocks": num_residual_blocks(target),
                        "residual_ms_per_example": 1000 * sum(r["residual_s"] for r in records) / n,
                        "features_B_ms_per_example": 1000 * sum(r["features_B_s"] for r in records) / n,
                        **fit,
                    }
                    (args.output / f"{pair}_repeat{repeat}_batches.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    rows.append(row)
                    save_results(args.output, rows)
                    print(json.dumps(row), flush=True)
                    del data
            del target, batches_b, tokenizer_b
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print((args.output / "table.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
