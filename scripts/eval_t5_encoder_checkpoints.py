#!/usr/bin/env python
"""Evaluate converted encoder-only T5 NLI checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Config, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5PreTrainedModel

NLI_TASKS = ("snli", "mnli", "sick", "qnli", "rte", "scitail")


class T5EncoderClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out_proj(features)


class T5EncoderForSequenceClassification(T5PreTrainedModel):
    """T5 encoder plus the linear head used by the converted checkpoints."""

    config_class = T5Config
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_unexpected = [r"decoder", r"lm_head"]

    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.classification_head = T5EncoderClassificationHead(config.d_model, config.num_labels)
        self.config.is_encoder_decoder = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classification_head(pooled)
        loss = None if labels is None else nn.functional.cross_entropy(logits, labels.view(-1).long())
        return SequenceClassifierOutput(loss=loss, logits=logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dirs", nargs="+", type=Path)
    parser.add_argument("--tasks", default=",".join(NLI_TASKS))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Any, mask_class: list[int], device: torch.device) -> tuple[int, int]:
    class_ids = torch.tensor(mask_class, dtype=torch.long, device=device)
    correct = 0
    total = 0
    model.eval()
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits = model(**inputs).logits.index_select(-1, class_ids)
        predictions = class_ids[logits.argmax(dim=-1)]
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return correct, total


def write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    from merge_and_rebase.data.text_loaders import build_nli_task_data, build_nli_tokenized_loader

    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(NLI_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; allowed: {list(NLI_TASKS)}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []

    for root in args.checkpoint_dirs:
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        print(f"\n=== {root.name} ===", flush=True)
        for task in tasks:
            checkpoint = root / task
            required = [checkpoint / name for name in ("config.json", "conversion.json", "model.safetensors")]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}; missing: {missing}")

            conversion = json.loads((checkpoint / "conversion.json").read_text(encoding="utf-8"))
            head_class_ids = [int(value) for value in conversion["head_class_ids"]]
            num_labels = int(conversion["num_labels"])
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = T5EncoderForSequenceClassification.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                local_files_only=True,
                torch_dtype=torch.float32,
            ).to(device)

            task_data = build_nli_task_data(task=task, split=args.split, max_samples=args.max_samples)
            tokenized = build_nli_tokenized_loader(
                task_data=task_data,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_length=args.max_length,
                shuffle=False,
                head_class_ids=head_class_ids,
            )
            correct, total = evaluate(model, tokenized.loader, tokenized.mask_class, device)
            row = {
                "checkpoint_group": root.name,
                "task": task,
                "split": args.split,
                "hf_split": task_data.meta["hf_split"],
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "checkpoint": str(checkpoint),
            }
            rows.append(row)
            write_results(rows, args.output_dir)
            print(f"{task:>7}: {row['accuracy']:.6f} ({correct}/{total})", flush=True)

            del model, tokenizer, tokenized
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nResults written to {args.output_dir / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
