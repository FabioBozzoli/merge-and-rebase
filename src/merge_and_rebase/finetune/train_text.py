from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import signal
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # type: ignore
from tqdm import tqdm

from merge_and_rebase.cli_args import add_logging_args, build_logging_overrides
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from merge_and_rebase.utils.helpers import parse_csv

from ..data.text_loaders import (
    NLI_TASKS,
    CausalTokenizedData,
    ResumableRandomSampler,
    build_causal_task_data,
    build_causal_tokenized_loader,
    build_nli_task_data,
    build_nli_tokenized_loader,
    default_head_class_ids_for_task,
    split_causal_task_data,
)
from ..models.text_lm import TextBuildConfig, TextLM
from ..utils.distributed import (
    DistInfo,
    all_reduce_sum_,
    broadcast_flag,
    init_distributed,
    reduce_gradients_,
    shutdown_distributed,
)
from .forward_mode import apply_training_forward_mode, resolve_training_forward_mode
from .schedulers import build_lr_scheduler

NLI_SUITES: dict[str, tuple[str, ...]] = {
    "nli6": tuple(NLI_TASKS),
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _set_seed(seed: int, *, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML config must be a mapping at top-level.")
        return cfg

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON config must be an object at top-level.")
        return cfg

    raise ValueError(f"Unsupported config extension: {p.suffix} (use .yaml/.yml or .json)")


def _get_common_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    common = cfg.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("config['common'] must be a dict.")
    return common


def _get_dataset_override(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    ds = cfg.get("datasets", {})
    if ds is None:
        return {}
    if not isinstance(ds, dict):
        raise ValueError("config['datasets'] must be a dict mapping task -> overrides.")
    ov = ds.get(task, {})
    if ov is None:
        return {}
    if not isinstance(ov, dict):
        raise ValueError(f"config['datasets']['{task}'] must be a dict.")
    return ov


def _resolve_tasks_from_cfg(cfg: dict[str, Any]) -> list[str] | None:
    order = cfg.get("datasets_order", None)
    if order is None:
        return None
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise ValueError("config['datasets_order'] must be a list[str].")
    return [str(x).strip().lower() for x in order]


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _safe_model_tag(model_name_or_path: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "__", str(model_name_or_path).strip())
    return cleaned.strip("_") or "model"


def _canonical_param_name(name: str) -> str:
    out = str(name)
    if out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    out = out.replace(".modules_to_save.default", "")
    return out


def _detect_head_roots(model: nn.Module) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for n, p in model.named_parameters():
        if not p.requires_grad and p.numel() == 0:
            continue
        cn = _canonical_param_name(n)
        root = cn.split(".")[0] if "." in cn else cn
        if root in {"score", "classifier", "classification_head"} and root not in seen:
            seen.add(root)
            roots.append(root)
    if roots:
        # Stable preferred order.
        order = {"score": 0, "classifier": 1, "classification_head": 2}
        roots.sort(key=lambda x: order.get(x, 99))
    return roots


def _extract_task_head(model: nn.Module) -> dict[str, torch.Tensor]:
    roots = _detect_head_roots(model)
    named = list(model.named_parameters())

    if roots:
        root = roots[0]
        out: dict[str, torch.Tensor] = {}
        for n, p in named:
            cn = _canonical_param_name(n)
            if cn == root or cn.startswith(root + "."):
                out[cn] = p.detach().cpu().clone()
        if out:
            return out

    # Fallback: pick the smallest matrix with out-dim == num_labels and related bias.
    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0))
    matrix_candidates: list[tuple[str, torch.Tensor]] = []
    if num_labels > 0:
        for n, p in named:
            cn = _canonical_param_name(n)
            if p.ndim == 2 and int(p.shape[0]) == num_labels:
                matrix_candidates.append((cn, p))
    if not matrix_candidates:
        raise RuntimeError("Unable to extract task head: no known classification-head parameters found.")

    matrix_candidates.sort(key=lambda kv: int(kv[1].numel()))
    weight_name, weight_param = matrix_candidates[0]
    out = {weight_name: weight_param.detach().cpu().clone()}

    bias_name = weight_name.rsplit(".", 1)[0] + ".bias" if "." in weight_name else "bias"
    for n, p in named:
        cn = _canonical_param_name(n)
        if cn == bias_name:
            out[cn] = p.detach().cpu().clone()
            break
    return out


def _resolve_head_class_ids(
    task: str,
    *,
    task_num_labels: int,
    head_num_labels: int,
    task_cfg: dict[str, Any],
) -> list[int]:
    explicit = _get(task_cfg, "data.head_class_ids", None)
    if explicit is not None:
        if not isinstance(explicit, list) or not all(isinstance(x, int) for x in explicit):
            raise ValueError(f"[{task}] data.head_class_ids must be a list[int] when provided.")
        out = [int(x) for x in explicit]
        if len(out) != int(task_num_labels):
            raise ValueError(
                f"[{task}] data.head_class_ids length mismatch. "
                f"got={len(out)} expected={int(task_num_labels)}"
            )
        return out

    mask_class = _get(task_cfg, "data.mask_class", None)
    if mask_class is not None:
        masked = int(mask_class)
        if masked < 0 or masked >= int(head_num_labels):
            raise ValueError(f"[{task}] data.mask_class={masked} is out of range for num_labels={head_num_labels}.")
        out = [i for i in range(int(head_num_labels)) if i != masked]
        if len(out) != int(task_num_labels):
            raise ValueError(
                f"[{task}] data.mask_class={masked} yields {len(out)} classes, "
                f"expected {int(task_num_labels)}."
            )
        return out

    if int(head_num_labels) == int(task_num_labels):
        return list(range(int(task_num_labels)))

    # Common 3-way shared head conventions used by NLI tasks.
    t = str(task).strip().lower()
    if int(task_num_labels) == 2 and int(head_num_labels) >= 3:
        if t in {"qnli", "rte"}:
            return [0, 2]
        if t == "scitail":
            return [0, 1]

    out = default_head_class_ids_for_task(task, num_labels=int(head_num_labels))
    if len(out) == int(task_num_labels):
        return out
    raise ValueError(
        f"[{task}] could not infer valid head_class_ids: "
        f"task_num_labels={int(task_num_labels)}, head_num_labels={int(head_num_labels)}. "
        "Set data.head_class_ids or data.mask_class explicitly."
    )


def _build_causal_task_loaders(
    *,
    task: str,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    task_cfg: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> tuple[CausalTokenizedData, CausalTokenizedData, CausalTokenizedData, dict[str, Any]]:
    hf_path = _get(task_cfg, "data.hf_path", None)
    if not isinstance(hf_path, str) or not hf_path.strip():
        raise ValueError(f"[{task}] data.hf_path is required for backbone.model_kind='causal_lm'.")
    prompt_fields = _get(task_cfg, "data.prompt_fields", None)
    if not isinstance(prompt_fields, str) or not prompt_fields.strip():
        raise ValueError(f"[{task}] data.prompt_fields is required (e.g. 'query' or 'instruction+input').")
    response_field = _get(task_cfg, "data.response_field", None)
    if not isinstance(response_field, str) or not response_field.strip():
        raise ValueError(f"[{task}] data.response_field is required.")

    hf_config = _get(task_cfg, "data.hf_config", None)
    response_index_field = _get(task_cfg, "data.response_index_field", None)
    common = {
        "task": task,
        "hf_path": str(hf_path),
        "hf_config": None if hf_config is None else str(hf_config),
        "prompt_fields": str(prompt_fields),
        "response_field": str(response_field),
        "response_index_field": None if response_index_field is None else str(response_index_field),
    }

    train_data = build_causal_task_data(
        split=str(_get(task_cfg, "data.split", "train")),
        max_samples=_get(task_cfg, "data.max_train_samples", None),
        **common,
    )
    val_split = _get(task_cfg, "data.val_split", None)
    if isinstance(val_split, str) and val_split.strip():
        val_data = build_causal_task_data(
            split=str(val_split),
            max_samples=_get(task_cfg, "data.max_val_samples", None),
            **common,
        )
    else:
        train_data, val_data = split_causal_task_data(
            train_data,
            val_fraction=float(_get(task_cfg, "data.val_fraction", 0.02)),
        )

    train_loader = build_causal_tokenized_loader(
        task_data=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=True,
        seed=int(_get(task_cfg, "seed", 42)),
        rank=int(rank),
        world_size=int(world_size),
    )
    val_loader = build_causal_tokenized_loader(
        task_data=val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=False,
    )

    meta = {
        "model_kind": "causal_lm",
        "train": train_loader.meta,
        "validation": val_loader.meta,
        # No held-out test split: SFT sets here ship train only, and the real
        # report card for a source model A is lm_eval downstream, not a
        # token-level score on a second carve of the same distribution.
        "test": "alias:validation",
    }
    # val is returned in the test slot too; the causal branch of the training
    # loop only ever iterates the val loader.
    return train_loader, val_loader, val_loader, meta


def _build_task_loaders(
    *,
    task: str,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    head_num_labels: int,
    task_cfg: dict[str, Any],
    model_kind: str = "sequence_classification",
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    if str(model_kind) == "causal_lm":
        return _build_causal_task_loaders(
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            task_cfg=task_cfg,
            rank=rank,
            world_size=world_size,
        )

    max_train_samples = _get(task_cfg, "data.max_train_samples", None)
    max_val_samples = _get(task_cfg, "data.max_val_samples", None)
    max_test_samples = _get(task_cfg, "data.max_test_samples", None)

    train_data = build_nli_task_data(task=task, split="train", max_samples=max_train_samples)
    val_data = build_nli_task_data(task=task, split="validation", max_samples=max_val_samples)
    test_data = build_nli_task_data(task=task, split="test", max_samples=max_test_samples)

    head_class_ids = _resolve_head_class_ids(
        task,
        task_num_labels=len(train_data.labels),
        head_num_labels=head_num_labels,
        task_cfg=task_cfg,
    )
    if len(head_class_ids) != len(train_data.labels):
        raise ValueError(
            f"[{task}] head_class_ids length mismatch. got={len(head_class_ids)} expected={len(train_data.labels)}"
        )

    train_loader = build_nli_tokenized_loader(
        task_data=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=True,
        head_class_ids=head_class_ids,
    )
    val_loader = build_nli_tokenized_loader(
        task_data=val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=False,
        head_class_ids=head_class_ids,
    )
    test_loader = build_nli_tokenized_loader(
        task_data=test_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=False,
        head_class_ids=head_class_ids,
    )

    meta = {
        "train": train_loader.meta,
        "validation": val_loader.meta,
        "test": test_loader.meta,
        "labels": list(train_data.labels),
        "label_texts": list(train_data.label_texts),
        "head_class_ids": list(head_class_ids),
    }
    return train_loader, val_loader, test_loader, meta


def _optimizer_from_name(params, name: str, lr: float, weight_decay: float) -> optim.Optimizer:
    opt = str(name).strip().lower()
    if opt == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _configure_text_strategy(
    *,
    model: nn.Module,
    strategy: str,
    strategy_cfg: dict[str, Any] | None,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    warmup_length: int,
    scheduler_name: str = "cosine",
    steps: int,
    device: torch.device,
    model_kind: str = "sequence_classification",
) -> tuple[nn.Module, optim.Optimizer, Any, dict[str, int], dict[str, Any]]:
    cfg = dict(strategy_cfg or {})
    name = str(strategy).strip().lower()
    is_causal = str(model_kind) == "causal_lm"
    peft_cfg_out: dict[str, Any] = {}

    if is_causal and name != "peft_lora":
        raise ValueError(
            f"model_kind='causal_lm' supports strategy.name='peft_lora' only (got '{name}'). "
            "'full' has no memory headroom under linearized_ntk and 'linear_probe' needs a "
            "classification head this model does not have."
        )

    if name == "full":
        for p in model.parameters():
            p.requires_grad = True

    elif name == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        roots = _detect_head_roots(model)
        if not roots:
            raise RuntimeError("linear_probe requested, but no classifier head root was found.")
        active_root = roots[0]
        for n, p in model.named_parameters():
            cn = _canonical_param_name(n)
            if cn == active_root or cn.startswith(active_root + "."):
                p.requires_grad = True

    elif name == "peft_lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:  # pragma: no cover - env dependent
            raise ImportError("PEFT LoRA strategy requires `peft` to be installed.") from e

        peft_cfg = cfg.get("peft", {}) if isinstance(cfg, dict) else {}
        if not isinstance(peft_cfg, dict):
            raise ValueError("strategy.peft must be a dict when using strategy.name='peft_lora'.")

        target_modules = peft_cfg.get("target_modules", None)
        if not isinstance(target_modules, list) or not all(isinstance(x, str) for x in target_modules):
            raise ValueError("strategy.peft.target_modules must be a list[str].")

        roots = _detect_head_roots(model)
        modules_to_save = peft_cfg.get("modules_to_save", None)
        if modules_to_save is None:
            modules_to_save = roots if roots else None
        if modules_to_save is not None and (
            not isinstance(modules_to_save, list) or not all(isinstance(x, str) for x in modules_to_save)
        ):
            raise ValueError("strategy.peft.modules_to_save must be a list[str] when provided.")

        task_type = TaskType.CAUSAL_LM if is_causal else TaskType.SEQ_CLS
        lora_cfg = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=int(peft_cfg.get("r", 16)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
            target_modules=[str(x) for x in target_modules],
            bias=str(peft_cfg.get("bias", "none")),
            modules_to_save=[str(x) for x in modules_to_save] if modules_to_save else None,
            # rsLoRA: scale alpha/sqrt(r) instead of alpha/r. Downstream delta-W
            # reconstruction (merge.subspaces.core_space, eval.llm_merge) reads the same key.
            use_rslora=bool(peft_cfg.get("use_rslora", False)),
        )
        model = get_peft_model(model, lora_cfg)
        peft_cfg_out = {
            "task_type": str(task_type).split(".")[-1],
            "inference_mode": False,
            "r": int(peft_cfg.get("r", 16)),
            "lora_alpha": int(peft_cfg.get("lora_alpha", 16)),
            "lora_dropout": float(peft_cfg.get("lora_dropout", 0.0)),
            "target_modules": [str(x) for x in target_modules],
            "bias": str(peft_cfg.get("bias", "none")),
            "modules_to_save": [str(x) for x in modules_to_save] if modules_to_save else [],
            "use_rslora": bool(peft_cfg.get("use_rslora", False)),
        }

    else:
        raise ValueError("strategy.name must be one of: full, linear_probe, peft_lora")

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(f"Strategy '{name}' produced zero trainable parameters.")

    opt = _optimizer_from_name(trainable_params, optimizer_name, lr, weight_decay)
    scheduler = build_lr_scheduler(
        opt,
        name=scheduler_name,
        base_lrs=lr,
        warmup_length=warmup_length,
        steps=steps,
    )

    info: dict[str, int] = {
        "trainable_params": int(sum(p.numel() for p in trainable_params)),
    }
    if name == "peft_lora":
        info["lora_params"] = int(
            sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower())
        )
    info["scheduler_name"] = scheduler_name

    return model, opt, scheduler, info, peft_cfg_out


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted next-token CE over unmasked (non -100) positions only."""
    return nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _causal_lm_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Same CE, summed instead of averaged, with the supervised-token count.

    The training loop accumulates summed-loss gradients and divides once per
    optimizer step by the window's total token count (all-reduced under data
    parallelism). That makes the step an exact token-weighted mean, independent
    of how rows are grouped into micro-batches or spread across ranks -- a
    per-micro-batch mean would silently up-weight short rows.
    """
    flat_labels = labels[:, 1:].reshape(-1)
    loss_sum = nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, int((flat_labels != -100).sum().item())


@torch.no_grad()
def _eval_causal(model: nn.Module, loader, device: str, dist_info: DistInfo | None = None) -> dict[str, float]:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)
    model.eval()

    info = dist_info or DistInfo()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for i, batch in enumerate(loader):
        # Each rank evaluates its stride of the val set; the sums are reduced below,
        # so all ranks end up with the same metrics and take the same stop decision.
        if info.enabled and i % info.world_size != info.rank:
            continue
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev, non_blocking=True)
        labels = batch["labels"].to(dev, non_blocking=True).long()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        flat_logits = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
        flat_labels = labels[:, 1:].reshape(-1)
        supervised = flat_labels != -100
        n_tokens = int(supervised.sum().item())
        if n_tokens == 0:
            continue

        total_loss += float(
            nn.functional.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction="sum"
            ).item()
        )
        total_tokens += n_tokens
        correct += int((flat_logits.argmax(dim=-1) == flat_labels)[supervised].sum().item())

    if info.enabled:
        totals = torch.tensor([total_loss, float(total_tokens), float(correct)], dtype=torch.float64, device=dev)
        all_reduce_sum_(totals, info)
        total_loss, total_tokens, correct = float(totals[0]), int(totals[1].item()), int(totals[2].item())

    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan"), "val_token_acc": float("nan")}
    loss = total_loss / total_tokens
    return {
        "val_loss": float(loss),
        "val_ppl": float(math.exp(min(loss, 80.0))),
        "val_token_acc": float(correct / total_tokens),
    }


@torch.no_grad()
def _top1(model: nn.Module, loader, device: str) -> float:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)
    model.eval()

    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev, non_blocking=True)
        labels = batch["labels"].to(dev, non_blocking=True).long()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    return float(correct / max(1, total))


def _save_peft_text_adapter(
    *,
    model: nn.Module,
    tokenizer: Any,
    task_dir: Path,
    strategy: str,
    suffix: str | None,
    peft_cfg: dict[str, Any] | None,
    build_cfg: TextBuildConfig,
) -> dict[str, Any]:
    if not hasattr(model, "save_pretrained"):
        raise ValueError("save_format='peft' expects a PEFT-wrapped model with .save_pretrained().")

    adapter_name = f"{strategy}_adapter" if suffix is None else f"{strategy}_{suffix}_adapter"
    adapter_dir = task_dir / adapter_name
    _ensure_dir(adapter_dir)
    model.save_pretrained(adapter_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(adapter_dir)

    meta = {
        "format": "peft",
        "peft_target": "text",
        "peft_adapter_dir": str(adapter_dir),
        "peft_cfg": peft_cfg if peft_cfg is not None else {},
        "backbone": {
            "kind": "hf_text",
            "model_name_or_path": build_cfg.model_name_or_path,
            "model_arch": build_cfg.model_arch,
            "model_kind": build_cfg.model_kind,
            "dtype": build_cfg.dtype,
        },
    }
    _save_json(adapter_dir / "merge_and_rebase_meta.json", meta)
    return meta


def _export_hf_merged_model(
    *,
    model: nn.Module,
    tokenizer: Any,
    out_dir: Path,
    trainable_state: dict[str, torch.Tensor],
    build_cfg: TextBuildConfig,
    forward_mode: str,
    peft_cfg: dict[str, Any] | None,
    data_meta: dict[str, Any],
) -> dict[str, Any]:
    """Merge LoRA into the base weights and write a plain HF model directory.

    This is what lm_eval's `hf_rebased` loads as `source_finetuned`; it must be
    a directory `AutoModelForCausalLM.from_pretrained` accepts, not an adapter.
    """
    if not hasattr(model, "merge_and_unload"):
        raise ValueError("save_format='hf' expects a PEFT-wrapped model with .merge_and_unload().")

    if trainable_state:
        incompatible = model.load_state_dict(trainable_state, strict=False)
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if unexpected:
            raise RuntimeError(f"Unexpected keys while restoring the best adapter: {unexpected[:5]}")

    merged = model.merge_and_unload()
    _ensure_dir(out_dir)
    merged.save_pretrained(out_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    meta = {
        "format": "hf",
        "forward_mode": forward_mode,
        "peft_cfg": peft_cfg if peft_cfg is not None else {},
        "backbone": {
            "kind": "hf_text",
            "model_name_or_path": build_cfg.model_name_or_path,
            "model_arch": build_cfg.model_arch,
            "model_kind": build_cfg.model_kind,
            "dtype": build_cfg.dtype,
        },
        "data": data_meta,
    }
    if forward_mode == "linearized_ntk":
        meta["linearized_warning"] = (
            "These weights are W0 + dW, but the function that was trained is "
            "f(x; W0) + J(x; W0) . dW. A standard HF forward on this directory is NOT the "
            "trained model. Consume it with feature_regime='linear' (steer_text / theseus), "
            "which applies the same first-order expansion around W0."
        )
    _save_json(out_dir / "merge_and_rebase_meta.json", meta)
    return meta


# What must agree for a resumed run to continue the *same* run: the LR curve is
# a pure function of (scheduler, warmup, total_steps) and the data order of
# (seed, rows, batch_size). lr itself is checked separately so a deliberate
# LR-override continuation stays possible (train.resume_allow_lr_change).
#
# world_size is deliberately NOT here. Rank r's j-th micro-batch of step k is
# shard_r[k*A/N + j] = perm[k*A + j*N + r] (bs=1; blocks of bs rows otherwise),
# so every optimizer step consumes the same A rows for any N dividing A, and the
# summed-then-normalized gradient over them is identical. A sweep on 4 GPUs can
# therefore continue on 8; total_steps still guards the horizon.
_RESUME_STRUCTURAL_KEYS = (
    "model_name_or_path",
    "model_kind",
    "strategy",
    "forward_mode",
    "peft",
    "optimizer_name",
    "scheduler_name",
    "warmup_length",
    "steps_per_epoch",
    "total_steps",
    "accumulate_grad_batches",
    "batch_size",
    "max_length",
    "seed",
    "num_train_rows",
)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_resume_checkpoint(resume_dir: Path, payload: dict[str, Any]) -> Path:
    """step_{N}.pt plus a resume_last.pt symlink. Written via a temp file so a
    kill mid-write never leaves a truncated checkpoint behind the symlink."""
    _ensure_dir(resume_dir)
    path = resume_dir / f"step_{int(payload['global_update_step']):07d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    last = resume_dir / "resume_last.pt"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(path.name)
    return path


def _load_resume_checkpoint(
    path: str | Path,
    *,
    fingerprint: dict[str, Any],
    allow_lr_change: bool,
) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    saved = ckpt.get("fingerprint", {})
    keys = list(_RESUME_STRUCTURAL_KEYS) + ([] if allow_lr_change else ["lr"])
    mismatched = {k: (saved.get(k), fingerprint.get(k)) for k in keys if saved.get(k) != fingerprint.get(k)}
    if mismatched:
        details = ", ".join(f"{k}: checkpoint={a!r} run={b!r}" for k, (a, b) in mismatched.items())
        raise ValueError(f"Resume checkpoint {path} does not match this run ({details}).")
    return ckpt


def train_task(
    *,
    task: str,
    build_cfg: TextBuildConfig,
    strategy: str,
    strategy_cfg: dict[str, Any] | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_length: int,
    scheduler_name: str = "cosine",
    optimizer_name: str,
    clip_grad_norm: float,
    accumulate_grad_batches: int,
    batch_size: int,
    num_workers: int,
    max_length: int,
    head_num_labels: int,
    early_stopping: bool,
    early_stopping_patience: int,
    eval_every_n_steps: int = 0,
    max_steps: int | None = None,
    save_every_n_steps: int = 0,
    resume_from: str | Path | None = None,
    resume_allow_lr_change: bool = False,
    save_best_to_disk: bool = True,
    seed: int,
    deterministic: bool,
    device: str,
    out_dir: Path,
    save_format: str,
    save_last_epoch: bool = False,
    task_cfg: dict[str, Any] | None = None,
    log_every_n_steps: int = 50,
    run_logger: Any | None = None,
    dist_info: DistInfo | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be >= 1.")
    ddp = dist_info or DistInfo()

    def _print(*args: Any) -> None:
        """Rank 0 owns stdout; the other ranks would only duplicate every line."""
        if ddp.is_main:
            print(*args)

    # train.accumulate_grad_batches is the GLOBAL window: N ranks each cover 1/N of it,
    # so the effective batch and the LR schedule's step count do not depend on N.
    accumulate_grad_batches_global = int(accumulate_grad_batches)
    if ddp.enabled:
        if accumulate_grad_batches_global % ddp.world_size != 0:
            raise ValueError(
                f"train.accumulate_grad_batches={accumulate_grad_batches_global} must be divisible by "
                f"world_size={ddp.world_size}."
            )
        accumulate_grad_batches = accumulate_grad_batches_global // ddp.world_size
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be >= 1 (or null for no cap).")

    dev = _device(device)
    _set_seed(seed, deterministic=deterministic)
    forward_mode = resolve_training_forward_mode(strategy_cfg)

    if build_cfg.model_kind not in {"sequence_classification", "causal_lm"}:
        raise ValueError(
            "train_text supports backbone.model_kind in {'sequence_classification', 'causal_lm'}."
        )
    is_causal = build_cfg.model_kind == "causal_lm"

    llm = TextLM.build(build_cfg)
    model = llm.model
    tokenizer = llm.tokenizer
    if is_causal:
        # The KV cache is dead weight for teacher-forced training and its
        # in-place state does not survive functional_call under jvp.
        model.config.use_cache = False

    train_loader, val_loader, test_loader, task_meta = _build_task_loaders(
        task=task,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        head_num_labels=head_num_labels,
        task_cfg=task_cfg or {},
        model_kind=build_cfg.model_kind,
        rank=ddp.rank,
        world_size=ddp.world_size,
    )
    if not is_causal:
        expected_num_labels = int(len(task_meta.get("labels", [])))
        model_num_labels = int(getattr(model.config, "num_labels", expected_num_labels))
        if model_num_labels != expected_num_labels:
            raise ValueError(
                f"[{task}] model head/logits mismatch: model_num_labels={model_num_labels} "
                f"but dataset_num_labels={expected_num_labels}. "
                "Ensure backbone.num_labels matches the dataset label space for this task."
            )

    task_dir = out_dir / _safe_model_tag(build_cfg.model_name_or_path) / task
    _ensure_dir(task_dir)
    if run_logger is not None:
        run_logger.log_event(
            "task_start",
            metrics={},
            context={
                "task": task,
                "strategy": strategy,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "effective_batch_size": int(batch_size * accumulate_grad_batches),
                "task_dir": str(task_dir),
            },
        )

    steps_per_epoch = math.ceil(len(train_loader.loader) / accumulate_grad_batches)
    total_steps = max(1, epochs * steps_per_epoch)
    model, opt, scheduler, trainable_info, peft_cfg_out = _configure_text_strategy(
        model=model,
        strategy=strategy,
        strategy_cfg=strategy_cfg,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        warmup_length=warmup_length,
        scheduler_name=scheduler_name,
        steps=total_steps,
        device=dev,
        model_kind=build_cfg.model_kind,
    )
    # 'lowrank' computes the same function without densifying the adapter (see
    # forward_mode._apply_lowrank_linearized_forward) and is 1.6x faster on CPU, but measured
    # 0.68x on an A100 at seq 1024 (0.97 vs 0.66 s/micro-batch) for ~2 GiB less: the dense
    # dW . x is cheap on tensor cores while the low-rank path adds many small kernels. Dense
    # stays the default; 'lowrank' is kept for CPU runs and as the equivalence reference.
    linearization_impl = str((strategy_cfg or {}).get("linearization_impl", "dense"))
    trainable_info = dict(trainable_info)
    trainable_info["forward_mode"] = forward_mode
    if forward_mode != "standard":
        trainable_info["linearization_impl"] = linearization_impl
    trainable_info.update(
        apply_training_forward_mode(
            model=model,
            forward_mode=forward_mode,
            device=dev,
            output_transform=lambda out: out.logits,
            output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
            impl=linearization_impl,
        )
    )

    # Higher-is-better score: top1 for classification, -val_loss for causal.
    best_val = float("-inf")
    best_state: dict[str, Any] | None = None
    best_head_payload: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    last_epoch = 0
    last_metrics: dict[str, float] = {}
    patience_left = int(early_stopping_patience)

    t_start = time.time()
    global_update_step = 0
    ckpt_stem = str(strategy) if forward_mode == "standard" else f"{strategy}__{forward_mode}"
    hf_export_dir = task_dir / f"{ckpt_stem}_hf"
    best_ckpt_path = task_dir / f"{ckpt_stem}_best_ep.pt"
    resume_dir = task_dir / "resume"
    best_trainable_state: dict[str, torch.Tensor] = {}
    best_step = -1
    last_eval_step = -1
    stop_reason = "completed"
    last_resume_path: Path | None = None

    n_train_batches = len(train_loader.loader)
    train_sampler = getattr(train_loader.loader, "sampler", None)
    if not isinstance(train_sampler, ResumableRandomSampler):
        train_sampler = None
    fingerprint: dict[str, Any] = {
        "model_name_or_path": build_cfg.model_name_or_path,
        "model_kind": build_cfg.model_kind,
        "strategy": strategy,
        "forward_mode": forward_mode,
        "peft": dict(peft_cfg_out),
        "optimizer_name": str(optimizer_name),
        "lr": float(lr),
        "scheduler_name": str(scheduler_name),
        "warmup_length": int(warmup_length),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "accumulate_grad_batches": int(accumulate_grad_batches_global),
        "world_size": int(ddp.world_size),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "seed": int(seed),
        "num_train_rows": int(train_sampler.num_rows) if train_sampler is not None else int(n_train_batches),
    }

    def _trainable_params() -> dict[str, nn.Parameter]:
        return {n: p for n, p in model.named_parameters() if p.requires_grad}

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    loss_scale = float(max_length if is_causal else 1)

    start_epoch = 1
    resume_skip_batches = 0
    resumed_ckpt: dict[str, Any] | None = None
    if resume_from is not None:
        resumed_ckpt = _load_resume_checkpoint(
            resume_from, fingerprint=fingerprint, allow_lr_change=resume_allow_lr_change
        )
        # Loaded strictly after apply_training_forward_mode: the dense impl
        # snapshots its linearization point with lora_B == 0 (the pretrained
        # weights) and would refuse a warm adapter. The lowrank impl has no
        # snapshot, so the order is harmless there.
        params = _trainable_params()
        saved_params = resumed_ckpt["trainable_state"]
        if set(params) != set(saved_params):
            diff = sorted(set(params) ^ set(saved_params))
            raise ValueError(f"Resume checkpoint trainable params differ from the model's (e.g. {diff[:3]}).")
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved_params[n].to(device=p.device, dtype=p.dtype))
        opt.load_state_dict(resumed_ckpt["optimizer"])
        global_update_step = int(resumed_ckpt["global_update_step"])
        start_epoch = int(resumed_ckpt["epoch"])
        saved_world = int(resumed_ckpt.get("world_size", resumed_ckpt["fingerprint"].get("world_size", 1)))
        rows_consumed = int(
            resumed_ckpt.get(
                "rows_consumed_in_epoch",
                int(resumed_ckpt["micro_batches_consumed_in_epoch"]) * int(batch_size) * saved_world,
            )
        )
        rows_per_micro_batch = int(batch_size) * int(ddp.world_size)
        if rows_consumed % rows_per_micro_batch != 0:
            raise ValueError(
                f"Cannot resume {rows_consumed} consumed rows on world_size={ddp.world_size} with "
                f"batch_size={batch_size}: the offset does not split evenly across ranks."
            )
        resume_skip_batches = rows_consumed // rows_per_micro_batch
        if resume_skip_batches >= n_train_batches:
            start_epoch += 1
            resume_skip_batches = 0
        if max_steps is not None and global_update_step >= int(max_steps):
            raise ValueError(
                f"Resuming at step {global_update_step} with max_steps={max_steps}: nothing to do. "
                "Raise or clear train.max_steps for the continuation."
            )

    def _build_checkpoint_payload(
        *,
        epoch_i: int,
        metrics_i: dict[str, float],
        kind: str,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        payload: dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "forward_mode": forward_mode,
            "backbone": {
                "kind": "hf_text",
                "model_name_or_path": build_cfg.model_name_or_path,
                "model_arch": build_cfg.model_arch,
                "model_kind": build_cfg.model_kind,
                "dtype": build_cfg.dtype,
            },
            "num_labels": int(getattr(model.config, "num_labels", head_num_labels)),
            "labels": list(task_meta.get("labels", [])),
            "label_texts": list(task_meta.get("label_texts", [])),
            "head_class_ids": list(task_meta.get("head_class_ids", [])),
            "metrics": {k: float(v) for k, v in metrics_i.items()},
        }
        if kind == "best_ep":
            payload["best_epoch"] = int(epoch_i)
        elif kind == "last_ep":
            payload["last_epoch"] = int(epoch_i)
            payload["best_epoch"] = int(best_epoch)
        else:
            raise ValueError("kind must be 'best_ep' or 'last_ep'")

        head_payload = {} if is_causal else _extract_task_head(model)

        if save_format == "full":
            payload["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload["format"] = "full"
        elif save_format == "head":
            payload["head"] = head_payload
            payload["format"] = "head"
        elif save_format == "peft":
            payload.update(
                _save_peft_text_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    task_dir=task_dir,
                    strategy=ckpt_stem,
                    suffix=kind,
                    peft_cfg=peft_cfg_out,
                    build_cfg=build_cfg,
                )
            )
        elif save_format == "hf":
            payload["format"] = "hf"
            payload["hf_dir"] = str(hf_export_dir)
            if kind == "best_ep":
                # merge_and_unload() is destructive, so the merged export cannot
                # run mid-loop. Stash the LoRA factors (a few MB) and merge once
                # after training instead.
                # ponytail: best epoch only; save_last_epoch still writes the .pt,
                # not a second merged directory.
                best_trainable_state.clear()
                best_trainable_state.update(
                    {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                )
        else:
            raise ValueError("save_format must be 'full', 'head', 'peft', or 'hf'")

        return payload, head_payload

    def _validate_and_track(*, epoch: int, train_loss: float, step: int | None = None) -> bool:
        """Evaluate, keep the best checkpoint, decay patience. True => stop.

        Also callable mid-epoch (train.eval_every_n_steps): a single pass over a
        591K-row corpus produces exactly one epoch-end evaluation, which leaves
        early stopping with nothing to act on.
        """
        nonlocal best_val, best_state, best_head_payload, best_epoch, last_epoch, last_metrics, patience_left
        nonlocal best_step, last_eval_step

        last_eval_step = global_update_step
        if is_causal:
            metrics = _eval_causal(model, val_loader.loader, str(dev), ddp)
            score = -float(metrics["val_loss"])
            desc = (
                f"val_loss={metrics['val_loss']:.4f}  "
                f"val_ppl={metrics['val_ppl']:.3f}  "
                f"val_tok_acc={metrics['val_token_acc']:.4f}"
            )
            log_metrics = {
                f"val/{task}/loss": float(metrics["val_loss"]),
                f"val/{task}/ppl": float(metrics["val_ppl"]),
                f"val/{task}/token_acc": float(metrics["val_token_acc"]),
            }
        else:
            val_acc = _top1(model, val_loader.loader, str(dev))
            test_acc = _top1(model, test_loader.loader, str(dev))
            metrics = {"val_top1": float(val_acc), "test_top1": float(test_acc)}
            score = float(val_acc)
            desc = f"val={val_acc:.4f}  test={test_acc:.4f}"
            log_metrics = {
                f"val/{task}/top1": float(val_acc),
                f"test/{task}/top1": float(test_acc),
            }

        last_epoch = epoch
        last_metrics = dict(metrics)
        stop = False

        if not math.isnan(score) and score > best_val:
            patience_left = int(early_stopping_patience)
            best_epoch = int(epoch)
            best_val = float(score)
            best_step = int(global_update_step)
            best_state, best_head_payload = _build_checkpoint_payload(
                epoch_i=best_epoch,
                metrics_i=metrics,
                kind="best_ep",
            )
            best_state["best_step"] = best_step
            if save_best_to_disk and ddp.is_main:
                # A killed job keeps its best so far. For save_format='hf' the
                # payload is metadata only, so the LoRA factors go alongside.
                torch.save(best_state, best_ckpt_path)
                if save_format == "hf":
                    torch.save(
                        {
                            "best_step": best_step,
                            "best_epoch": best_epoch,
                            "metrics": dict(best_state["metrics"]),
                            "trainable_state": dict(best_trainable_state),
                        },
                        task_dir / f"{ckpt_stem}_best_trainable.pt",
                    )
        else:
            patience_left -= 1
            if early_stopping and patience_left <= 0:
                _print(f"[{task}] Early stopping triggered.")
                stop = True

        where = f"epoch {epoch:03d}/{epochs}" + ("" if step is None else f" step {step}")
        _print(
            f"[{task}] {where}  "
            f"loss={train_loss:.4f}  {desc} "
            f"patience={patience_left}/{early_stopping_patience}"
        )
        if run_logger is not None:
            run_logger.log_event(
                "eval" if step is not None else "epoch_end",
                metrics={
                    f"train/{task}/loss": float(train_loss),
                    f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                    **log_metrics,
                    f"train/{task}/seconds": float(time.time() - t_start),
                },
                step=int(step if step is not None else epoch),
                context={
                    "task": task,
                    "epoch": int(epoch),
                    "update_step": int(global_update_step),
                    "patience_left": int(patience_left),
                },
            )
        model.train()
        return stop

    def _resume_payload(*, epoch_i: int, consumed: int) -> dict[str, Any]:
        return {
            "fingerprint": dict(fingerprint),
            "global_update_step": int(global_update_step),
            "epoch": int(epoch_i),
            "micro_batches_consumed_in_epoch": int(consumed),
            # Global, so a resume on a different number of ranks can find its offset.
            "rows_consumed_in_epoch": int(consumed) * int(batch_size) * int(ddp.world_size),
            "world_size": int(ddp.world_size),
            "trainable_state": {n: p.detach().cpu().clone() for n, p in _trainable_params().items()},
            "optimizer": opt.state_dict(),
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "best_step": int(best_step),
            "patience_left": int(patience_left),
            "last_metrics": dict(last_metrics),
            "best_trainable_state": dict(best_trainable_state),
            # 'full' payloads carry the whole state_dict; only the small ones ride along.
            "best_state": best_state if save_format in {"hf", "peft"} else None,
            "rng": _rng_state(),
        }

    if resumed_ckpt is not None:
        best_val = float(resumed_ckpt["best_val"])
        best_epoch = int(resumed_ckpt["best_epoch"])
        best_step = int(resumed_ckpt["best_step"])
        patience_left = int(resumed_ckpt["patience_left"])
        last_metrics = dict(resumed_ckpt.get("last_metrics", {}))
        best_trainable_state.update(resumed_ckpt.get("best_trainable_state", {}))
        if resumed_ckpt.get("best_state") is not None:
            best_state, best_head_payload = resumed_ckpt["best_state"], {}
        _restore_rng_state(resumed_ckpt["rng"])
        last_eval_step = global_update_step if best_step == global_update_step else -1
        _print(
            f"[{task}] resumed from {resume_from} at step {global_update_step} (saved on {saved_world} "
            f"rank(s), now {ddp.world_size}) "
            f"(epoch {start_epoch}, micro-batch {resume_skip_batches}/{n_train_batches}, "
            f"best_step={best_step}, patience={patience_left}/{early_stopping_patience})"
        )
        del resumed_ckpt

    # Slurm sends SIGUSR1 ahead of the walltime (sbatch --signal=USR1@<secs>):
    # finish the current optimizer step, write a resume checkpoint, exit cleanly.
    preempt = threading.Event()
    prev_sigusr1 = None
    if threading.current_thread() is threading.main_thread():
        prev_sigusr1 = signal.signal(signal.SIGUSR1, lambda *_: preempt.set())

    stop_training = False
    train_loss = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)

        skip = resume_skip_batches if epoch == start_epoch else 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch, start_index=skip * batch_size)
            batches = iter(train_loader.loader)
        else:
            batches = itertools.islice(iter(train_loader.loader), skip, None)

        window_batch_count = 0
        window_size = 1
        window_items = 0
        with tqdm(
            total=n_train_batches,
            initial=skip,
            desc=f"[{task}] Epoch {epoch}/{epochs}",
            unit="batch",
            disable=not ddp.is_main,
        ) as pbar:
            for i, batch in enumerate(batches, start=skip):
                if window_batch_count == 0:
                    remaining = n_train_batches - i
                    window_size = min(accumulate_grad_batches, remaining)

                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True).long()

                if is_causal:
                    # No labels= into the model: HF would run its own shifted CE
                    # *inside* the jvp, and under linearized_ntk the loss must be
                    # computed on f(x;W0) + J.dW after the tangent is added.
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss_sum, n_items = _causal_lm_loss_sum(outputs.logits, labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss_sum = nn.functional.cross_entropy(logits, labels, reduction="sum")
                    n_items = int(labels.numel())
                # Divided by a constant only to keep magnitudes near the per-item loss;
                # the exact 1/total_items factor is applied to the grads at the step.
                (loss_sum / loss_scale).backward()
                window_items += n_items

                window_batch_count += 1
                should_step = window_batch_count == window_size
                if window_batch_count == window_size:
                    # One flat all-reduce per optimizer step, then the single division that
                    # turns the accumulated sums into the token-weighted mean gradient.
                    # Clipping must see the final gradient, so both happen before it.
                    total_items = torch.tensor([float(window_items)], dtype=torch.float64, device=dev)
                    all_reduce_sum_(total_items, ddp)
                    reduce_gradients_(
                        trainable_params_list, ddp, scale=loss_scale / max(1.0, float(total_items.item()))
                    )
                    window_items = 0
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scheduler(global_update_step)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    global_update_step += 1
                    window_batch_count = 0

                # Running average over items (supervised tokens for causal), matching
                # the way the gradient is normalized.
                running_loss += float(loss_sum.item())
                n_seen += n_items

                train_loss = running_loss / max(1, n_seen)
                pbar.update(1)
                # refresh=False: let update() redraw at tqdm's mininterval (TQDM_MININTERVAL
                # in batch jobs) instead of once per batch, which bloats multi-day logs.
                pbar.set_postfix(
                    {"loss": f"{train_loss:.4f}", "lr": f"{opt.param_groups[0]['lr']:.6f}"}, refresh=False
                )
                if (
                    run_logger is not None
                    and log_every_n_steps > 0
                    and global_update_step > 0
                    and global_update_step % log_every_n_steps == 0
                    and should_step
                ):
                    run_logger.log_event(
                        "train_step",
                        metrics={
                            f"train/{task}/loss": float(train_loss),
                            f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                        },
                        step=int(global_update_step),
                        context={
                            "task": task,
                            "epoch": int(epoch),
                        },
                    )

                if not should_step:
                    continue

                if (
                    eval_every_n_steps > 0
                    and global_update_step > 0
                    and global_update_step % eval_every_n_steps == 0
                ):
                    if _validate_and_track(epoch=epoch, train_loss=train_loss, step=global_update_step):
                        stop_reason = "early_stopping"
                        stop_training = True
                        break

                preempted = broadcast_flag(preempt.is_set(), ddp, device=dev)
                hit_cap = max_steps is not None and global_update_step >= int(max_steps)
                if hit_cap and last_eval_step != global_update_step:
                    # The sweep's comparison point: every trial ends with a val number.
                    _validate_and_track(epoch=epoch, train_loss=train_loss, step=global_update_step)
                if (
                    hit_cap
                    or preempted
                    or (save_every_n_steps > 0 and global_update_step % save_every_n_steps == 0)
                ):
                    if ddp.is_main:
                        last_resume_path = _save_resume_checkpoint(
                            resume_dir, _resume_payload(epoch_i=epoch, consumed=i + 1)
                        )
                        _print(f"[{task}] resume checkpoint: {last_resume_path}")
                if hit_cap or preempted:
                    stop_reason = "max_steps" if hit_cap else "preempted"
                    _print(f"[{task}] stopping at step {global_update_step} ({stop_reason}).")
                    stop_training = True
                    break

        if stop_training:
            break
        # Skip the epoch-end eval when a step-level eval just ran on this exact step.
        if last_eval_step != global_update_step and _validate_and_track(epoch=epoch, train_loss=train_loss):
            stop_reason = "early_stopping"
            break

    if prev_sigusr1 is not None:
        signal.signal(signal.SIGUSR1, prev_sigusr1)
    seconds = time.time() - t_start

    if stop_reason == "preempted":
        # Walltime is close: skip the merged HF export (minutes at 3B) and leave
        # the resume checkpoint as the artifact. The best .pt is already on disk
        # when save_best_to_disk is on.
        summary = {
            "task": task,
            "status": "preempted",
            "stop_reason": stop_reason,
            "global_update_step": int(global_update_step),
            "best_step": int(best_step),
            "resume_ckpt_path": str(last_resume_path),
            "seconds": float(seconds),
        }
        if ddp.is_main:
            _save_json(task_dir / f"{ckpt_stem}.json", summary)
        _print(f"[{task}] preempted at step {global_update_step}; resume from {last_resume_path}")
        return summary, {}

    if best_state is None or best_head_payload is None:
        fallback_best_epoch = best_epoch if best_epoch > 0 else last_epoch
        if not last_metrics:
            last_metrics = (
                _eval_causal(model, val_loader.loader, str(dev), ddp)
                if is_causal
                else {"val_top1": float("nan"), "test_top1": _top1(model, test_loader.loader, str(dev))}
            )
        best_state, best_head_payload = _build_checkpoint_payload(
            epoch_i=fallback_best_epoch,
            metrics_i=last_metrics,
            kind="best_ep",
        )

    if ddp.is_main:
        torch.save(best_state, best_ckpt_path)

    last_ckpt_path: Path | None = None
    if save_last_epoch:
        if last_epoch <= 0:
            last_epoch = epochs
        last_state, _ = _build_checkpoint_payload(
            epoch_i=last_epoch,
            metrics_i=last_metrics,
            kind="last_ep",
        )
        last_ckpt_path = task_dir / f"{ckpt_stem}_last_ep.pt"
        if ddp.is_main:
            torch.save(last_state, last_ckpt_path)

    if save_format == "hf" and ddp.is_main:
        best_state["hf_meta"] = _export_hf_merged_model(
            model=model,
            tokenizer=tokenizer,
            out_dir=hf_export_dir,
            trainable_state=best_trainable_state,
            build_cfg=build_cfg,
            forward_mode=forward_mode,
            peft_cfg=peft_cfg_out,
            data_meta=task_meta,
        )
        torch.save(best_state, best_ckpt_path)
        _print(f"[{task}] saved merged HF model: {hf_export_dir}")

    summary = {
        "task": task,
        "strategy": strategy,
        "forward_mode": forward_mode,
        "save_format": save_format,
        "save_last_epoch": bool(save_last_epoch),
        "ckpt_path": str(best_ckpt_path),
        "best_ckpt_path": str(best_ckpt_path),
        "last_ckpt_path": str(last_ckpt_path) if last_ckpt_path is not None else None,
        "metrics": best_state.get("metrics", {}),
        "seconds": float(seconds),
        "trainable": trainable_info,
        "best_epoch": int(best_state.get("best_epoch", -1)),
        "best_step": int(best_step),
        "last_epoch": int(last_epoch),
        "global_update_step": int(global_update_step),
        "total_steps": int(total_steps),
        "status": "finished",
        "stop_reason": stop_reason,
        "resume_from": None if resume_from is None else str(resume_from),
        "resume_ckpt_path": None if last_resume_path is None else str(last_resume_path),
        "last_metrics": {k: float(v) for k, v in last_metrics.items()},
        "meta": task_meta,
        "hparams": {
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "optimizer": str(optimizer_name),
            "warmup_length": int(warmup_length),
            "clip_grad_norm": float(clip_grad_norm),
            "accumulate_grad_batches": int(accumulate_grad_batches_global),
            "accumulate_grad_batches_per_rank": int(accumulate_grad_batches),
            "world_size": int(ddp.world_size),
            "batch_size": int(batch_size),
            "effective_batch_size": int(batch_size * accumulate_grad_batches_global),
            "num_workers": int(num_workers),
            "max_length": int(max_length),
            "eval_every_n_steps": int(eval_every_n_steps),
            "max_steps": None if max_steps is None else int(max_steps),
            "save_every_n_steps": int(save_every_n_steps),
            "scheduler_name": str(scheduler_name),
            "early_stopping": bool(early_stopping),
            "early_stopping_patience": int(early_stopping_patience),
            "seed": int(seed),
        },
    }
    if ddp.is_main:
        _save_json(task_dir / f"{ckpt_stem}.json", summary)

    _print(f"[{task}] saved best: {best_ckpt_path}")
    if last_ckpt_path is not None:
        _print(f"[{task}] saved last: {last_ckpt_path}")
    if run_logger is not None:
        run_logger.log_event(
            "task_end",
            metrics={
                **{
                    f"{k.split('_', 1)[0]}/{task}/{k.split('_', 1)[1]}": float(v)
                    for k, v in summary["metrics"].items()
                    if "_" in k
                },
                f"train/{task}/seconds": float(summary["seconds"]),
            },
            context={
                "task": task,
                "summary": summary,
            },
        )

    return summary, best_head_payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fine-tune text sequence-classification models from a config file (YAML/JSON).")

    g = p.add_argument_group("Config")
    g.add_argument("--text-config", type=str, required=True, help="Path to text config (.yaml/.yml/.json).")

    g = p.add_argument_group("Task selection overrides (optional)")
    g.add_argument("--suite", type=str, default=None, choices=sorted(NLI_SUITES.keys()))
    g.add_argument("--tasks", type=str, default=None, help="Comma-separated task names (overrides suite/order).")

    g = p.add_argument_group("Runtime overrides (optional)")
    g.add_argument("--device", type=str, default=None, help="Override config device, e.g. cuda, cuda:0, cpu.")
    add_logging_args(p)

    return p


def resolve_tasks(args, cfg_file: dict[str, Any]) -> list[str]:
    if args.tasks and args.tasks.strip():
        tasks = [str(x).strip().lower() for x in parse_csv(args.tasks)]
        return tasks
    if args.suite is not None:
        return list(NLI_SUITES[args.suite])

    tasks = _resolve_tasks_from_cfg(cfg_file)
    return tasks if tasks is not None else list(NLI_SUITES["nli6"])


def main() -> None:
    run_logger = None
    ddp = DistInfo()
    try:
        parser = build_parser()
        args = parser.parse_args()

        cfg_file = _load_config(args.text_config)
        common = _get_common_cfg(cfg_file)

        tasks = resolve_tasks(args, cfg_file)

        global_cfg = deepcopy(common)

        backbone_name = str(_get(global_cfg, "backbone.name", "hf_text"))
        if backbone_name != "hf_text":
            raise ValueError(f"Unsupported backbone '{backbone_name}' (only hf_text is supported).")

        model_name_or_path = _get(global_cfg, "backbone.model_name_or_path", None)
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("common.backbone.model_name_or_path is required.")

        model_arch = str(_get(global_cfg, "backbone.model_arch", "auto"))
        model_kind = str(_get(global_cfg, "backbone.model_kind", "sequence_classification"))
        if model_kind not in {"sequence_classification", "causal_lm"}:
            raise ValueError(
                "common.backbone.model_kind must be 'sequence_classification' or 'causal_lm'."
            )
        is_causal = model_kind == "causal_lm"
        if is_causal and _resolve_tasks_from_cfg(cfg_file) is None and not (args.tasks and args.tasks.strip()):
            raise ValueError(
                "model_kind='causal_lm' requires config['datasets_order'] (or --tasks): causal task "
                "names are keys under config['datasets'], not entries in a fixed registry."
            )

        trust_remote_code = bool(_get(global_cfg, "backbone.trust_remote_code", False))
        use_fast_tokenizer = bool(_get(global_cfg, "backbone.use_fast_tokenizer", True))

        device = str(args.device) if args.device is not None else str(_get(global_cfg, "device", "cuda"))
        # One process per GPU (srun --ntasks-per-node=N): join the group and pin this rank's
        # device before anything allocates. A plain single-process run gets world_size == 1.
        ddp = init_distributed(device=device)
        if ddp.enabled and device.startswith("cuda"):
            device = f"cuda:{ddp.local_rank}"
        if ddp.enabled:
            print(f"[rank {ddp.rank}/{ddp.world_size}] device={device}", flush=True)
        dtype = _get(global_cfg, "dtype", None)
        deterministic = bool(_get(global_cfg, "deterministic", False))

        out_dir = Path(_get(global_cfg, "output.out_dir", "src/checkpoints/finetune_text"))
        save_format_default = str(_get(global_cfg, "output.save_format", "full"))
        save_last_epoch_default = bool(_get(global_cfg, "output.save_last_epoch", False))
        extract_heads_default = bool(_get(global_cfg, "output.extract_heads", False))
        heads_path_default = _get(global_cfg, "output.heads_path", None)
        logging_cfg = merge_logging_config(_get(global_cfg, "logging", {}), build_logging_overrides(args))

        model_tag = _safe_model_tag(model_name_or_path)
        run_ts = int(time.time())
        run_path = default_summary_path(
            entrypoint="finetune.train_text",
            logging_cfg=logging_cfg,
            default_parent=out_dir / model_tag,
            timestamp=run_ts,
        )
        startup_cfg = deepcopy(common)
        startup_cfg["config"] = args.text_config
        startup_cfg["tasks"] = list(tasks)
        startup_cfg["device"] = device
        startup_cfg["dtype"] = dtype
        startup_cfg["deterministic"] = deterministic
        startup_cfg["logging"] = logging_cfg
        startup_cfg["summary"] = str(run_path)
        startup_cfg.setdefault("backbone", {})
        startup_cfg["backbone"]["name"] = backbone_name
        startup_cfg["backbone"]["model_name_or_path"] = model_name_or_path
        startup_cfg["backbone"]["model_arch"] = model_arch
        startup_cfg["backbone"]["model_kind"] = model_kind
        startup_cfg["backbone"]["trust_remote_code"] = trust_remote_code
        startup_cfg["backbone"]["use_fast_tokenizer"] = use_fast_tokenizer
        startup_cfg.setdefault("output", {})
        startup_cfg["output"]["out_dir"] = str(out_dir)
        startup_cfg["output"]["save_format"] = save_format_default
        startup_cfg["output"]["save_last_epoch"] = save_last_epoch_default
        startup_cfg["output"]["extract_heads"] = extract_heads_default
        startup_cfg["output"]["heads_path"] = heads_path_default

        all_summaries: dict[str, Any] = {
            "config_path": args.text_config,
            "common": common,
            "cli": {
                "suite": args.suite,
                "tasks": args.tasks,
                "device": args.device,
                "logging": build_logging_overrides(args),
            },
            "resolved": {
                "tasks": tasks,
                "build_cfg": {
                    "backbone": backbone_name,
                    "model_name_or_path": model_name_or_path,
                    "model_arch": model_arch,
                    "model_kind": model_kind,
                    "dtype": dtype,
                    "device": device,
                },
                "run_path": str(run_path),
            },
            "results": {},
        }
        # Rank 0 owns the run log: N ranks writing the same jsonl would interleave.
        if ddp.is_main:
            run_logger = start_run(
                entrypoint="finetune.train_text",
                logging_cfg=logging_cfg,
                summary_path=run_path,
                metadata={
                    "config_path": args.text_config,
                    "summary_path": str(run_path),
                    "resolved_config": startup_cfg,
                },
            )

        extracted_heads: dict[str, dict[str, torch.Tensor]] = {}

        for task in tasks:
            task = str(task).strip().lower()
            if not is_causal and task not in NLI_TASKS:
                raise ValueError(f"Unknown task '{task}'. Supported: {list(NLI_TASKS)}")

            task_cfg = deepcopy(common)
            _deep_update(task_cfg, _get_dataset_override(cfg_file, task))
            task_logging_cfg = merge_logging_config(_get(task_cfg, "logging", {}), build_logging_overrides(args))

            epochs = _get(task_cfg, "train.epochs", None)
            if epochs is None:
                raise ValueError(f"[{task}] train.epochs missing. Set common.train.epochs or datasets.{task}.train.epochs.")
            epochs = int(epochs)

            strategy_cfg = _get(task_cfg, "strategy", {})
            if not isinstance(strategy_cfg, dict):
                raise ValueError(f"[{task}] strategy must be a dict.")
            resolve_training_forward_mode(strategy_cfg)
            strategy = str(_get(task_cfg, "strategy.name", "full"))
            if strategy not in {"full", "linear_probe", "peft_lora"}:
                raise ValueError(f"[{task}] Unsupported strategy '{strategy}'. Use one of: full, linear_probe, peft_lora")
            if is_causal and strategy != "peft_lora":
                raise ValueError(
                    f"[{task}] model_kind='causal_lm' supports strategy.name='peft_lora' only (got '{strategy}')."
                )

            optimizer_name = str(_get(task_cfg, "train.optimizer.name", "adamw"))
            lr = float(_get(task_cfg, "train.lr", 1e-4))
            weight_decay = float(_get(task_cfg, "train.weight_decay", 0.0))
            warmup_length = int(_get(task_cfg, "train.lr_scheduler.warmup_steps", 500))
            scheduler_name = str(_get(task_cfg, "train.lr_scheduler.name", "cosine"))
            clip_grad_norm = float(_get(task_cfg, "train.grad_clip_norm", 1.0))
            accumulate_grad_batches = int(_get(task_cfg, "train.accumulate_grad_batches", 1))
            if accumulate_grad_batches <= 0:
                raise ValueError(f"[{task}] train.accumulate_grad_batches must be >= 1.")

            batch_size = int(_get(task_cfg, "data.batch_size", 8))
            num_workers = int(_get(task_cfg, "data.num_workers", 0))
            max_length = int(_get(task_cfg, "data.max_length", 512))
            if is_causal:
                head_num_labels = 0
            else:
                task_num_labels = int(len(build_nli_task_data(task=task, split="train", max_samples=1).labels))
                cfg_head_num_labels = int(
                    _get(task_cfg, "backbone.num_labels", _get(global_cfg, "backbone.num_labels", task_num_labels))
                )
                if cfg_head_num_labels != task_num_labels:
                    print(
                        f"[{task}] overriding backbone.num_labels from {cfg_head_num_labels} "
                        f"to {task_num_labels} to match dataset labels."
                    )
                head_num_labels = int(task_num_labels)

            seed = int(_get(task_cfg, "seed", 42))
            early_stopping = bool(_get(task_cfg, "train.early_stopping", False))
            early_stopping_patience = int(_get(task_cfg, "train.early_stopping_patience", 5))
            eval_every_n_steps = int(_get(task_cfg, "train.eval_every_n_steps", 0))
            max_steps_raw = _get(task_cfg, "train.max_steps", None)
            max_steps = None if max_steps_raw is None else int(max_steps_raw)
            save_every_n_steps = int(_get(task_cfg, "train.save_every_n_steps", 0))
            resume_from_raw = _get(task_cfg, "train.resume_from", None)
            resume_from = str(resume_from_raw) if resume_from_raw else None
            resume_allow_lr_change = bool(_get(task_cfg, "train.resume_allow_lr_change", False))
            save_best_to_disk = bool(_get(task_cfg, "train.save_best_to_disk", True))

            task_out_dir = Path(_get(task_cfg, "output.out_dir", str(out_dir)))
            save_format = str(_get(task_cfg, "output.save_format", save_format_default))
            save_last_epoch = bool(_get(task_cfg, "output.save_last_epoch", save_last_epoch_default))
            extract_heads = bool(_get(task_cfg, "output.extract_heads", extract_heads_default))

            if save_format not in {"full", "head", "peft", "hf"}:
                raise ValueError(f"[{task}] output.save_format must be one of: full, head, peft, hf")
            if save_format in {"peft", "hf"} and strategy != "peft_lora":
                raise ValueError(f"[{task}] save_format='{save_format}' requires strategy.name='peft_lora'.")
            if is_causal and save_format in {"head"}:
                raise ValueError(f"[{task}] save_format='head' is meaningless for model_kind='causal_lm'.")
            if is_causal and extract_heads:
                raise ValueError(f"[{task}] output.extract_heads is unsupported for model_kind='causal_lm'.")

            build_cfg = TextBuildConfig(
                model_name_or_path=str(model_name_or_path),
                model_arch=str(_get(task_cfg, "backbone.model_arch", model_arch)),
                device=str(device),
                dtype=_get(task_cfg, "dtype", dtype),
                model_kind=str(_get(task_cfg, "backbone.model_kind", model_kind)),
                num_labels=max(1, int(head_num_labels)),
                trust_remote_code=bool(_get(task_cfg, "backbone.trust_remote_code", trust_remote_code)),
                use_fast_tokenizer=bool(_get(task_cfg, "backbone.use_fast_tokenizer", use_fast_tokenizer)),
            )

            summary, head_payload = train_task(
                task=task,
                build_cfg=build_cfg,
                strategy=strategy,
                strategy_cfg=strategy_cfg,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                warmup_length=warmup_length,
                scheduler_name=scheduler_name,
                optimizer_name=optimizer_name,
                clip_grad_norm=clip_grad_norm,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                head_num_labels=head_num_labels,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                eval_every_n_steps=eval_every_n_steps,
                max_steps=max_steps,
                save_every_n_steps=save_every_n_steps,
                resume_from=resume_from,
                resume_allow_lr_change=resume_allow_lr_change,
                save_best_to_disk=save_best_to_disk,
                seed=seed,
                deterministic=deterministic,
                device=str(device),
                out_dir=task_out_dir,
                save_format=save_format,
                save_last_epoch=save_last_epoch,
                task_cfg=task_cfg,
                log_every_n_steps=int(task_logging_cfg.get("log_every_n_steps", 50)),
                run_logger=run_logger,
                dist_info=ddp,
            )

            all_summaries["results"][task] = summary
            if extract_heads:
                extracted_heads[task] = head_payload

        if ddp.is_main:
            _save_json(run_path, all_summaries)
            run_logger.log_summary(all_summaries)
            print(f"\nSaved run summary: {run_path}")

        if extracted_heads and ddp.is_main:
            heads_path_raw = heads_path_default
            if isinstance(heads_path_raw, str) and heads_path_raw.strip():
                heads_path = Path(heads_path_raw)
            else:
                heads_path = out_dir / model_tag / "heads.pt"
            _ensure_dir(heads_path.parent)
            torch.save(extracted_heads, heads_path)
            print(f"Saved extracted task heads: {heads_path}")
            run_logger.log_event(
                "artifact_saved",
                metrics={},
                context={
                    "artifact": "heads.pt",
                    "path": str(heads_path),
                },
            )
        if run_logger is not None:
            run_logger.finish("success")
        shutdown_distributed(ddp)
    except Exception as exc:
        finish_with_error(run_logger, exc)
        shutdown_distributed(ddp)
        raise


if __name__ == "__main__":
    main()
