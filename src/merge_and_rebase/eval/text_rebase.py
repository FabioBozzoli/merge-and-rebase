"""
Rebase task vectors from a source base LM A to a target base LM B and evaluate.

Text twin of ``eval/vision_rebase.py``, for HuggingFace ``transformers`` models
(T5 / Qwen / Llama / anything ``models/text_lm.py`` can build). The two files are
deliberately parallel -- same config surface, same alpha grid and early stopping,
same ``untransported`` / ``target_zeroshot`` baselines, same ``final_summary``
keys -- so a text run and a vision run of the same method are directly
comparable, and ``scripts/audit_rebase_summary.py`` reads both.

Nothing under ``rebase/methods/`` is modified. What the methods need arrives
from ``rebase/text/``:

- ``gradfix`` runs unchanged: it is model-agnostic and driven by a ``GradRecipe``,
  and ``models/grad_recipes.py`` already ships the text ones.
- ``theseus`` and ``bico`` run through :class:`~merge_and_rebase.rebase.text.TextEncoderShim`
  plus an ``"inputs"``-aliased loader, which is all that stands between their
  image-shaped forward and an HF dict batch.
- ``steer`` cannot be adapted at all (CLIP classifiers, zero-shot text heads,
  ViT block tables), so ``rebase/text/steer_text.py`` rebuilds its scaffolding
  for text and imports its math verbatim. Use ``"method": "steer_text"``.

Not supported, with an explicit error rather than a silent wrong answer:
``transfusion`` (its permutation spec is OpenCLIP-ViT only) and ``bico_gradin``
(it calls ``requires_grad_`` on the input, which integer ``input_ids`` reject).

Usage:
    python -m merge_and_rebase.eval.text_rebase --config configs/text_rebase_t5_theseus.json
"""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from merge_and_rebase.utils.helpers import load_json, parse_csv

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
from ..data.text_loaders import (
    NLI_TASKS,
    NLIExample,
    NLITaskData,
    build_nli_task_data,
    build_nli_tokenized_loader,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model, resolve_ckpt_path
from ..merge.methods._common import axpy_state_dict
from ..merge.task_vectors import TaskVector
from ..models.grad_recipes import causal_lm_recipe, seq_classification_recipe
from ..models.text_lm import TextBuildConfig, TextLM
from ..rebase import get_method, list_methods
from ..rebase.runtime import format_rebase_method_label, resolve_rebase_method_config
from ..rebase.text import (  # noqa: F401  -- import registers "steer_text"
    TextEncoderShim,
    alias_inputs_loader,
    attach_local_labels,
    balanced_indices,
    count_transformer_blocks,
    describe_key_coverage,
    feature_separability,
    neutralize_intermediate_head_layers,
    steer_text_correction_context,
    subset_loader,
    text_param_filter,
    train_linear_probe_head,
)
from ..rebase.text.steer_text import _head_as_identity, _pooled_features
from ..run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from ..utils.alpha_search import PerTaskAlphaTracker, average_scores
from .llm_merge import (
    _default_prompt_for_task,
    _head_class_ids_for_task,
    _inject_task_head,
    _load_task_heads,
    _resolve_eval_mode,
)
from .print_utils import pretty_print_task_accuracies
from .rebase_metrics import normalized_accuracy_ratio
from .utils import to_cpu_fp32

SUITES: dict[str, tuple[str, ...]] = {"nli6": tuple(NLI_TASKS)}

# Fixed on purpose, independent of cfg["seed"]: the val/test carve must stay
# identical across a seed sweep (gradfix's few-shot draws, theseus/bico/steer's
# own calibration draws) so runs only compare different calibration samples
# against the same held-out data, not also reshuffle what is held out.
# Same rationale and same role as vision_rebase._VAL_TEST_SPLIT_SEED.
_VAL_TEST_SPLIT_SEED = 0

_UNSUPPORTED_METHODS: dict[str, str] = {
    "transfusion": (
        "transfusion needs a weight-permutation spec, and rebase/permutations/spec.py provides "
        "only CLIP_Visual_PermutationSpecBuilder (OpenCLIP ViT key names). A T5/Qwen spec builder "
        "does not exist yet."
    ),
    "steer": (
        "steer is CLIP-only (OpenClipClassifier, zero-shot text heads, ViT/ResNet block tables). "
        "Use \"method\": \"steer_text\", the text port in rebase/text/steer_text.py."
    ),
    "bico_gradin": (
        "bico_gradin marks the model input as requiring grad, which integer input_ids cannot do. "
        "Use \"method\": \"bico\"."
    ),
}


@dataclass
class TextLoaders:
    """Per-task train/val/test loaders for one tokenizer. Twin of ``VisionLoaders``."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    mask_class: list[int]
    head_class_ids: list[int]
    label_texts: list[str]
    examples: dict[str, list[NLIExample]]
    local_labels: dict[str, list[int]]


def _slice_task_data(task_data: NLITaskData, indices: list[int]) -> NLITaskData:
    return NLITaskData(
        task=task_data.task,
        examples=[task_data.examples[i] for i in indices],
        labels=list(task_data.labels),
        label_texts=list(task_data.label_texts),
        meta={**task_data.meta, "num_examples": len(indices)},
    )


def _build_task_splits(
    *,
    task: str,
    eval_split: str,
    val_fraction: float,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> dict[str, NLITaskData]:
    """Train + a reproducible val/test carve of ``eval_split``.

    ``data/text_loaders.py`` hands back real HF splits, and for qnli/rte/mnli its
    ``test`` split *is* ``validation`` -- so searching alpha on "val" and
    reporting on "test" would read the same rows. Carving val out of the eval
    split instead reproduces exactly what ``build_vision_loaders`` does ("val is
    a reproducible random slice of test ... val/test are disjoint subsets"), and
    keeps the alpha selection honest.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    train_td = build_nli_task_data(task=task, split="train", max_samples=max_train_samples)
    eval_td = build_nli_task_data(task=task, split=eval_split, max_samples=max_eval_samples)

    generator = torch.Generator().manual_seed(_VAL_TEST_SPLIT_SEED)
    perm = torch.randperm(len(eval_td.examples), generator=generator).tolist()
    n_val = int(round(float(val_fraction) * len(perm)))
    if n_val <= 0 or n_val >= len(perm):
        raise ValueError(
            f"val_fraction={val_fraction} carves {n_val} of {len(perm)} examples for task '{task}'; "
            "raise max_samples_per_task or val_fraction."
        )
    return {
        "train": train_td,
        "val": _slice_task_data(eval_td, sorted(perm[:n_val])),
        "test": _slice_task_data(eval_td, sorted(perm[n_val:])),
    }


def _tokenize_splits(
    *,
    splits: dict[str, NLITaskData],
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    head_class_ids: list[int],
    premise_hypothesis_template: str | None = None,
) -> TextLoaders:
    tokenized = {
        name: build_nli_tokenized_loader(
            task_data=task_data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            shuffle=False,
            head_class_ids=head_class_ids,
            premise_hypothesis_template=premise_hypothesis_template,
        )
        for name, task_data in splits.items()
    }
    local_labels = {name: [int(ex.label) for ex in splits[name].examples] for name in splits}
    for name, tk in tokenized.items():
        # Enables theseus/bico's shots_per_class calibration; must be the local
        # 0..K-1 ids, see rebase/text/adapters.attach_local_labels.
        attach_local_labels(tk.loader.dataset, local_labels[name])

    any_tk = tokenized["test"]
    return TextLoaders(
        train=tokenized["train"].loader,
        val=tokenized["val"].loader,
        test=tokenized["test"].loader,
        mask_class=list(any_tk.mask_class),
        head_class_ids=list(head_class_ids),
        label_texts=list(splits["test"].label_texts),
        examples={name: list(td.examples) for name, td in splits.items()},
        local_labels=local_labels,
    )


def _resolve_eval_loader(loaders_obj: TextLoaders, split: str) -> DataLoader:
    if split == "val" and loaders_obj.val is not None:
        return loaders_obj.val
    return loaders_obj.test


def _scale_delta(delta_sd: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
    if float(weight) == 1.0:
        return delta_sd
    return {k: v * float(weight) for k, v in delta_sd.items()}


def _check_untransported_compatibility(
    base_sd: dict[str, torch.Tensor], delta_sd: dict[str, torch.Tensor]
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for key, value in delta_sd.items():
        if key not in base_sd:
            issues.append(f"missing in target: {key}")
        elif tuple(base_sd[key].shape) != tuple(value.shape):
            issues.append(f"shape mismatch for {key}: target={tuple(base_sd[key].shape)} delta={tuple(value.shape)}")
    return (not issues), issues


def _norm_acc(result_acc: float, baseline_acc: float) -> float:
    return normalized_accuracy_ratio(result_acc, baseline_acc)


def _average_defined(values: list[float]) -> float:
    defined = [v for v in values if v == v]
    return average_scores(defined) if defined else float("nan")


@torch.no_grad()
def _pooled_feature_report(
    *,
    llm: TextLM,
    loaders: TextLoaders,
    split: str,
    device: str,
    label: str,
) -> dict[str, float]:
    """Pooled-feature class separability for one model, on one split.

    Answers the question a bare accuracy number cannot: is a near-chance
    result caused by the *classifier* on top of the features, or by the
    features themselves carrying no class signal to begin with? Run on the
    source and the target with the same metric and the numbers are directly
    comparable -- if A's fine-tuned features separate and B's pretrained ones
    do not, the pooling machinery works and the gap is what fine-tuning built.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    chunks: list[torch.Tensor] = []
    llm.model.eval()
    with _head_as_identity(llm.model):
        for batch in _resolve_eval_loader(loaders, split):
            chunks.append(_pooled_features(llm.model, batch, dev).cpu().double())
    features = torch.cat(chunks, dim=0)
    labels = torch.as_tensor([int(y) for y in loaders.local_labels[split]], dtype=torch.long)[: features.shape[0]]

    stats = feature_separability(torch.nn.functional.normalize(features, dim=-1), labels)
    print(
        f"  [{label}] pooled-feature separability on '{split}': "
        f"within={stats['within_class_cosine']:.4f} between={stats['between_class_cosine']:.4f} "
        f"gap={stats['gap']:+.4f} pairwise_std={stats['pairwise_std']:.4f}"
    )
    return stats


def _checkpoint_training_metadata(ckpt_path: str) -> dict[str, Any]:
    """What the fine-tuning run itself recorded inside the checkpoint.

    ``finetune/train_text.py`` stores the accuracy it measured
    (``metrics.val_top1`` / ``metrics.test_top1``) plus the label space it
    trained against. ``io.ckpt.load_ckpt`` unwraps straight to the tensors and
    drops all of it, so read the raw payload here.

    This is the cheapest possible way to tell "the checkpoint never learned the
    task" apart from "the checkpoint is fine and our evaluation path disagrees
    with the one training used" -- the two produce identical downstream
    symptoms but need opposite fixes.
    """
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"error": f"checkpoint is a {type(payload).__name__}, not a payload dict"}
    return {
        key: payload[key]
        for key in ("metrics", "best_epoch", "last_epoch", "strategy", "forward_mode", "num_labels", "labels", "head_class_ids", "format")
        if key in payload
    }


def _evaluate_source_finetuned(
    *,
    llm_source_finetuned: TextLM,
    llm_source_pretrained: TextLM,
    source_loaders: TextLoaders,
    task: str,
    device: str,
    eval_mode: str,
    ckpt_path: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Score A's own fine-tuned checkpoint, plus feature diagnostics for A and A-pretrained.

    This is the control the rest of the run is missing. Every reported number
    (target_zeroshot, rebased) is read through B's head, so a near-chance
    result there is ambiguous on its own. A's fine-tuned accuracy is measured
    with A's own trained head on the same task split, so it isolates whether
    the architecture, tokenization and pooling can support this task *at all*
    once something has actually been trained through them.
    """
    out: dict[str, Any] = {"task": task, "split": split}
    recorded: dict[str, Any] = {}
    if ckpt_path is not None and _is_hub_model_reference(ckpt_path):
        print(f"  [A checkpoint] '{ckpt_path}' is a full HF Hub model, not a local checkpoint file -- no training metadata to read.")
    elif ckpt_path is not None:
        recorded = _checkpoint_training_metadata(ckpt_path)
        out["checkpoint_recorded"] = recorded
        if "error" in recorded:
            print(f"  [A checkpoint] could not read training metadata: {recorded['error']}")
        else:
            metrics = recorded.get("metrics", {}) or {}
            print(
                f"  [A checkpoint] as recorded by training: val_top1={metrics.get('val_top1')} "
                f"test_top1={metrics.get('test_top1')} strategy={recorded.get('strategy')} "
                f"labels={recorded.get('labels')} head_class_ids={recorded.get('head_class_ids')}"
            )

    acc: float | None = None
    if eval_mode == "head_logits":
        acc = float(
            llm_source_finetuned.sequence_classification_accuracy(
                _resolve_eval_loader(source_loaders, split),
                device=device,
                mask_class=source_loaders.mask_class,
            )
        )
        out["source_finetuned_accuracy"] = acc
        print(f"  [A finetuned] accuracy on '{split}' with its own trained head: {acc:.4f}")

    out["source_finetuned_features"] = _pooled_feature_report(
        llm=llm_source_finetuned, loaders=source_loaders, split=split, device=device, label="A finetuned"
    )
    out["source_pretrained_features"] = _pooled_feature_report(
        llm=llm_source_pretrained, loaders=source_loaders, split=split, device=device, label="A pretrained"
    )
    a_ft_gap = out["source_finetuned_features"]["gap"]
    a_pre_gap = out["source_pretrained_features"]["gap"]
    recorded_acc = (recorded.get("metrics") or {}).get("test_top1") if recorded else None
    chance = 1.0 / max(1, len(source_loaders.mask_class))

    if acc is not None and acc < chance + 0.05:
        # A cannot classify its own task. Nothing downstream (a head built on
        # B, a transported delta derived from A) can be judged until this is
        # resolved, because every one of those numbers is measured through it.
        if recorded_acc is not None and float(recorded_acc) > chance + 0.15:
            print(
                f"  => MISMATCH: training recorded test_top1={float(recorded_acc):.4f} for this checkpoint, but\n"
                f"     evaluating it here gives {acc:.4f} (chance is {chance:.2f}). The weights learned the task;\n"
                "     this evaluation path disagrees with the one training used. Compare tokenization\n"
                "     (max_length, text-pair encoding), head_class_ids and the label order above against the\n"
                "     finetune config -- the rebase results mean nothing until these agree."
            )
        else:
            print(
                f"  => A is at chance ({acc:.4f}, chance is {chance:.2f}) on its own fine-tuning task, and the\n"
                "     checkpoint's own recorded metrics do not contradict that. The checkpoint never learned\n"
                "     the task, so the delta being transported carries no task signal and every downstream\n"
                "     number here is measuring noise. Fix or retrain A before interpreting any rebase result."
            )
    elif a_ft_gap > 0.05 and abs(a_pre_gap) < 0.02:
        print(
            "  => A's pooled features separate the classes only AFTER fine-tuning. The pooling position and\n"
            "     tokenization are therefore fine: the class structure is something training creates, not\n"
            "     something a pretrained-only checkpoint already has. A target head built from B's untrained\n"
            "     features (nearest-mean) has nothing to latch onto -- train a linear probe on B instead, or\n"
            "     start from an instruction-tuned/LM-adapted B."
        )
    elif a_ft_gap < 0.02:
        print(
            "  => A classifies well but its pooled features show no class separation. The tensor being read\n"
            "     as 'the pooled feature' is then not the one the classifier actually uses -- a pooling-position\n"
            "     bug upstream, worth fixing before any conclusion about the rebase methods."
        )
    return out


def _is_hub_model_reference(ref: str) -> bool:
    """Whether ``ref`` names a full HF Hub model repo rather than a local checkpoint file.

    Distinguishes ``varun-v-rao/t5-base-snli`` (a complete, already fine-tuned
    ``AutoModelForSequenceClassification`` checkpoint, loadable on its own via
    ``from_pretrained``) from a local ``.pt``/``.bin`` file holding a task
    delta relative to a shared local base. Anything that exists on disk is a
    local file, never a Hub reference, even if its name happens to contain a
    slash.
    """
    if Path(ref).exists():
        return False
    return "/" in ref and not ref.lower().endswith((".pt", ".bin", ".safetensors", ".ckpt", ".pth"))


def _load_tuned_source_state_dict(
    ref: str,
    *,
    base_sd: dict[str, torch.Tensor],
    source_cfg: TextBuildConfig,
    model_kind: str,
) -> dict[str, torch.Tensor]:
    """A task's fine-tuned state dict, aligned to A's base key space.

    Two shapes of ``tuned_ckpts`` entry are supported: a local file (the
    existing convention -- a delta-shaped checkpoint alongside a shared local
    base, read with ``load_ckpt``), or a bare HF Hub model id, which is loaded
    as a complete, already fine-tuned ``AutoModelForSequenceClassification``/
    ``AutoModelForSeq2SeqLM`` in its own right (e.g. a community checkpoint
    such as ``varun-v-rao/t5-base-snli``) -- there is no local delta file for
    these at all, the Hub repo *is* the fine-tuned model.

    For the Hub case, ``source_model_name_or_path`` in the config must be the
    actual pretrained base that checkpoint was fine-tuned from (tokenizer
    vocab size and architecture must match, or key alignment below silently
    drops every mismatched tensor) -- this function has no way to verify that
    against the Hub repo's own model card, so a task_delta computed from a
    wrong guess is a silent correctness risk, not just a missed transport.
    """
    if _is_hub_model_reference(ref):
        print(f"  Loading '{ref}' as a full HF Hub sequence-classification checkpoint (not a local delta file).")
        hub_cfg = TextBuildConfig(
            model_name_or_path=ref,
            model_arch=source_cfg.model_arch,
            device=source_cfg.device,
            dtype=source_cfg.dtype,
            model_kind=model_kind,
            num_labels=source_cfg.num_labels,
            trust_remote_code=source_cfg.trust_remote_code,
            use_fast_tokenizer=source_cfg.use_fast_tokenizer,
        )
        hub_llm = TextLM.build(hub_cfg)
        tuned_raw: dict[str, torch.Tensor] = dict(hub_llm.model.state_dict())
        del hub_llm
    else:
        tuned_raw = dict(load_ckpt(ref))

    aligned = align_to_base_keys(tuned_raw, base_sd)
    if not aligned:
        raise ValueError(f"No tensors from '{ref}' aligned to the source model's keys.")
    return aligned


def _build_llm(cfg: dict[str, Any], *, role: str, model_kind: str, device: str) -> tuple[TextLM, TextBuildConfig]:
    name_key = f"{role}_model_name_or_path"
    if not cfg.get(name_key):
        raise ValueError(f"Provide config['{name_key}'] (the {role} base model A/B).")
    build_cfg = TextBuildConfig(
        model_name_or_path=str(cfg[name_key]),
        model_arch=str(cfg.get(f"{role}_model_arch", cfg.get("model_arch", "auto"))),
        device=device,
        dtype=cfg.get("dtype", None),
        model_kind=model_kind,
        num_labels=int(cfg.get("num_labels", 3)),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
    )
    return TextLM.build(build_cfg), build_cfg


def _model_tag(build_cfg: TextBuildConfig) -> str:
    return str(build_cfg.model_name_or_path).replace("/", "__")


def main() -> None:
    run_logger = None
    try:
        p = argparse.ArgumentParser("Rebase task vectors from source base LM A to target base LM B and evaluate")

        add_config_arg(p)
        add_suite_arg(p, choices=sorted(SUITES.keys()))
        add_tasks_arg(p, help_text="Comma-separated task names, or 'all'.")

        p.add_argument("--source-model-name-or-path", type=str, default=None)
        p.add_argument("--source-model-arch", type=str, default=None, choices=["llama", "t5", "auto"])
        p.add_argument("--target-model-name-or-path", type=str, default=None)
        p.add_argument("--target-model-arch", type=str, default=None, choices=["llama", "t5", "auto"])

        add_device_dtype_args(p, device_default=None, dtype_default=None)

        p.add_argument("--eval-mode", type=str, default=None, choices=["auto", "prompt", "head_logits"])
        p.add_argument("--source-task-heads", type=str, default=None)
        p.add_argument("--target-task-heads", type=str, default=None)
        p.add_argument("--head-key-pattern", type=str, default=None)
        p.add_argument("--prompt-template", type=str, default=None)
        p.add_argument("--max-prompt-tokens", type=int, default=None)

        p.add_argument("--split", type=str, default=None, choices=["train", "validation", "test"])
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--max-length", type=int, default=None)
        p.add_argument("--max-samples-per-task", type=int, default=None)
        p.add_argument("--max-train-samples", type=int, default=None)
        p.add_argument("--val-fraction", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)

        p.add_argument("--tuned-ckpts", type=str, nargs="+", default=None)
        p.add_argument("--weights", type=float, nargs="*", default=None)
        p.add_argument("--strict-load", action="store_true", default=None)
        p.add_argument("--method", type=str, choices=list_methods(), default=None)
        p.add_argument("--method-params", type=str, default=None, help="JSON object for rebase-method kwargs.")

        p.add_argument("--mask-mode", type=str, default=None, choices=["normal", "force"])
        p.add_argument("--vote", type=str, default=None, choices=["mean", "majority", "max"])
        p.add_argument("--grad-batch-size", type=int, default=None)
        p.add_argument("--grad-examples-per-class", type=int, default=None)

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
        p.add_argument("--alpha-search-split", type=str, default=None, choices=["val", "test"])
        p.add_argument("--save-transported-tvs-dir", type=str, default=None)
        p.add_argument(
            "--eval-source-finetuned",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Also score A's own fine-tuned checkpoint (with its own trained head) on each task, "
            "as a control for whether the architecture/tokenization/pooling support the task at all.",
        )
        p.add_argument(
            "--source-input-template",
            type=str,
            default=None,
            help="A's fine-tune expects premise/hypothesis as one formatted string, e.g. "
            "'premise: {premise} hypothesis: {hypothesis}', instead of the tokenizer's own pair encoding.",
        )
        p.add_argument(
            "--target-input-template",
            type=str,
            default=None,
            help="Same as --source-input-template, for B.",
        )
        p.add_argument(
            "--feature-cache-dir",
            type=str,
            default=None,
            help="steer_text's on-disk cache for pooled features (keyed by source/target/task/regime/split). "
            "A top-level shortcut for method_params.feature_cache_dir, so it survives a --method-params "
            "override that replaces the rest of the dict.",
        )
        add_logging_args(p)

        args = p.parse_args()
        method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")

        cfg: dict[str, Any] = {}
        if args.config is not None:
            cfg = load_json(args.config)

        cli: dict[str, Any] = {
            "source_model_name_or_path": args.source_model_name_or_path,
            "source_model_arch": args.source_model_arch,
            "target_model_name_or_path": args.target_model_name_or_path,
            "target_model_arch": args.target_model_arch,
            "suite": getattr(args, "suite", None),
            "tasks": getattr(args, "tasks", None),
            "device": args.device,
            "dtype": args.dtype,
            "eval_mode": args.eval_mode,
            "source_task_heads": args.source_task_heads,
            "target_task_heads": args.target_task_heads,
            "head_key_pattern": args.head_key_pattern,
            "prompt_template": args.prompt_template,
            "max_prompt_tokens": args.max_prompt_tokens,
            "split": args.split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_length": args.max_length,
            "max_samples_per_task": args.max_samples_per_task,
            "max_train_samples": args.max_train_samples,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "tuned_ckpts": args.tuned_ckpts,
            "weights": args.weights,
            "strict_load": args.strict_load,
            "method": args.method,
            "method_params": method_params_cli,
            "mask_mode": args.mask_mode,
            "vote": args.vote,
            "grad_batch_size": args.grad_batch_size,
            "grad_examples_per_class": args.grad_examples_per_class,
            "alpha_search": getattr(args, "alpha_search", None),
            "alpha_selection": args.alpha_selection,
            "alpha_patience": args.alpha_patience,
            "alpha_search_split": args.alpha_search_split,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_step": args.alpha_step,
            "alpha": args.alpha,
            "save_transported_tvs_dir": args.save_transported_tvs_dir,
            "eval_source_finetuned": args.eval_source_finetuned,
            "source_input_template": args.source_input_template,
            "target_input_template": args.target_input_template,
            "feature_cache_dir": args.feature_cache_dir,
        }
        cfg = merge_non_none(cfg, {k: v for k, v in cli.items() if v is not None})
        logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
        cfg["logging"] = logging_cfg

        method_name, method_params = resolve_rebase_method_config(cfg)
        if method_name in _UNSUPPORTED_METHODS:
            raise ValueError(f"Rebase method '{method_name}' is not supported for text models. {_UNSUPPORTED_METHODS[method_name]}")
        method = get_method(method_name)
        method_label = format_rebase_method_label(method_name, method_params)
        theseus_like_method = method_name in {"theseus", "theseus_reference"}
        bico_mode = method_name == "bico"
        steer_mode = method_name == "steer_text"
        shim_mode = theseus_like_method or bico_mode

        if shim_mode and "seq_align" not in method_params:
            # The vision default, "interpolate2d", reshapes tokens into a square
            # patch grid (theseus._interp_2d_tokens). Text token counts are not
            # square, so linear interpolation is the right alignment here.
            method_params["seq_align"] = "interpolate"
            print("Rebase: defaulting method_params.seq_align='interpolate' (text token sequences are not a square grid).")

        strict_load = bool(cfg.get("strict_load", False))
        device = str(cfg.get("device", "cuda"))
        eval_source_finetuned = bool(cfg.get("eval_source_finetuned", False))
        # Some checkpoints (e.g. a community Hub upload) were fine-tuned on a
        # single formatted string ("premise: {premise} hypothesis: {hypothesis}")
        # rather than the tokenizer's own two-segment pair encoding -- the two
        # are different token sequences, and a model trained on one performs at
        # chance on the other. See scripts/probe_nli_input_format.py to find the
        # right template for a given checkpoint before setting this.
        source_input_template = cfg.get("source_input_template", None)
        target_input_template = cfg.get("target_input_template", None)

        grad_batch_size = int(cfg["grad_batch_size"]) if cfg.get("grad_batch_size") is not None else None
        grad_examples_per_class = (
            int(cfg["grad_examples_per_class"]) if cfg.get("grad_examples_per_class") is not None else None
        )

        alpha_search = bool(cfg.get("alpha_search", False))
        alpha_patience_raw = cfg.get("alpha_patience", 0)
        alpha_patience = int(alpha_patience_raw) if alpha_patience_raw is not None else 0
        if alpha_patience < 0:
            raise ValueError("alpha_patience must be >= 0")

        alpha_search_split = str(cfg.get("alpha_search_split", "val")).strip().lower()
        if alpha_search_split not in {"val", "test"}:
            raise ValueError("alpha_search_split must be one of: val, test")

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

        suite_name = str(cfg.get("suite", "nli6"))
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITES)}")
        suite_tasks = SUITES[suite_name]

        tasks_arg = cfg.get("tasks", "all")
        if tasks_arg == "all":
            tasks = list(suite_tasks)
        else:
            tasks = [t.strip().lower() for t in (parse_csv(tasks_arg) if isinstance(tasks_arg, str) else list(tasks_arg))]
            bad = [t for t in tasks if t not in suite_tasks]
            if bad:
                raise ValueError(f"Unknown tasks: {bad}. Allowed: {sorted(suite_tasks)}")

        target_task_heads_path = cfg.get("target_task_heads", cfg.get("task_heads", None))
        target_task_heads_path = str(target_task_heads_path) if target_task_heads_path is not None else None
        source_task_heads_path = cfg.get("source_task_heads", None)
        source_task_heads_path = str(source_task_heads_path) if source_task_heads_path is not None else None
        eval_mode = _resolve_eval_mode(str(cfg.get("eval_mode", "auto")), target_task_heads_path)
        head_key_pattern = str(cfg.get("head_key_pattern", "classification_head"))
        model_kind = str(cfg.get("model_kind", "sequence_classification" if eval_mode == "head_logits" else "causal_lm"))

        linear_probe_head = bool(cfg.get("linear_probe_head", False))
        if linear_probe_head and target_task_heads_path is not None:
            raise ValueError(
                "linear_probe_head and target_task_heads/task_heads are mutually exclusive: the target head "
                "is trained from scratch instead of loaded, so a fixed head file would just be overwritten "
                "before it's ever used."
            )
        if linear_probe_head and method_name not in {"theseus", "bico", "steer_text"}:
            raise ValueError("linear_probe_head is only supported for method in {theseus, bico, steer_text}.")
        linear_probe_epochs = int(cfg.get("linear_probe_epochs", 200))
        linear_probe_lr = float(cfg.get("linear_probe_lr", 1e-2))
        # None -> ~10 log lines over the run. Each line scores support/val/test,
        # so a small value here buys more curve at the cost of extra eval passes.
        linear_probe_log_every = cfg.get("linear_probe_log_every", None)
        linear_probe_log_every = int(linear_probe_log_every) if linear_probe_log_every is not None else None

        if eval_mode == "head_logits":
            if model_kind != "sequence_classification":
                raise ValueError("eval_mode='head_logits' requires model_kind='sequence_classification'.")
            if target_task_heads_path is None and not linear_probe_head:
                raise ValueError(
                    "eval_mode='head_logits' requires config['target_task_heads'] (or 'task_heads'), unless "
                    "'linear_probe_head' is set."
                )
        elif model_kind == "sequence_classification":
            raise ValueError("eval_mode='prompt' requires model_kind='causal_lm' (there is no head to score).")
        if linear_probe_head and eval_mode != "head_logits":
            raise ValueError(
                "linear_probe_head trains a classification head, so it requires eval_mode='head_logits' "
                "(set it explicitly -- 'auto' resolves to 'prompt' when no target_task_heads path is given)."
            )
        if steer_mode and eval_mode != "head_logits":
            raise ValueError(
                "steer_text fits a correction in the pooled-feature space of a linear classification head, "
                "so it requires eval_mode='head_logits' with task heads."
            )

        run_summary_path = default_summary_path(
            entrypoint="eval.text_rebase",
            logging_cfg=logging_cfg,
            default_parent=None,
        )
        run_logger = start_run(
            entrypoint="eval.text_rebase",
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
        if isinstance(tuned_by_task, dict):
            tuned_by_task = {str(t).strip().lower(): resolve_ckpt_path(str(v)) for t, v in tuned_by_task.items()}
        elif tuned_by_task is not None:
            tuned_by_task = {t: resolve_ckpt_path(str(v)) for t, v in zip(tasks, tuned_by_task, strict=True)}
        if not tuned_by_task:
            raise ValueError("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'.")
        missing_ckpts = [t for t in tasks if t not in tuned_by_task]
        if missing_ckpts:
            raise ValueError(f"tuned_ckpts is missing task keys: {missing_ckpts}.")

        merge_weights = cfg.get("weights", None)
        if merge_weights is None:
            merge_weights = [1.0] * len(tasks)
        merge_weights = [float(w) for w in merge_weights]

        llm_source, source_cfg = _build_llm(cfg, role="source", model_kind=model_kind, device=device)
        llm_target, target_cfg = _build_llm(cfg, role="target", model_kind=model_kind, device=device)
        source_tag = _model_tag(source_cfg)
        target_tag = _model_tag(target_cfg)

        print(f"Source model (A): {source_cfg.model_name_or_path} ({source_cfg.model_arch})")
        print(f"Target model (B): {target_cfg.model_name_or_path} ({target_cfg.model_arch})")

        source_blocks = count_transformer_blocks(llm_source.model)
        target_blocks = count_transformer_blocks(llm_target.model)
        print(f"Transformer depth: source={source_blocks} target={target_blocks}")
        if source_blocks != target_blocks:
            print(
                "  NOTE: source and target depths differ. There is no text equivalent of the ViT-only "
                "block-extension preprocess, so only name-matching parameters are transported; the target's "
                "extra blocks keep their pretrained weights."
            )

        # Must happen BEFORE target_base_sd is snapshotted, so every later reload
        # keeps the identity. Without it B's head keeps a per-process random
        # `dense` (T5's d_model x d_model layer, absent from every base
        # checkpoint): steer_text's feature cache is keyed by
        # source/target/task/regime/split only, so cached features from an
        # earlier process silently belong to a *different* random rotation than
        # the live model's, and the probe fits tanh(random_rotation(feature))
        # rather than the feature. The nearest-mean path avoids this by baking
        # identity into its head file (scripts/build_nearest_mean_head.py); with
        # no head file to inject, this is where it has to happen.
        neutralized_head_layers: dict[str, torch.Tensor] = {}
        if linear_probe_head:
            neutralized_head_layers = neutralize_intermediate_head_layers(llm_target.model)
            if neutralized_head_layers:
                names = sorted({k.rsplit(".", 1)[0] for k in neutralized_head_layers})
                print(f"Neutralized {len(names)} intermediate head layer(s) to identity for linear probing: {names}")

        source_base_sd = to_cpu_fp32({k: v for k, v in llm_source.model.state_dict().items()})
        target_base_sd = to_cpu_fp32({k: v for k, v in llm_target.model.state_dict().items()})

        # The head is task-specific and is injected from task_heads right before
        # scoring, so transporting it would be overwritten anyway -- and its rows
        # live in a class-id space that means nothing under a different task.
        delta_key_filter = text_param_filter(exclude_head=(eval_mode == "head_logits"))

        target_task_heads: dict[str, Any] | None
        if target_task_heads_path:
            target_task_heads = _load_task_heads(target_task_heads_path)
        elif linear_probe_head:
            # Populated per task below, once each task's probe has been trained --
            # empty (not None) so the eval-time re-injection machinery activates,
            # but with no task key yet until training fills it in.
            target_task_heads = {}
        else:
            target_task_heads = None
        source_task_heads = _load_task_heads(source_task_heads_path) if source_task_heads_path else None

        user_prompt_template = cfg.get("prompt_template", None)
        max_prompt_tokens = cfg.get("max_prompt_tokens", None)
        max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens is not None else None

        eval_split = str(cfg.get("split", "test"))
        val_fraction = float(cfg.get("val_fraction", 0.1))
        batch_size = int(cfg.get("batch_size", 8))
        num_workers = int(cfg.get("num_workers", 0))
        max_length = int(cfg.get("max_length", 512))
        max_samples_per_task = cfg.get("max_samples_per_task", None)
        max_samples_per_task = int(max_samples_per_task) if max_samples_per_task is not None else None
        max_train_samples = cfg.get("max_train_samples", None)
        max_train_samples = int(max_train_samples) if max_train_samples is not None else None
        seed = int(cfg.get("seed", 42))

        print(f"Eval mode: {eval_mode} | model_kind: {model_kind}")
        print(f"Rebase method: {method_label}")

        per_task: list[dict[str, Any]] = []
        transported_deltas: list[dict[str, torch.Tensor]] = []
        original_deltas: list[dict[str, torch.Tensor]] = []
        transport_timings: dict[str, dict[str, float]] = {}
        transported_artifacts: dict[str, list[str]] = {}
        steer_prepared_by_task: dict[str, dict[str, Any]] = {}
        source_eval_rows: list[dict[str, Any]] = []

        for task in tasks:
            splits = _build_task_splits(
                task=task,
                eval_split=eval_split,
                val_fraction=val_fraction,
                max_train_samples=max_train_samples,
                max_eval_samples=max_samples_per_task,
            )
            head_class_ids = _head_class_ids_for_task(
                task=task,
                task_num_labels=len(splits["test"].labels),
                head_num_labels=int(cfg.get("num_labels", 3)),
                masked_class=(cfg.get("task_mask_class", {}) or {}).get(task, None),
            )
            loaders = _tokenize_splits(
                splits=splits,
                tokenizer=llm_target.tokenizer,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                head_class_ids=head_class_ids,
                premise_hypothesis_template=target_input_template,
            )
            source_loaders: TextLoaders | None = None
            if shim_mode or steer_mode or eval_source_finetuned:
                source_loaders = _tokenize_splits(
                    splits=splits,
                    tokenizer=llm_source.tokenizer,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_length=max_length,
                    head_class_ids=head_class_ids,
                    premise_hypothesis_template=source_input_template,
                )

            prompt_template = (
                user_prompt_template
                if isinstance(user_prompt_template, str)
                else _default_prompt_for_task(splits["test"])
            )
            per_task.append(
                {
                    "task": task,
                    "loaders": loaders,
                    "head_class_ids": head_class_ids,
                    "label_texts": list(splits["test"].label_texts),
                    "prompt_template": prompt_template,
                }
            )
            print(
                f"  {task}: train={len(splits['train'].examples)} val={len(splits['val'].examples)} "
                f"test={len(splits['test'].examples)} head_class_ids={head_class_ids}"
            )

            ckpt_path = str(tuned_by_task[task])
            llm_source_finetuned: TextLM | None = None
            if steer_mode:
                # steer never touches weights: it needs a live finetuned source
                # model to run forward passes through during feature collection.
                # Align the checkpoint first and verify it actually changed
                # something -- a silent strict=False no-op would leave the
                # "finetuned" source identical to the pretrained one, making
                # delta_A zero and every steer result meaningless.
                llm_source_finetuned = deepcopy(llm_source)
                source_model_sd = llm_source_finetuned.model.state_dict()
                aligned = _load_tuned_source_state_dict(
                    ckpt_path, base_sd=source_model_sd, source_cfg=source_cfg, model_kind=model_kind
                )
                # Count differences *before* loading: state_dict() hands back
                # references to the live parameters, which load_into_model
                # overwrites in place.
                changed = sum(
                    1
                    for k, v in aligned.items()
                    if not torch.equal(
                        v.detach().cpu().to(source_model_sd[k].dtype), source_model_sd[k].detach().cpu()
                    )
                )
                if changed == 0:
                    raise ValueError(
                        f"Tuned checkpoint for task '{task}' is identical to the source base model "
                        f"({len(aligned)} keys aligned, none differ): {ckpt_path}."
                    )
                load_into_model(llm_source_finetuned.model, aligned, strict=False)
                if source_task_heads is not None:
                    _inject_task_head(
                        model=llm_source_finetuned.model,
                        task=task,
                        task_heads=source_task_heads,
                        head_key_pattern=head_key_pattern,
                        head_class_ids=head_class_ids,
                    )
                print(f"  {task}: source finetuned model loaded ({len(aligned)} keys, {changed} differ from base)")
                tuned_sd: dict[str, torch.Tensor] = {}
                task_delta: dict[str, torch.Tensor] = {}
            else:
                aligned = _load_tuned_source_state_dict(
                    ckpt_path, base_sd=source_base_sd, source_cfg=source_cfg, model_kind=model_kind
                )
                tuned_sd = to_cpu_fp32(aligned)
                task_delta = TaskVector.from_checkpoints(
                    source_base_sd, tuned_sd, strict=False, key_filter=delta_key_filter
                ).delta
                matched, total, unmatched = describe_key_coverage(task_delta, target_base_sd)
                print(f"Loaded tuned checkpoint for '{task}' ({len(tuned_sd)} keys)")
                if shim_mode:
                    # theseus/bico transport across shape-mismatched same-name
                    # layers via their own per-layer activation alignment, so
                    # a low (even zero) exact-match count here is expected on
                    # a cross-width pair and does not mean nothing will
                    # transport -- see "matched" below only as "transported
                    # without any alignment step".
                    print(
                        f"  {task}: delta has {total} keys, {matched} match the target base by name AND shape "
                        f"exactly (transported as-is); the rest are aligned per-layer by {method_name} if the "
                        "names still match"
                    )
                else:
                    print(f"  {task}: delta has {total} keys, {matched} match the target base by name and shape")
                if unmatched:
                    print(f"  {task}: first unmatched delta keys: {unmatched}")

            if eval_source_finetuned:
                if source_loaders is None:
                    raise RuntimeError("eval_source_finetuned requires source loaders.")
                # steer already built a live finetuned A; every other method only
                # needs A's weights as a state dict, so build one here just for
                # this control.
                a_finetuned = llm_source_finetuned
                if a_finetuned is None:
                    a_finetuned = deepcopy(llm_source)
                    a_aligned = _load_tuned_source_state_dict(
                        ckpt_path, base_sd=a_finetuned.model.state_dict(), source_cfg=source_cfg, model_kind=model_kind
                    )
                    load_into_model(a_finetuned.model, a_aligned, strict=False)
                    if source_task_heads is not None:
                        _inject_task_head(
                            model=a_finetuned.model,
                            task=task,
                            task_heads=source_task_heads,
                            head_key_pattern=head_key_pattern,
                            head_class_ids=head_class_ids,
                        )
                print(f"\n--- Source control: evaluating A's own finetuned checkpoint on '{task}' ---")
                source_eval_rows.append(
                    _evaluate_source_finetuned(
                        llm_source_finetuned=a_finetuned,
                        llm_source_pretrained=llm_source,
                        source_loaders=source_loaders,
                        task=task,
                        device=device,
                        eval_mode=eval_mode,
                        ckpt_path=ckpt_path,
                    )
                )
                if a_finetuned is not llm_source_finetuned:
                    del a_finetuned

            if eval_mode == "head_logits" and target_task_heads is not None and task in target_task_heads:
                # steer_text reads w_b off the live head at prepare() time, so the
                # task head must already be in place before the method runs.
                # Under linear_probe_head, the head hasn't been trained yet for
                # this task (target_task_heads starts empty), so this is skipped
                # and the model's own from-scratch random init stays in place --
                # exactly the starting point linear probing is supposed to fit.
                _inject_task_head(
                    model=llm_target.model,
                    task=task,
                    task_heads=target_task_heads,
                    head_key_pattern=head_key_pattern,
                    head_class_ids=head_class_ids,
                )

            print(f"\n--- Transporting '{task}' with method '{method.name}' ---")
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            prepare_started = time.perf_counter()

            if method_name == "gradfix":
                grad_loader = loaders.train
                if grad_examples_per_class is not None:
                    indices = balanced_indices(
                        loaders.local_labels["train"], grad_examples_per_class, seed=seed
                    )
                    grad_loader = subset_loader(loaders.train, indices, batch_size=grad_batch_size)
                elif grad_batch_size is not None:
                    grad_loader = DataLoader(
                        loaders.train.dataset,
                        batch_size=int(grad_batch_size),
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=loaders.train.collate_fn,
                    )
                print(f"  {task}: grad dataloader - {len(grad_loader.dataset)} samples, batch_size={grad_loader.batch_size}")

                if eval_mode == "head_logits":
                    recipe = seq_classification_recipe(
                        device=device,
                        mask_class=loaders.mask_class,
                        reduction="none" if str(method_params.get("vote", "mean")) in {"majority", "max"} else "mean",
                    )
                else:
                    recipe = causal_lm_recipe(llm_target.tokenizer, device=device)
                prepared = method.prepare(
                    target_model=llm_target.model,
                    target_dataloader=grad_loader,
                    recipe=recipe,
                    device=device,
                    **method_params,
                )
            elif shim_mode:
                if source_loaders is None:
                    raise RuntimeError(f"Method '{method_name}' requires source loaders.")
                source_model_for_method = deepcopy(llm_source.model)
                target_model_for_method = deepcopy(llm_target.model)
                load_into_model(source_model_for_method, source_base_sd, strict=False)
                load_into_model(target_model_for_method, target_base_sd, strict=False)
                source_shim = TextEncoderShim(source_model_for_method, llm_source.tokenizer.pad_token_id or 0)
                target_shim = TextEncoderShim(target_model_for_method, llm_target.tokenizer.pad_token_id or 0)

                prepare_kwargs: dict[str, Any] = {
                    "source_model": source_shim,
                    "target_model": target_shim,
                    "source_dataloader": alias_inputs_loader(source_loaders.train),
                    "target_dataloader": alias_inputs_loader(loaders.train),
                    "target_base": target_base_sd,
                    "delta": task_delta,
                    "device": device,
                }
                if bico_mode:
                    # bico drives its own forward through the recipe, so the
                    # attention_mask and labels reach the model correctly.
                    if eval_mode == "head_logits":
                        prepare_kwargs["source_recipe"] = seq_classification_recipe(
                            device=device, mask_class=source_loaders.mask_class
                        )
                        prepare_kwargs["target_recipe"] = seq_classification_recipe(
                            device=device, mask_class=loaders.mask_class
                        )
                    else:
                        prepare_kwargs["source_recipe"] = causal_lm_recipe(llm_source.tokenizer, device=device)
                        prepare_kwargs["target_recipe"] = causal_lm_recipe(llm_target.tokenizer, device=device)

                prepared = method.prepare(**prepare_kwargs, **method_params)
                del source_model_for_method, target_model_for_method, source_shim, target_shim
            elif steer_mode:
                if source_loaders is None or llm_source_finetuned is None:
                    raise RuntimeError("steer_text requires source loaders and a finetuned source model.")
                steer_params = dict(method_params)
                steer_seed = int(steer_params.pop("seed", seed))
                if cfg.get("feature_cache_dir") is not None:
                    steer_params["feature_cache_dir"] = str(cfg["feature_cache_dir"])
                prepared = method.prepare(
                    llm_source=llm_source_finetuned,
                    llm_source_pretrained=llm_source,
                    llm_target=llm_target,
                    source_loaders=source_loaders,
                    target_loaders=loaders,
                    task=task,
                    mask_class=loaders.mask_class,
                    device=device,
                    source_tag=source_tag,
                    target_tag=target_tag,
                    seed=steer_seed,
                    **steer_params,
                )
                steer_prepared_by_task[task] = prepared
            else:
                prepared = None

            prepare_seconds = time.perf_counter() - prepare_started
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
                peak_memory_bytes = float(torch.cuda.max_memory_allocated())
            else:
                peak_memory_bytes = 0.0

            transport_started = time.perf_counter()
            transported_delta = method.transport(
                source_base=source_base_sd,
                target_base=target_base_sd,
                delta=task_delta,
                strict=strict_load,
                prepared=prepared,
                **method_params,
            )
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
            transport_timings[task] = {
                "prepare_seconds": prepare_seconds,
                "transport_seconds": time.perf_counter() - transport_started,
                "peak_memory_allocated_bytes": peak_memory_bytes,
            }
            transported_deltas.append(transported_delta)
            original_deltas.append(task_delta)
            print(f"  {task}: transported delta computed for {len(transported_delta)} params")
            run_logger.log_event(
                "transport_task_end",
                metrics={f"rebase/{task}/transported_param_count": float(len(transported_delta))},
                context={"task": task, "method": method.name},
            )

            if linear_probe_head:
                probe_shots = method_params.get("shots_per_class") if shim_mode else method_params.get("few_shot")
                if probe_shots is None:
                    raise ValueError(
                        f"linear_probe_head requires method_params."
                        f"{'shots_per_class' if shim_mode else 'few_shot'} to be set -- the probe trains on "
                        "the exact same support set (same count, same seed) as the rebasin transport itself."
                    )
                probe_indices = balanced_indices(loaders.local_labels["train"], int(probe_shots), seed=seed)
                # Mini-batched at the run's own batch_size, not one giant batch of
                # the whole support set -- shots_per_class=300 x 3 classes is 900
                # examples, which OOMs a large target model in a single forward pass
                # regardless of how small batch_size is set elsewhere.
                probe_loader = subset_loader(loaders.train, probe_indices, batch_size=batch_size)
                print(f"  {task}: linear-probing the target head from scratch on {len(probe_indices)} support examples")

                # "support" is the probe's own training set: if that one doesn't
                # rise, the probe simply isn't training (epochs/lr), independently
                # of anything the transport/correction did upstream. val/test are
                # scored under the same condition the probe was fit in (for
                # steer_text, that means with the correction hook active), so they
                # are not the same numbers as the final baseline/rebased table.
                probe_eval_loaders = {"support": probe_loader, "val": loaders.val, "test": loaders.test}
                probe_alpha = float(cfg.get("alpha", 1.0))
                if steer_mode:
                    load_into_model(llm_target.model, target_base_sd, strict=strict_load)
                    with steer_text_correction_context(llm_target, prepared, alpha=probe_alpha):
                        trained_head_sd = train_linear_probe_head(
                            llm_target.model,
                            probe_loader,
                            device=device,
                            mask_class=loaders.mask_class,
                            lr=linear_probe_lr,
                            steps=linear_probe_epochs,
                            eval_loaders=probe_eval_loaders,
                            log_every=linear_probe_log_every,
                            log_prefix=f"  [probe:{task}]",
                        )
                else:
                    probe_backbone_sd = axpy_state_dict(target_base_sd, transported_delta, alpha=probe_alpha)
                    load_into_model(llm_target.model, probe_backbone_sd, strict=strict_load)
                    del probe_backbone_sd
                    trained_head_sd = train_linear_probe_head(
                        llm_target.model,
                        probe_loader,
                        device=device,
                        mask_class=loaders.mask_class,
                        lr=linear_probe_lr,
                        steps=linear_probe_epochs,
                        eval_loaders=probe_eval_loaders,
                        log_every=linear_probe_log_every,
                        log_prefix=f"  [probe:{task}]",
                    )
                # Ship the identity intermediate layers alongside the trained
                # final linear, exactly as build_nearest_mean_head.py does, so
                # the per-evaluation head injection restores the same feature
                # space the probe (and steer's correction) were fit in.
                target_task_heads[task] = {**trained_head_sd, **neutralized_head_layers}

            save_transport_dir = cfg.get("save_transported_tvs_dir", None)
            if save_transport_dir and not steer_mode:
                os.makedirs(save_transport_dir, exist_ok=True)
                native_path = os.path.join(save_transport_dir, f"{task}_{method.name}_transported.pt")
                torch.save(to_cpu_fp32(transported_delta), native_path)
                print(f"  {task}: saved transported TV -> {native_path}")
                transported_artifacts[task] = [native_path]

        rebased_deltas = [_scale_delta(d, w) for d, w in zip(transported_deltas, merge_weights, strict=True)]
        untransported_deltas = [_scale_delta(d, w) for d, w in zip(original_deltas, merge_weights, strict=True)]
        print(f"Prepared {len(tasks)} transported deltas (task-independent alpha mode)")

        can_eval_untransported_by_task: list[bool] = []
        for task_name, delta_sd in zip(tasks, untransported_deltas, strict=True):
            if steer_mode:
                # steer_text never produces a weight-space delta, so there is no
                # "untransported" baseline concept -- always fall back to target
                # zeroshot (uncorrected B), which is the correct baseline
                # semantics for feature-space steering.
                can_eval_untransported_by_task.append(False)
                print(f"Untransported baseline for '{task_name}': not applicable (steer_text method).")
                continue
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
            loaders_obj: TextLoaders = item["loaders"]
            if eval_mode == "head_logits":
                # Re-injected on every call: each alpha step loads a fresh state
                # dict into the target, which restores the base (untrained) head.
                _inject_task_head(
                    model=llm_target.model,
                    task=item["task"],
                    task_heads=target_task_heads,
                    head_key_pattern=head_key_pattern,
                    head_class_ids=item["head_class_ids"],
                )
                return float(
                    llm_target.sequence_classification_accuracy(
                        _resolve_eval_loader(loaders_obj, split),
                        device=device,
                        mask_class=loaders_obj.mask_class,
                    )
                )
            return float(
                llm_target.nli_accuracy(
                    examples=loaders_obj.examples[split],
                    label_texts=item["label_texts"],
                    prompt_template=item["prompt_template"],
                    device=device,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )

        def _eval_all_tasks(split: str) -> list[float]:
            return [_eval_task(item, split) for item in per_task]

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

        def _load_into_target_model(sd: dict[str, torch.Tensor]) -> None:
            load_into_model(llm_target.model, sd, strict=strict_load)

        def _eval_zeroshot_all_tasks(split: str) -> list[float]:
            if split not in baseline_cache_zeroshot:
                _load_into_target_model(target_base_sd)
                baseline_cache_zeroshot[split] = _eval_all_tasks(split)
            return list(baseline_cache_zeroshot[split])

        def _eval_baseline_task(split: str, idx: int, alpha: float) -> float:
            if can_eval_untransported_by_task[idx]:
                baseline_sd = axpy_state_dict(target_base_sd, untransported_deltas[idx], alpha=float(alpha))
                _load_into_target_model(baseline_sd)
                del baseline_sd
                return _eval_task(per_task[idx], split)
            return _eval_zeroshot_all_tasks(split)[idx]

        def _eval_baseline_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            return {idx: _eval_baseline_task(split, idx, alpha) for idx in indices}

        def _eval_rebased_task_indices(split: str, indices: list[int], alpha: float) -> dict[int, float]:
            if steer_mode:
                # No weight-space delta to axpy in: reset to the plain target
                # base, then apply the learned correction as a forward-pass hook
                # for the duration of each task's evaluation.
                _load_into_target_model(target_base_sd)
                out: dict[int, float] = {}
                for idx in indices:
                    task_name = per_task[idx]["task"]
                    with steer_text_correction_context(
                        llm_target, steer_prepared_by_task[task_name], alpha=float(alpha)
                    ):
                        out[idx] = _eval_task(per_task[idx], split)
                return out

            out = {}
            for idx in indices:
                rebase_sd_task = axpy_state_dict(target_base_sd, rebased_deltas[idx], alpha=float(alpha))
                _load_into_target_model(rebase_sd_task)
                del rebase_sd_task
                out[idx] = _eval_task(per_task[idx], split)
            return out

        if alpha_selection == "shared":
            best_rebase_avg = float("-inf")
            best_baseline_avg = float("-inf")
            best_alpha = float(positive_alphas[0] if positive_alphas else alphas[0])
            has_untransported = any(can_eval_untransported_by_task)
            best_baseline_alpha = (
                float(positive_alphas[0] if positive_alphas else alphas[0]) if has_untransported else 0.0
            )
            sweep_results: list[dict[str, Any]] = []
            shared_bad_steps = 0

            for alpha in alphas:
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: shared) ===")

                idxs = list(range(len(per_task)))
                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, idxs, float(alpha))
                rebase_by_idx = _eval_rebased_task_indices(alpha_search_split, idxs, float(alpha))
                baseline_accs = [baseline_by_idx[i] for i in idxs]
                rebase_accs = [rebase_by_idx[i] for i in idxs]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for i, item in enumerate(per_task):
                    norm = _norm_acc(rebase_accs[i], baseline_accs[i])
                    print(
                        f"  {item['task']:<{task_col}}  {baseline_accs[i]:>{metric_col}.6f}  "
                        f"{rebase_accs[i]:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                    )

                avg_rebase = average_scores(rebase_accs)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                sweep_results.append(
                    {"alpha": float(alpha), "baseline_accs": baseline_accs, "rebase_accs": rebase_accs}
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
                    if has_untransported:
                        avg_baseline_for_track = float("-inf") if avg_baseline != avg_baseline else float(avg_baseline)
                        if avg_baseline_for_track > best_baseline_avg + eps:
                            best_baseline_avg = avg_baseline_for_track
                            best_baseline_alpha = float(alpha)

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
                print(
                    f"  alpha={r['alpha']:.3f}  {baseline_label}={_average_defined(r['baseline_accs']):.6f}  "
                    f"{result_label}={average_scores(r['rebase_accs']):.6f}"
                )
            print(
                f"\nBest alpha: rebase={best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f}) | "
                f"baseline={best_baseline_alpha:.3f} (avg baseline val acc={best_baseline_avg:.6f})"
            )

            print(
                f"\n(Re-running on test split: rebase at alpha={best_alpha:.3f}, baseline at alpha={best_baseline_alpha:.3f})"
            )
            all_indices = list(range(len(per_task)))
            baseline_test_by_idx = _eval_baseline_task_indices("test", all_indices, float(best_baseline_alpha))
            rebase_test_by_idx = _eval_rebased_task_indices("test", all_indices, float(best_alpha))
            baseline_test_accs = [baseline_test_by_idx[i] for i in all_indices]
            rebase_test_accs = [rebase_test_by_idx[i] for i in all_indices]
            selected_alpha_by_task = [float(best_alpha)] * len(per_task)
            selected_baseline_alpha_by_task = [float(best_baseline_alpha)] * len(per_task)

        else:
            tracker = PerTaskAlphaTracker(
                task_names=[str(item["task"]) for item in per_task],
                initial_alpha=float(positive_alphas[0] if positive_alphas else alphas[0]),
                patience=alpha_patience,
            )
            # For tasks where the untransported baseline is infeasible, the
            # baseline is target_zeroshot and alpha-independent. Pre-seed the
            # secondary tracker so it never participates in alpha optimization.
            for idx in range(len(per_task)):
                if not can_eval_untransported_by_task[idx]:
                    tracker.best_secondary_alpha[idx] = 0.0
                    tracker.best_secondary_acc[idx] = _eval_baseline_task(alpha_search_split, idx, 0.0)
                    tracker.secondary_active[idx] = False
            sweep_results = []

            for alpha in alphas:
                eval_indices = tracker.eval_active_indices()
                if not eval_indices:
                    print("\nAll tasks have early-stopped on both streams; ending per-task alpha sweep.")
                    break

                primary_indices = set(tracker.primary_active_indices())
                print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: per_task) ===")

                baseline_by_idx = _eval_baseline_task_indices(alpha_search_split, eval_indices, float(alpha))
                rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                rebase_by_idx_active = _eval_rebased_task_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                rebase_by_idx: dict[int, float] = {
                    idx: (rebase_by_idx_active[idx] if idx in primary_indices else float("-inf"))
                    for idx in eval_indices
                }

                baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                print(
                    f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                    best_secondary = float(tracker.best_secondary_acc[idx])
                    norm_baseline = best_secondary if best_secondary != float("-inf") else baseline_acc
                    if idx in primary_indices:
                        display_rebase = rebase_acc
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        display_rebase = frozen if frozen != float("-inf") else 0.0
                    marker = " " if idx in primary_indices else "*"
                    print(
                        f" {marker}{per_task[idx]['task']:<{task_col - 1}}  {baseline_acc:>{metric_col}.6f}  "
                        f"{display_rebase:>{metric_col}.6f}  {_norm_acc(display_rebase, norm_baseline):>{metric_col}.6f}"
                    )

                all_rebase_vals: list[float] = []
                for idx in range(len(per_task)):
                    if idx in primary_indices and idx in rebase_by_idx:
                        all_rebase_vals.append(float(rebase_by_idx[idx]))
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        all_rebase_vals.append(frozen if frozen != float("-inf") else 0.0)
                avg_rebase = average_scores(all_rebase_vals)
                avg_baseline = _average_defined(baseline_accs)
                avg_norm = _average_defined(
                    [
                        _norm_acc(
                            float(rebase_by_idx[idx])
                            if idx in primary_indices
                            else max(float(tracker.best_primary_acc[idx]), 0.0),
                            baseline_accs[i],
                        )
                        for i, idx in enumerate(eval_indices)
                    ]
                )
                print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
                print(
                    f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
                )

                stopped_primary: list[int] = []
                stopped_secondary: list[int] = []
                if float(alpha) > 0.0:
                    stopped_primary, stopped_secondary = tracker.update(
                        alpha=float(alpha),
                        indices=eval_indices,
                        primary_accs=rebase_accs,
                        secondary_accs=baseline_accs,
                    )
                    if stopped_primary:
                        names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                        print(f"  Early-stopping REBASED tasks at alpha={alpha:.3f}: {names}")
                    if stopped_secondary:
                        names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_secondary)
                        print(f"  Early-stopping BASELINE tasks at alpha={alpha:.3f}: {names}")

                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(alpha),
                        "alpha/avg_acc": float(avg_rebase),
                        "alpha/avg_norm_acc": float(avg_norm),
                    },
                    context={
                        "active_tasks": [per_task[idx]["task"] for idx in eval_indices],
                        "per_task_baseline": {per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in eval_indices},
                        "per_task_rebased": {per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in eval_indices},
                        "stopped_primary": [int(idx) for idx in stopped_primary],
                        "stopped_secondary": [int(idx) for idx in stopped_secondary],
                    },
                )
                sweep_results.append(
                    {
                        "alpha": float(alpha),
                        "active_indices": list(eval_indices),
                        "baseline_accs": baseline_accs,
                        "rebase_accs": rebase_accs,
                    }
                )

            print("\n=== Alpha search summary (per-task) ===")
            for idx, item in enumerate(per_task):
                print(
                    f"  {item['task']}: rebase_alpha={tracker.best_primary_alpha[idx]:.3f}  "
                    f"rebase_val={tracker.best_primary_acc[idx]:.6f} | "
                    f"baseline_alpha={tracker.best_secondary_alpha[idx]:.3f}  "
                    f"baseline_val={tracker.best_secondary_acc[idx]:.6f}"
                )
            print(f"\nAvg per-task best rebase val acc: {tracker.best_avg():.6f}")

            print("\n(Re-running per-task best alphas on test split — decoupled per stream)")
            baseline_test_accs = []
            rebase_test_accs = []
            selected_alpha_by_task = []
            selected_baseline_alpha_by_task = []
            for idx, item in enumerate(per_task):
                rebase_alpha = float(tracker.best_primary_alpha[idx])
                baseline_alpha = float(tracker.best_secondary_alpha[idx])
                selected_alpha_by_task.append(rebase_alpha)
                selected_baseline_alpha_by_task.append(baseline_alpha)
                print(f"  {item['task']}: rebase_alpha={rebase_alpha:.3f}  baseline_alpha={baseline_alpha:.3f}")
                baseline_test_accs.append(_eval_baseline_task("test", idx, baseline_alpha))
                # Routed through _eval_rebased_task_indices on purpose: the vision
                # twin inlines an axpy here, which silently evaluates the
                # uncorrected target for steer (whose delta is empty).
                rebase_test_accs.append(_eval_rebased_task_indices("test", [idx], rebase_alpha)[idx])
            best_alpha = float(sum(selected_alpha_by_task) / max(1, len(selected_alpha_by_task)))
            best_baseline_alpha = float(
                sum(selected_baseline_alpha_by_task) / max(1, len(selected_baseline_alpha_by_task))
            )

        norm_accs = [_norm_acc(r, b) for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

        pretty_print_task_accuracies(
            suite_name,
            f"{method_label}, alpha={alpha_selection}",
            f"A={source_cfg.model_name_or_path} → B={target_cfg.model_name_or_path}",
            per_task,
            rebase_test_accs,
            norm_accs,
            single_accs=baseline_test_accs,
            baseline_label=baseline_label,
            result_label=result_label,
        )

        if alpha_selection == "per_task":
            print("\nSelected test-time alpha by task:")
            for item, r_a, b_a in zip(per_task, selected_alpha_by_task, selected_baseline_alpha_by_task, strict=True):
                print(f"  {item['task']}: rebase={r_a:.3f}  baseline={b_a:.3f}")

        final_summary = {
            "suite": suite_name,
            "tasks": tasks,
            "method": method.name,
            "method_label": method_label,
            "eval_mode": eval_mode,
            "source_model": source_cfg.model_name_or_path,
            "target_model": target_cfg.model_name_or_path,
            "alpha_selection": alpha_selection,
            "best_alpha": float(best_alpha),
            "best_baseline_alpha": float(best_baseline_alpha),
            "baseline_label": baseline_label,
            "metric_definitions": {
                "absolute_accuracy": "top-1 accuracy in [0, 1] (rebased/transported at the rebased's own best alpha)",
                "baseline_accuracy": "untransported baseline top-1 at the baseline's own best alpha",
                "normalized_accuracy_ratio": (
                    "absolute_accuracy (at rebased best alpha) / baseline_accuracy (at baseline best alpha); "
                    "each stream independently optimizes alpha on the alpha-search split"
                ),
                "normalized_accuracy_ratio_display": (
                    "ratio (decimal, not a percentage); values above 1.0 indicate the rebased/transported "
                    "model exceeds the untransported baseline; report as a decimal ratio, never multiplied by 100"
                ),
            },
            "test_results": {
                "per_task_baseline_accuracy": {
                    item["task"]: float(baseline_test_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_absolute_accuracy": {
                    item["task"]: float(rebase_test_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_normalized_accuracy_ratio": {
                    item["task"]: float(norm_accs[i]) for i, item in enumerate(per_task)
                },
                "per_task_baseline": {item["task"]: float(baseline_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_rebased": {item["task"]: float(rebase_test_accs[i]) for i, item in enumerate(per_task)},
                "per_task_norm": {item["task"]: float(norm_accs[i]) for i, item in enumerate(per_task)},
                "avg_rebased": float(sum(rebase_test_accs) / len(rebase_test_accs)),
                "avg_norm": float(sum(norm_accs) / len(norm_accs)),
            },
            "selected_alpha_by_task": {item["task"]: float(selected_alpha_by_task[i]) for i, item in enumerate(per_task)},
            "selected_baseline_alpha_by_task": {
                item["task"]: float(selected_baseline_alpha_by_task[i]) for i, item in enumerate(per_task)
            },
            "steer_diagnostics": {t: p.get("diagnostics", {}) for t, p in steer_prepared_by_task.items()},
            "source_finetuned_control": source_eval_rows,
            "transported_artifacts": transported_artifacts,
            "transport_timings": transport_timings,
            "saved_merged_path": None,
        }
        run_logger.log_summary(final_summary)
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
