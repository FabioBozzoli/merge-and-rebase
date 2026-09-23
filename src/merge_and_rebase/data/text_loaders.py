from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

NLI_TASKS = ("snli", "mnli", "sick", "qnli", "rte", "scitail")

# Blank-line join shared with lm_eval.models.hf_rebased._calibration_texts.
_CAUSAL_TEXT_JOIN = "\n\n"

# Rows per batched tokenizer call. Bounds the transient Python-list peak.
_TOKENIZE_CHUNK = 1000


@dataclass(frozen=True)
class NLIExample:
    premise: str
    hypothesis: str
    label: int


@dataclass(frozen=True)
class NLITaskData:
    task: str
    examples: list[NLIExample]
    labels: list[str]
    label_texts: list[str]
    meta: dict[str, Any]


@dataclass(frozen=True)
class NLITokenizedData:
    task: str
    loader: DataLoader
    mask_class: list[int]
    meta: dict[str, Any]


@dataclass(frozen=True)
class _TaskSpec:
    hf_path: str
    hf_configs: tuple[str | None, ...]
    split_map: dict[str, tuple[str, ...]]
    premise_keys: tuple[str, ...]
    hypothesis_keys: tuple[str, ...]
    label_keys: tuple[str, ...]
    labels: tuple[str, ...]
    label_texts: tuple[str, ...]
    label_int_map: dict[int, str] | None = None
    label_str_map: dict[str, str] | None = None


_TASK_SPECS: dict[str, _TaskSpec] = {
    "snli": _TaskSpec(
        hf_path="stanfordnlp/snli",
        hf_configs=(None,),
        split_map={"train": ("train",), "validation": ("validation",), "test": ("test",)},
        premise_keys=("premise",),
        hypothesis_keys=("hypothesis",),
        label_keys=("label",),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
    ),
    "mnli": _TaskSpec(
        hf_path="nyu-mll/glue",
        hf_configs=("mnli",),
        split_map={
            "train": ("train",),
            "validation": ("validation_matched", "validation_mismatched", "validation"),
            # GLUE test labels are unavailable; use labeled validation splits for eval.
            "test": ("validation_matched", "validation_mismatched", "validation"),
        },
        premise_keys=("premise",),
        hypothesis_keys=("hypothesis",),
        label_keys=("label",),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
    ),
    "sick": _TaskSpec(
        hf_path="yangwang825/sick",
        hf_configs=(None,),
        split_map={
            "train": ("train",),
            "validation": ("validation", "dev", "trial"),
            "test": ("test",),
        },
        premise_keys=("text1", "sentence_A", "sentence1", "premise"),
        hypothesis_keys=("text2", "sentence_B", "sentence2", "hypothesis"),
        label_keys=("label", "entailment_label"),
        labels=("entailment", "neutral", "contradiction"),
        label_texts=("entailment", "neutral", "contradiction"),
        label_int_map={0: "entailment", 1: "neutral", 2: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "neutral": "neutral",
            "contradiction": "contradiction",
            "entails": "entailment",
        },
    ),
    "qnli": _TaskSpec(
        hf_path="nyu-mll/glue",
        hf_configs=("qnli",),
        # GLUE test labels are unavailable; use validation for eval.
        split_map={"train": ("train",), "validation": ("validation",), "test": ("validation",)},
        premise_keys=("question", "premise"),
        hypothesis_keys=("sentence", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "contradiction"),
        label_texts=("entailment", "contradiction"),
        label_int_map={0: "entailment", 1: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "not_entailment": "contradiction",
            "contradiction": "contradiction",
        },
    ),
    "rte": _TaskSpec(
        hf_path="nyu-mll/glue",
        hf_configs=("rte",),
        # GLUE test labels are unavailable; use validation for eval.
        split_map={"train": ("train",), "validation": ("validation",), "test": ("validation",)},
        premise_keys=("sentence1", "premise"),
        hypothesis_keys=("sentence2", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "contradiction"),
        label_texts=("entailment", "contradiction"),
        label_int_map={0: "entailment", 1: "contradiction"},
        label_str_map={
            "entailment": "entailment",
            "not_entailment": "contradiction",
            "contradiction": "contradiction",
        },
    ),
    "scitail": _TaskSpec(
        hf_path="allenai/scitail",
        hf_configs=("tsv_format", None),
        split_map={"train": ("train",), "validation": ("validation", "dev"), "test": ("test",)},
        premise_keys=("sentence1", "premise"),
        hypothesis_keys=("sentence2", "hypothesis"),
        label_keys=("label",),
        labels=("entailment", "neutral"),
        label_texts=("entailment", "neutral"),
        label_int_map={0: "entailment", 1: "neutral"},
        label_str_map={
            "entails": "entailment",
            "entailment": "entailment",
            "neutral": "neutral",
            "not_entails": "neutral",
        },
    ),
}


def _first_non_empty(ex: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in ex and isinstance(ex[k], str):
            s = ex[k].strip()
            if s:
                return s
    return None


def _norm_label_str(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _to_label_name(raw: Any, spec: _TaskSpec) -> str | None:
    if isinstance(raw, bool):
        raw = int(raw)

    if isinstance(raw, int):
        if spec.label_int_map is not None:
            return spec.label_int_map.get(int(raw), None)
        if 0 <= int(raw) < len(spec.labels):
            return spec.labels[int(raw)]
        return None

    if isinstance(raw, str):
        k = _norm_label_str(raw)
        if spec.label_str_map is not None and k in spec.label_str_map:
            return spec.label_str_map[k]
        if k in spec.labels:
            return k
        return None

    return None


def _load_dataset_with_fallbacks(
    *,
    task: str,
    spec: _TaskSpec,
    split: str,
):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Text task loading requires `datasets` (install with `.[data]`).") from e

    if split not in spec.split_map:
        raise ValueError(f"Unsupported split '{split}' for task '{task}'.")

    errors: list[str] = []
    for cfg in spec.hf_configs:
        for hf_split in spec.split_map[split]:
            try:
                if cfg is None:
                    ds = load_dataset(spec.hf_path, split=hf_split)
                else:
                    ds = load_dataset(spec.hf_path, cfg, split=hf_split)
                return ds, cfg, hf_split
            except Exception as e:  # pragma: no cover - best-effort fallback path
                errors.append(f"path={spec.hf_path}, config={cfg}, split={hf_split}: {type(e).__name__}: {e}")
                continue

    joined = "\n".join(errors[:8])
    raise RuntimeError(f"Failed to load dataset for task '{task}'. Tried:\n{joined}")


def build_nli_task_data(
    *,
    task: str,
    split: str = "validation",
    max_samples: int | None = None,
) -> NLITaskData:
    task_key = str(task).strip().lower()
    if task_key not in _TASK_SPECS:
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {list(NLI_TASKS)}")
    spec = _TASK_SPECS[task_key]

    ds, used_cfg, used_split = _load_dataset_with_fallbacks(task=task_key, spec=spec, split=split)

    label_index = {n: i for i, n in enumerate(spec.labels)}
    rows: list[NLIExample] = []
    skipped = 0

    for ex in ds:
        premise = _first_non_empty(ex, spec.premise_keys)
        hypothesis = _first_non_empty(ex, spec.hypothesis_keys)
        if premise is None or hypothesis is None:
            skipped += 1
            continue

        raw_label = None
        for lk in spec.label_keys:
            if lk in ex:
                raw_label = ex[lk]
                break
        if raw_label is None:
            skipped += 1
            continue

        label_name = _to_label_name(raw_label, spec)
        if label_name is None or label_name not in label_index:
            skipped += 1
            continue

        rows.append(NLIExample(premise=premise, hypothesis=hypothesis, label=label_index[label_name]))

    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    if not rows:
        raise ValueError(
            f"No usable examples loaded for task '{task_key}' (split='{split}'). "
            "Check dataset availability/mapping."
        )

    meta = {
        "task": task_key,
        "hf_path": spec.hf_path,
        "hf_config": used_cfg,
        "hf_split": used_split,
        "num_examples": len(rows),
        "num_skipped": skipped,
        "labels": list(spec.labels),
        "label_texts": list(spec.label_texts),
    }
    return NLITaskData(
        task=task_key,
        examples=rows,
        labels=list(spec.labels),
        label_texts=list(spec.label_texts),
        meta=meta,
    )


class _TokenizedNLIDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], labels: list[int]) -> None:
        if len(features) != len(labels):
            raise ValueError(f"features/labels length mismatch: {len(features)} vs {len(labels)}")
        self.features = features
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = int(idx)
        out = dict(self.features[i])
        out["labels"] = int(self.labels[i])
        return out


def default_head_class_ids_for_task(task: str, num_labels: int) -> list[int]:
    t = str(task).strip().lower()
    if t in {"qnli", "rte"} and int(num_labels) >= 3:
        # binary entailment tasks often use class ids {0,2} in a 3-way head space.
        return [0, 2]
    if t == "scitail" and int(num_labels) >= 2:
        return [0, 1]
    if num_labels == 3:
        return [0, 1, 2]
    return list(range(num_labels))


def build_nli_tokenized_loader(
    *,
    task_data: NLITaskData,
    tokenizer: Any,
    batch_size: int = 8,
    num_workers: int = 0,
    max_length: int = 512,
    shuffle: bool = False,
    head_class_ids: list[int] | None = None,
    premise_hypothesis_template: str | None = None,
) -> NLITokenizedData:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0.")
    if int(max_length) <= 4:
        raise ValueError("max_length must be > 4.")

    mapped_class_ids = (
        list(head_class_ids)
        if head_class_ids is not None
        else default_head_class_ids_for_task(task_data.task, num_labels=len(task_data.labels))
    )
    if len(mapped_class_ids) != len(task_data.labels):
        raise ValueError(
            f"head_class_ids length mismatch for task '{task_data.task}': "
            f"{len(mapped_class_ids)} vs {len(task_data.labels)}"
        )

    premises = [ex.premise for ex in task_data.examples]
    hypotheses = [ex.hypothesis for ex in task_data.examples]
    local_labels = [int(ex.label) for ex in task_data.examples]
    labels = [int(mapped_class_ids[y]) for y in local_labels]

    if premise_hypothesis_template is None:
        enc = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=int(max_length),
            padding=False,
        )
    else:
        # Some checkpoints were fine-tuned on a single formatted string
        # (e.g. "premise: {premise} hypothesis: {hypothesis}") rather than
        # the tokenizer's own two-segment pair encoding above -- the two are
        # different token sequences, and a model trained on one performs at
        # chance on the other (see scripts/probe_nli_input_format.py, which
        # is how a mismatch here gets diagnosed in the first place).
        texts = [
            premise_hypothesis_template.format(premise=p, hypothesis=h)
            for p, h in zip(premises, hypotheses, strict=True)
        ]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=int(max_length),
            padding=False,
        )
    features: list[dict[str, Any]] = []
    n = len(labels)
    for i in range(n):
        feat = {}
        for k, v in enc.items():
            feat[k] = v[i]
        features.append(feat)
    dataset = _TokenizedNLIDataset(features=features, labels=labels)

    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feats = [{k: v for k, v in x.items() if k != "labels"} for x in batch]
        ys = torch.tensor([int(x["labels"]) for x in batch], dtype=torch.long)
        padded = tokenizer.pad(feats, return_tensors="pt")
        padded["labels"] = ys
        return padded

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    mask_class = sorted(set(mapped_class_ids))
    meta = dict(task_data.meta)
    meta.update(
        {
            "num_examples_tokenized": len(dataset),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "shuffle": bool(shuffle),
            "mask_class": list(mask_class),
            "head_class_ids": list(mapped_class_ids),
            "premise_hypothesis_template": premise_hypothesis_template,
        }
    )
    return NLITokenizedData(task=task_data.task, loader=loader, mask_class=mask_class, meta=meta)


# --- Causal-LM (prompt -> response) loading ------------------------------
#
# Deliberately generic: the dataset is described in the training config rather
# than registered here, because the target sets (dart-math, Magicoder, alpaca,
# hellaswag) share one shape and differ only in field names.

# Same rationale and same role as eval.text_rebase._VAL_TEST_SPLIT_SEED.
_CAUSAL_VAL_SPLIT_SEED = 0


@dataclass(frozen=True)
class CausalExample:
    prompt: str
    response: str


@dataclass(frozen=True)
class CausalTaskData:
    task: str
    examples: list[CausalExample]
    meta: dict[str, Any]


@dataclass(frozen=True)
class CausalTokenizedData:
    task: str
    loader: DataLoader
    meta: dict[str, Any]


def _causal_response(row: dict[str, Any], *, response_field: str, response_index_field: str | None) -> str | None:
    raw = row.get(response_field, None)
    if response_index_field is None:
        return str(raw).strip() if isinstance(raw, str) and raw.strip() else None
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        idx = int(str(row.get(response_index_field, "")).strip())
    except (TypeError, ValueError):
        return None
    if not 0 <= idx < len(raw):
        return None
    picked = str(raw[idx]).strip()
    return picked or None


def build_causal_task_data(
    *,
    task: str,
    hf_path: str,
    hf_config: str | None = None,
    split: str = "train",
    prompt_fields: str,
    response_field: str,
    response_index_field: str | None = None,
    max_samples: int | None = None,
) -> CausalTaskData:
    """Rows of (prompt, response) from an arbitrary HF dataset.

    ``prompt_fields`` is ``+``-separated and joined by blank lines, matching
    ``lm_eval.models.hf_rebased._calibration_texts`` exactly: the text a source
    model is trained on here must be the text steer_text later calibrates on,
    or Stage 1 silently reads a different distribution than it was fit for.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Text task loading requires `datasets` (install with `.[data]`).") from e

    names = [n for n in str(prompt_fields).split("+") if n]
    if not names:
        raise ValueError("prompt_fields must name at least one dataset column.")

    # ponytail: streaming + head-of-split, like _calibration_texts. Swap in
    # .shuffle(seed) on a non-streaming load if head-of-split ordering bias shows up.
    rows = load_dataset(hf_path, hf_config, split=split, streaming=True)

    limit = None if max_samples is None else max(0, int(max_samples))
    examples: list[CausalExample] = []
    skipped = 0
    for row in rows:
        if limit is not None and len(examples) >= limit:
            break
        prompt = _CAUSAL_TEXT_JOIN.join(str(row[k]) for k in names if row.get(k))
        response = _causal_response(
            row, response_field=response_field, response_index_field=response_index_field
        )
        if not prompt.strip() or response is None:
            skipped += 1
            continue
        examples.append(CausalExample(prompt=prompt, response=response))

    if not examples:
        raise ValueError(
            f"No usable examples for task '{task}' "
            f"(path={hf_path}, config={hf_config}, split={split}, "
            f"prompt_fields={prompt_fields}, response_field={response_field})."
        )

    meta = {
        "task": task,
        "hf_path": hf_path,
        "hf_config": hf_config,
        "hf_split": split,
        "prompt_fields": str(prompt_fields),
        "response_field": str(response_field),
        "response_index_field": response_index_field,
        "num_examples": len(examples),
        "num_skipped": skipped,
    }
    return CausalTaskData(task=task, examples=examples, meta=meta)


def split_causal_task_data(
    task_data: CausalTaskData,
    *,
    val_fraction: float,
) -> tuple[CausalTaskData, CausalTaskData]:
    """Seeded disjoint (train, val) carve.

    dart-math / Magicoder / alpaca ship a single ``train`` split, so validation
    has to be carved out of it; the seeded permutation keeps the carve identical
    across runs (same pattern as eval.text_rebase._build_task_splits).
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    generator = torch.Generator().manual_seed(_CAUSAL_VAL_SPLIT_SEED)
    perm = torch.randperm(len(task_data.examples), generator=generator).tolist()
    n_val = int(round(float(val_fraction) * len(perm)))
    if n_val <= 0 or n_val >= len(perm):
        raise ValueError(
            f"val_fraction={val_fraction} carves {n_val} of {len(perm)} examples for "
            f"task '{task_data.task}'; raise max_train_samples or val_fraction."
        )

    def _slice(indices: list[int], name: str) -> CausalTaskData:
        meta = dict(task_data.meta)
        meta.update({"split_role": name, "num_examples": len(indices)})
        return CausalTaskData(
            task=task_data.task,
            examples=[task_data.examples[i] for i in indices],
            meta=meta,
        )

    return _slice(sorted(perm[n_val:]), "train"), _slice(sorted(perm[:n_val]), "validation")


class _TokenizedCausalDataset(Dataset):
    """Token ids as int32 arrays, not Python lists.

    591K rows of dart-math-uniform at max_length=1024 is ~13.6 GB as lists of
    Python ints (28 bytes per int plus an 8-byte slot, x2 for labels) and ~1.7 GB
    as int32 -- the difference between OOM-before-step-1 and fitting.
    """

    def __init__(self, rows: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.rows[int(idx)]


class ResumableRandomSampler(Sampler[int]):
    """Seeded per-epoch permutation that can start mid-epoch.

    DataLoader(shuffle=True) draws its order from the global RNG, so a restarted
    job cannot reproduce it. Here epoch ``e`` is ``randperm(n, seed + e)`` and a
    resume skips the consumed prefix by index -- no batch is loaded to be thrown
    away. ``start_index`` must be a multiple of the batch size so the remaining
    batches are the same batches the uninterrupted run would have seen.

    Under data parallelism every rank draws the same permutation and keeps its
    own stride of it, truncated so all ranks get the same number of rows -- the
    ranks must run the same number of optimizer steps or the collectives hang.
    ``start_index`` counts rows within a rank's own shard.
    """

    def __init__(self, num_rows: int, *, seed: int, rank: int = 0, world_size: int = 1) -> None:
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(f"rank={rank} outside [0, world_size={world_size}).")
        self.num_rows = int(num_rows)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.start_index = 0

    @property
    def num_rows_per_rank(self) -> int:
        return self.num_rows // self.world_size

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if not 0 <= int(start_index) <= self.num_rows_per_rank:
            raise ValueError(f"start_index={start_index} outside [0, {self.num_rows_per_rank}].")
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def permutation(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.num_rows, generator=generator).tolist()

    def rank_indices(self) -> list[int]:
        perm = self.permutation()
        usable = self.num_rows_per_rank * self.world_size  # drop the ragged tail
        return perm[:usable][self.rank :: self.world_size]

    def __iter__(self):
        return iter(self.rank_indices()[self.start_index :])

    def __len__(self) -> int:
        return self.num_rows_per_rank - self.start_index


def _ignore_sigusr1(_worker_id: int) -> None:
    """Slurm's pre-walltime SIGUSR1 is for the trainer; its default action
    would kill a DataLoader worker and take the epoch down with it."""
    import signal

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


@dataclass(frozen=True)
class _CausalCollator:
    """Right-pads to the batch max. A module-level class, not a closure, so
    DataLoader workers can pickle it (py3.12+ defaults to forkserver)."""

    pad_id: int

    def __call__(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, torch.Tensor]:
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (ids, ys) in enumerate(batch):
            n = len(ids)
            input_ids[i, :n] = torch.from_numpy(ids.astype(np.int64))
            labels[i, :n] = torch.from_numpy(ys.astype(np.int64))
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_causal_tokenized_loader(
    *,
    task_data: CausalTaskData,
    tokenizer: Any,
    batch_size: int = 1,
    num_workers: int = 0,
    max_length: int = 512,
    shuffle: bool = False,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> CausalTokenizedData:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0.")
    if int(max_length) <= 4:
        raise ValueError("max_length must be > 4.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer exposes neither pad_token_id nor eos_token_id.")
    eos_id = tokenizer.eos_token_id

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    dropped = 0
    limit = int(max_length)
    # Chunked so the tokenizer's Python-list output never materializes for the
    # whole split at once; batched calls also use the fast tokenizer's threads.
    for start in range(0, len(task_data.examples), _TOKENIZE_CHUNK):
        chunk = task_data.examples[start : start + _TOKENIZE_CHUNK]
        # Prompt and response tokenized separately and concatenated:
        # tok(a + b) != tok(a) + tok(b) in general, and the -100 boundary has to
        # land exactly on the join.
        prompt_batch = tokenizer([ex.prompt for ex in chunk], add_special_tokens=True)["input_ids"]
        response_batch = tokenizer(
            [_CAUSAL_TEXT_JOIN + ex.response for ex in chunk], add_special_tokens=False
        )["input_ids"]

        for prompt_ids, response_ids in zip(prompt_batch, response_batch, strict=True):
            n_prompt = len(prompt_ids)
            if n_prompt >= limit:
                # Prompt alone fills max_length: nothing is supervised, so the
                # row would contribute a zero-token loss term.
                dropped += 1
                continue
            if eos_id is not None:
                response_ids = list(response_ids) + [int(eos_id)]
            ids = np.fromiter(
                itertools.chain(prompt_ids, response_ids), dtype=np.int32, count=n_prompt + len(response_ids)
            )[:limit]
            ys = ids.copy()
            ys[:n_prompt] = -100
            rows.append((ids, ys))

    if not rows:
        raise ValueError(
            f"All {len(task_data.examples)} rows of task '{task_data.task}' lost their response to "
            f"truncation at max_length={max_length}."
        )

    # shuffle=True means a ResumableRandomSampler, so a restarted job replays
    # the same order and can skip straight to where the checkpoint left off.
    loader = DataLoader(
        _TokenizedCausalDataset(rows),
        batch_size=int(batch_size),
        sampler=(
            ResumableRandomSampler(len(rows), seed=int(seed), rank=int(rank), world_size=int(world_size))
            if shuffle
            else None
        ),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=_CausalCollator(pad_id=int(pad_id)),
        worker_init_fn=_ignore_sigusr1 if int(num_workers) > 0 else None,
    )

    meta = dict(task_data.meta)
    meta.update(
        {
            "num_examples_tokenized": len(rows),
            "num_dropped_truncated": dropped,
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "shuffle": bool(shuffle),
            "shuffle_seed": int(seed) if shuffle else None,
            "world_size": int(world_size) if shuffle else 1,
        }
    )
    return CausalTokenizedData(task=task_data.task, loader=loader, meta=meta)
