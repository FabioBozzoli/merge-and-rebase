"""
Thin adapters that let the *existing* rebase methods run on HuggingFace text
models without touching a single line under ``rebase/methods/``.

Three of the four methods the text orchestrator drives need nothing at all or
almost nothing:

- ``gradfix`` is already model-agnostic (``rebase/methods/gradfix.py``): every
  model-specific decision lives in the injected ``GradRecipe``, and
  ``models/grad_recipes.py`` already ships ``causal_lm_recipe`` and
  ``seq_classification_recipe`` for HF dict batches.
- ``theseus`` and ``bico`` are *mostly* agnostic. ``theseus._visual_module``,
  ``_visual_state_dict`` and ``_visual_delta_keys`` all fall back to "the whole
  model / the whole state dict" when there is no ``visual.`` prefix, and
  ``_has_fused_mha`` is False on HF attention (already split q/k/v), so the
  OpenCLIP qkv split/merge is a no-op. What they *do* assume is an image-shaped
  forward: ``theseus._encode_image`` calls ``model(images)`` positionally, and
  ``theseus._extract_model_inputs`` only looks for image-ish batch keys.

:class:`TextEncoderShim` and :func:`alias_inputs_loader` close exactly that gap.

Importing this module also applies :func:`_patch_theseus_bico_embedding_hooks`,
a small runtime patch (not a file edit) needed because ``theseus``/``bico``'s
activation hooks assume every parameterized submodule captures a
``[batch, token, hidden]`` tensor. T5's relative-position bias is an
``nn.Embedding`` that breaks that assumption; see the function's docstring.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Head roots emitted by finetune/train_text.py (kept in sync with its
# _detect_head_roots); ordered by preference.
_HEAD_ROOTS: tuple[str, ...] = ("score", "classifier", "classification_head")

# Parameter-name fragments that never belong in a transported task vector.
_NEVER_TRANSPORT: tuple[str, ...] = ("position_ids", "num_batches_tracked", "rotary_emb.inv_freq")


class TextEncoderShim(nn.Module):
    """Present an HF text model to ``theseus``/``bico`` as if it were a CLIP model.

    Two attributes carry the whole trick:

    ``visual``
        ``theseus._visual_module(model)`` returns ``model.visual`` when present.
        Pointing it at the *unwrapped* HF model means the activation hooks in
        ``theseus._ActivationHook`` / ``bico._BiCoHook`` register on the HF
        submodules and are keyed by their plain names
        (``encoder.block.0.layer.0.SelfAttention.q``) -- exactly the names used
        by ``state_dict()``, by ``target_base`` and by the delta. Wrapping the
        model as a normal submodule instead would prefix every hook key and
        silently match nothing in ``_precompute_transforms``.

    ``encode_image``
        ``theseus._encode_image`` prefers it over a positional ``model(x)`` call,
        so this is where the ``attention_mask`` that a positional call would
        drop gets rebuilt from the pad id.

    ``forward`` stays a plain delegation because ``bico`` runs its forward
    through the injected recipe, which calls ``model(input_ids=..., labels=...)``.
    """

    def __init__(self, model: nn.Module, pad_token_id: int) -> None:
        super().__init__()
        self.visual = model
        self.pad_token_id = int(pad_token_id)

    def encode_image(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        out = self.visual(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", None)
        if logits is not None:
            return logits
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.visual(*args, **kwargs)


def alias_inputs_loader(loader: DataLoader) -> DataLoader:
    """Re-emit ``loader``'s batches with an extra ``"inputs"`` alias for ``input_ids``.

    ``theseus._extract_model_inputs`` accepts the key ``"inputs"`` already, so
    this single alias satisfies both ``theseus``'s ``_encode_image`` feed and
    ``bico``'s batch-size assertion (``bico.py`` extracts the tensor purely to
    compare ``shape[0]``, then discards it) without any change to either file.

    The returned loader keeps ``.dataset``, ``.batch_size`` and ``.collate_fn``,
    which is what ``theseus._iter_random_dataset_batches`` needs to build paired
    source/target batches over the *same* dataset indices.
    """
    inner = loader.collate_fn
    if inner is None:
        raise ValueError("alias_inputs_loader expects a loader with an explicit collate_fn.")

    def _collate(batch: list[Any]) -> dict[str, Any]:
        out = dict(inner(batch))
        out["inputs"] = out["input_ids"]
        return out

    return DataLoader(
        loader.dataset,
        batch_size=int(loader.batch_size or 1),
        shuffle=False,
        num_workers=int(getattr(loader, "num_workers", 0)),
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=bool(getattr(loader, "drop_last", False)),
        collate_fn=_collate,
    )


def attach_local_labels(dataset: Any, local_labels: Sequence[int]) -> None:
    """Expose per-example labels to ``theseus``/``bico``'s ``shots_per_class`` path.

    ``theseus._dataset_labels`` reads either an ``HFVisionDataset``-shaped
    ``.split``/``.label_key`` pair or a ``TensorDataset``-shaped ``.tensors``
    tuple whose second entry is the labels. Setting ``.tensors`` on the
    tokenized NLI dataset instance hits the second branch.

    The labels must be the *local* ``0..K-1`` ids, not the ones remapped into
    the shared head's class-id space: ``_class_balanced_indices`` iterates
    ``range(labels.max() + 1)`` and raises on an empty class, which is exactly
    what ``head_class_ids=[0, 2]`` (qnli, rte) would produce for class 1.
    """
    labels = torch.as_tensor([int(y) for y in local_labels], dtype=torch.long)
    if len(labels) != len(dataset):
        raise ValueError(f"local_labels has {len(labels)} entries but dataset has {len(dataset)}.")
    dataset.tensors = (torch.zeros(len(labels)), labels)


def balanced_indices(labels: Sequence[int], per_class: int, *, seed: int = 42) -> list[int]:
    """``eval.utils.balanced_sample_indices`` for label lists instead of datasets.

    The dataset-shaped original unpacks ``x, y = dataset[i]``, which a tokenized
    NLI dataset (dict items) cannot satisfy. Same sampling semantics: up to
    ``per_class`` random indices per observed class, classes need not be
    contiguous.
    """
    if int(per_class) <= 0:
        raise ValueError("per_class must be > 0.")
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    out: list[int] = []
    for cls in sorted(by_class):
        idxs = by_class[cls]
        out.extend(idxs if len(idxs) <= int(per_class) else rng.sample(idxs, int(per_class)))
    return out


def subset_loader(loader: DataLoader, indices: Sequence[int], *, batch_size: int | None = None) -> DataLoader:
    """A loader over ``Subset(loader.dataset, indices)`` reusing the same collate."""
    return DataLoader(
        Subset(loader.dataset, list(indices)),
        batch_size=int(batch_size or loader.batch_size or 1),
        shuffle=False,
        num_workers=int(getattr(loader, "num_workers", 0)),
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


def _head_root_and_linears(model: nn.Module) -> tuple[str, list[tuple[str, nn.Linear]]]:
    for root in _HEAD_ROOTS:
        module = getattr(model, root, None)
        if module is None:
            continue
        if isinstance(module, nn.Linear):
            return root, [(root, module)]
        linears = [(f"{root}.{n}", m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]
        if linears:
            return root, linears
    raise ValueError(
        "Could not locate a classification head on this model. Expected one of "
        f"{list(_HEAD_ROOTS)} holding an nn.Linear (an AutoModelForSequenceClassification). "
        "Methods needing a head (steer_text) require model_kind='sequence_classification'."
    )


def head_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    """Return ``(qualified_name, module)`` of the final classification ``nn.Linear``.

    Its *input* is, by construction, the pooled sentence feature the classifier
    consumes -- whatever pooling rule the architecture uses internally (T5 takes
    the decoder's eos position, Qwen/Llama the last non-pad token). Capturing it
    with a forward pre-hook is therefore architecture-independent, which is why
    ``steer_text`` never has to reimplement per-model pooling.

    Note this is the input to the *final* Linear only. Some heads (T5's, see
    :func:`head_intermediate_linears`) insert randomly-initialized layers
    before it, so "input to the final Linear" is not the same as "the model's
    own pretrained representation" -- ``steer_text`` doesn't care (it fits and
    evaluates in that same space consistently either way), but a method that
    wants the model's actual pretrained features does.
    """
    _, linears = _head_root_and_linears(model)
    return linears[-1]


def head_intermediate_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every ``nn.Linear`` under the classification head *except* the final one.

    An encoder-decoder head like T5's ``T5ClassificationHead`` is
    ``dense -> tanh -> out_proj``: ``dense`` has no pretrained weights (a
    fresh ``AutoModelForSequenceClassification.from_pretrained`` always
    initializes it randomly -- there is nothing in the base checkpoint to
    load it from), so whatever ``head_linear`` captures as "the pooled
    feature" for T5 is actually that pretrained representation passed
    through an untrained random rotation, not the representation itself. A
    decoder-only head (a bare ``score`` Linear) has no such layer -- this
    returns ``[]`` for it.

    Callers that want to build a classifier directly on the model's own
    pretrained features (e.g. a nearest-mean head) should neutralize these
    to the identity transform first; see ``scripts/build_nearest_mean_head.py``.
    """
    _, linears = _head_root_and_linears(model)
    return linears[:-1]


def text_param_filter(*, exclude_head: bool = True):
    """``key_filter`` for ``TaskVector.from_checkpoints`` on HF text models.

    Twin of ``vision_rebase._visual_only_filter``: float tensors only, minus the
    buffers that must never be transported. The head is excluded by default
    because ``head_logits`` evaluation injects a per-task head from
    ``task_heads`` right before scoring, so a transported head would be
    overwritten anyway -- and its rows live in a task-specific class-id space
    that has no meaning under a different task.
    """

    def _filter(k: str, v: torch.Tensor) -> bool:
        if not torch.is_floating_point(v):
            return False
        if any(frag in k for frag in _NEVER_TRANSPORT):
            return False
        if exclude_head and any(k == root or k.startswith(root + ".") for root in _HEAD_ROOTS):
            return False
        return True

    return _filter


def count_transformer_blocks(model: nn.Module) -> dict[str, int]:
    """Count blocks per stack (``encoder``/``decoder``/``layers``) for the depth report."""
    from .steer_text import block_index_of  # local import: shared regex table

    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, int]] = set()
    for name, _ in model.named_parameters():
        found = block_index_of(name)
        if found is None:
            continue
        stack, index = found
        if (stack, index) not in seen:
            seen.add((stack, index))
            counts[stack] += 1
    return dict(counts)


def describe_key_coverage(delta: Mapping[str, torch.Tensor], target_base: Mapping[str, torch.Tensor]) -> tuple[int, int, list[str]]:
    """``(matched, total, first unmatched keys)`` for the source-delta / target-base overlap."""
    unmatched = [k for k, v in delta.items() if k not in target_base or tuple(target_base[k].shape) != tuple(v.shape)]
    return len(delta) - len(unmatched), len(delta), unmatched[:5]


def _patch_theseus_bico_embedding_hooks() -> None:
    """
    Make ``theseus._ActivationHook`` / ``bico._BiCoHook`` skip ``nn.Embedding``
    submodules when they register activation hooks.

    Both classes hook *every* submodule that owns parameters directly
    (``if list(module.parameters(recurse=False))``), with no assumption
    weaker than "captures a ``[batch, token, hidden]`` tensor" -- see
    ``theseus._standardize_tokens``/``_align_features``. No CLIP submodule
    that logic was written against looks like anything else.

    T5 breaks that assumption: its relative-position bias
    (``SelfAttention.relative_attention_bias``, present once per
    encoder/decoder on block 0) is an ``nn.Embedding`` whose captured
    input/output has no fixed hidden dimension -- its trailing dimension is
    the *token count of the current batch*. Two calibration batches with
    different token counts (ordinary dynamic padding) then produce
    differently-shaped captures for the very same hook key, and
    ``ActivationStore.update``'s cross-batch covariance accumulation
    (``self.at_b += a.T @ b``) crashes with a shape mismatch.

    ``theseus.py``/``bico.py`` are not to be edited, so this patches the two
    hook classes' ``_register_hooks`` in place instead of the file on disk.
    It is applied once, at import time of this module -- i.e. only when
    ``eval/text_rebase.py`` runs; ``eval/vision_rebase.py`` never imports
    ``rebase.text`` and is completely unaffected.
    """
    from ..methods import bico as _bico
    from ..methods import theseus as _theseus

    if getattr(_theseus._ActivationHook, "_skips_embeddings", False):
        return  # idempotent: harmless if this module is imported more than once

    def _register_hooks_no_embeddings(self: Any) -> None:
        self.handles.append(self.model.register_forward_hook(self._make_hook("")))
        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, nn.Embedding):
                continue
            if list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _register_bico_hooks_no_embeddings(self: Any) -> None:
        self._forward_handles.append(self.model.register_forward_hook(self._make_forward_hook("")))
        self._backward_handles.append(self.model.register_full_backward_hook(self._make_backward_hook("")))
        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, nn.Embedding):
                continue
            if list(module.parameters(recurse=False)):
                self._forward_handles.append(module.register_forward_hook(self._make_forward_hook(name)))
                self._backward_handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    _theseus._ActivationHook._register_hooks = _register_hooks_no_embeddings
    _theseus._ActivationHook._skips_embeddings = True
    _bico._BiCoHook._register_hooks = _register_bico_hooks_no_embeddings
    _bico._BiCoHook._skips_embeddings = True


_patch_theseus_bico_embedding_hooks()
