"""
Linear probing on a frozen CLIP backbone, for ``eval/vision_rebase.py``.

The vision twin of ``rebase/text/adapters.py``'s ``train_linear_probe_head``, and
it follows the same two rules that one had to learn the hard way:

1. **Only the classifier is trained.** Here the classifier is the zero-shot head
   ``_zs_text_features`` ``[C, D]`` -- the matrix ``OpenClipClassifier.forward``
   multiplies image features by. Every model parameter is frozen; the head is
   optimized as a standalone tensor.

2. **It starts from the head that is already there**, never from a fresh random
   draw. For ``steer`` that is load-bearing: Stage 1 builds its correction
   through ``pinv(w_b)`` of the target's zero-shot head, so the correction only
   produces the right logit shift through *that* matrix. It also makes epoch 0
   exactly the zero-shot accuracy, which is the number the probe has to beat.

The training forward is literally the evaluation forward (``clf(images)`` with
the head swapped in), so there is no train/eval skew to reason about: whatever
wrapper is active around ``encode_image`` -- notably ``steer_correction_context``
-- applies identically during training and during scoring. The backbone stays in
``eval()`` mode throughout, so a ResNet CLIP's BatchNorm running stats are never
updated by probing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..models.openclip_classifier import normalize_features


def _forward_with_head(clf: Any, images: torch.Tensor, *, head: torch.Tensor) -> torch.Tensor:
    """``clf``'s own forward, with ``head`` swapped in as the zero-shot head.

    Mirrors ``OpenClipClassifier.top1_with_text_features``' swap-and-restore, so
    the row normalization convention is identical to the eval path: rows are
    normalized here, and ``eval_task_top1(..., text_features=...)`` normalizes
    them again on the way out. Only the direction of each row matters, in both.
    """
    previous = clf._zs_text_features
    clf._zs_text_features = normalize_features(head) if clf.normalize else head
    try:
        return clf(images)
    finally:
        clf._zs_text_features = previous


@torch.no_grad()
def _probe_accuracy(clf: Any, loader: DataLoader, *, device: str, head: torch.Tensor) -> float:
    correct = total = 0
    for images, labels in loader:
        logits = _forward_with_head(clf, images.to(device), head=head)
        correct += int((logits.argmax(dim=-1) == labels.to(device)).sum())
        total += int(labels.numel())
    return float(correct) / float(total) if total else float("nan")


def train_zeroshot_head_probe(
    clf: Any,
    loader: DataLoader,
    *,
    device: str,
    lr: float = 1e-3,
    steps: int = 50,
    eval_loaders: Mapping[str, DataLoader] | None = None,
    log_every: int | None = None,
    log_prefix: str = "[probe]",
) -> torch.Tensor:
    """Fit ``clf``'s zero-shot head on ``loader``, backbone frozen.

    ``clf`` must already hold the task's zero-shot head (call
    ``build_zeroshot_text_features`` first): that head is the starting point, and
    is restored on the way out so the classifier is left exactly as handed over.

    Returns the trained ``[C, D]`` head, ready to pass to
    ``eval_task_top1(..., text_features=...)``.

    Pass ``eval_loaders`` (e.g. ``{"support": ..., "val": ..., "test": ...}``) to
    print the mean training loss plus accuracy on each split as it goes;
    ``log_every`` sets the epoch interval (default ~10 lines over the run).
    Epoch 0 is logged before any update -- for a probe that starts from the
    zero-shot head, that line *is* the zero-shot accuracy.
    """
    head = getattr(clf, "_zs_text_features", None)
    if not isinstance(head, torch.Tensor) or head.numel() == 0:
        raise RuntimeError(
            "train_zeroshot_head_probe needs the task's zero-shot head in place: "
            "call clf.build_zeroshot_text_features(...) first."
        )

    batches = list(loader)
    if not batches:
        raise ValueError("train_zeroshot_head_probe got an empty loader.")

    weight = head.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)

    original_requires_grad = {n: p.requires_grad for n, p in clf.named_parameters()}
    for p in clf.parameters():
        p.requires_grad_(False)
    # Never clf.train(): the backbone is frozen, and train mode would let a
    # ResNet CLIP's BatchNorm mutate its running stats from the probe's batches.
    clf.eval()

    total_epochs = int(steps)
    every = int(log_every) if log_every else max(1, total_epochs // 10)

    def _eval_line(header: str) -> str:
        parts = [header]
        for split_name, split_loader in (eval_loaders or {}).items():
            acc = _probe_accuracy(clf, split_loader, device=device, head=weight)
            parts.append(f"{split_name}={acc:.4f}")
        return "  ".join(parts)

    if eval_loaders:
        print(_eval_line(f"{log_prefix} epoch 0/{total_epochs}  (zero-shot head, no update yet)"))

    optimizer = torch.optim.Adam([weight], lr=float(lr))
    try:
        for epoch in range(1, total_epochs + 1):
            epoch_loss = 0.0
            for images, labels in batches:
                logits = _forward_with_head(clf, images.to(device), head=weight)
                loss = torch.nn.functional.cross_entropy(logits, labels.to(device).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())

            if epoch == 1 or epoch == total_epochs or epoch % every == 0:
                header = f"{log_prefix} epoch {epoch}/{total_epochs}  loss={epoch_loss / len(batches):.4f}"
                print(_eval_line(header) if eval_loaders else header)
    finally:
        clf._zs_text_features = head
        for n, p in clf.named_parameters():
            p.requires_grad_(original_requires_grad[n])

    return weight.detach()
