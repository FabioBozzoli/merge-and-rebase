"""The standalone text linear-probe entrypoint (eval/text_linear_probe.py).

The control has to stay comparable to the rebasin runs it bounds, which is why
these tests pin the support draw and the summary shape rather than accuracy: the
probe's job is to be the same probe, run without any rebasin machinery around it.
"""

from __future__ import annotations

import torch

from merge_and_rebase.eval import text_linear_probe as tlp
from merge_and_rebase.rebase.text.adapters import balanced_indices

from test_text_rebase import _text_loaders, _tiny_t5_encoder


def test_support_draw_is_balanced_and_seed_reproducible() -> None:
    """The support set is the experiment's only randomness; a silent change to it
    would move every number this control reports."""
    loaders = _text_loaders(n=24, seed=0)
    labels = loaders.local_labels["train"]

    first = balanced_indices(labels, 3, seed=33)
    again = balanced_indices(labels, 3, seed=33)
    other = balanced_indices(labels, 3, seed=54)

    assert first == again
    assert first != other
    assert len(first) == 3 * len(set(labels))
    counts = {c: sum(1 for i in first if labels[i] == c) for c in set(labels)}
    assert set(counts.values()) == {3}


def test_probe_task_reports_init_and_probed_accuracy(monkeypatch) -> None:
    """probe_task returns the row the summary is built from, with the two columns
    that make the control readable: the head before training and after it."""
    loaders = _text_loaders(n=24, seed=0)
    # The real TextLoaders carries a val split; the tiny fixture does not, and
    # probe_task hands eval_loaders straight to the probe's logging.
    loaders.val = loaders.test
    model = _tiny_t5_encoder(seed=0)

    class _LLM:
        def __init__(self) -> None:
            self.model = model
            self.tokenizer = None

        def sequence_classification_accuracy(self, loader, *, device, mask_class):  # noqa: ARG002
            return 0.5

    monkeypatch.setattr(tlp, "_build_task_splits", lambda **_kwargs: {})
    monkeypatch.setattr(tlp, "_tokenize_splits", lambda **_kwargs: loaders)
    monkeypatch.setattr(tlp, "_resolve_eval_loader", lambda loaders_obj, split: loaders_obj.test)

    row = tlp.probe_task(
        llm=_LLM(), task="snli", cfg={"num_labels": 3, "batch_size": 4}, device="cpu",
        seed=33, shots=2, epochs=2, lr=0.05, dropout=False, log_every=None,
    )

    assert set(row) == {"task", "support_size", "init_test_accuracy", "probed_test_accuracy", "probed_val_accuracy"}
    assert row["task"] == "snli"
    assert row["support_size"] == 2 * len(set(loaders.local_labels["train"]))
    assert all(isinstance(row[k], float) for k in row if k.endswith("accuracy"))


def test_probe_trains_only_the_head() -> None:
    """No rebasin here means nothing but the final linear may move -- this is the
    whole claim the control rests on."""
    loaders = _text_loaders(n=12, seed=0)
    model = _tiny_t5_encoder(seed=0)
    before = {n: p.clone() for n, p in model.named_parameters()}

    from merge_and_rebase.rebase.text import train_linear_probe_head

    train_linear_probe_head(
        model, loaders.train, device="cpu", mask_class=loaders.mask_class, lr=0.05, steps=3, dropout=False,
    )

    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved == {"classification_head.out_proj.weight", "classification_head.out_proj.bias"}
