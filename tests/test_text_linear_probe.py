"""The standalone text linear-probe entrypoint (eval/text_linear_probe.py).

The control has to stay comparable to the rebasin runs it bounds, which is why
these tests pin the support draw and the summary shape rather than accuracy: the
probe's job is to be the same probe, run without any rebasin machinery around it.
"""

from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval import text_linear_probe as tlp
from merge_and_rebase.rebase.methods.steer import _few_shot
from merge_and_rebase.rebase.text.adapters import balanced_indices

from test_text_rebase import _text_loaders, _tiny_roberta_encoder, _tiny_t5_encoder


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

    assert set(row) == {
        "task", "support_size", "init_test_accuracy", "probed_test_accuracy", "probed_val_accuracy",
        "selected_lr", "lr_sweep",
    }
    assert row["selected_lr"] == 0.05 and set(row["lr_sweep"]) == {"0.05"}
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


# --------------------------------------------------------------------------
# Support draw: the probe can see exactly the examples steer_text saw
# --------------------------------------------------------------------------


def test_steer_support_is_exactly_steers_few_shot() -> None:
    labels = _text_loaders(n=30, seed=0).local_labels["train"]

    drawn = tlp._draw_support(labels, 4, 33, "steer")
    expected = _few_shot(torch.as_tensor(labels, dtype=torch.long), 4, 33).tolist()

    assert drawn == expected
    assert drawn == tlp._draw_support(labels, 4, 33, "steer")
    assert drawn != tlp._draw_support(labels, 4, 54, "steer")
    assert sorted(labels[i] for i in drawn) == sorted([0] * 4 + [1] * 4)


def test_balanced_support_is_unchanged_and_differs_from_steer() -> None:
    labels = _text_loaders(n=30, seed=0).local_labels["train"]

    assert tlp._draw_support(labels, 4, 33, "balanced") == balanced_indices(labels, 4, seed=33)
    assert set(tlp._draw_support(labels, 4, 33, "balanced")) != set(tlp._draw_support(labels, 4, 33, "steer"))


def test_support_draw_rejects_unknown_mode_and_too_few_examples() -> None:
    labels = _text_loaders(n=10, seed=0).local_labels["train"]  # 5 per class

    with pytest.raises(ValueError, match="linear_probe_support"):
        tlp._draw_support(labels, 2, 33, "random")
    with pytest.raises(ValueError, match="need 6"):  # steer raises where balanced_indices caps silently
        tlp._draw_support(labels, 6, 33, "steer")


# --------------------------------------------------------------------------
# lr list: chosen on val, every lr restarts from the centroids
# --------------------------------------------------------------------------


def test_select_lr_uses_val_and_breaks_ties_towards_the_smaller_lr() -> None:
    assert tlp._select_lr({1e-4: {"val": 0.5, "test": 0.9}, 1e-3: {"val": 0.6, "test": 0.1}}) == 1e-3
    assert tlp._select_lr({1e-3: {"val": 0.6, "test": 0.0}, 1e-4: {"val": 0.6, "test": 0.0}}) == 1e-4


def test_parse_lrs_accepts_number_or_list() -> None:
    assert tlp._parse_lrs(1e-2) == [1e-2]
    assert tlp._parse_lrs([1e-4, 1e-3]) == [1e-4, 1e-3]
    with pytest.raises(ValueError):
        tlp._parse_lrs([])


def _probe_with_fake_llm(monkeypatch, model, *, val_by_call, test_by_call=None, lr, seen_heads=None, support="balanced"):
    loaders = _text_loaders(n=24, seed=0)
    loaders.val = loaders.test
    calls = {"n": 0}

    class _LLM:
        def __init__(self) -> None:
            self.model = model
            self.tokenizer = None

        def sequence_classification_accuracy(self, loader, *, device, mask_class):  # noqa: ARG002
            i = calls["n"]
            calls["n"] += 1
            # order per task: init(test), then per lr: val, test
            if i == 0:
                return 0.1
            k = i - 1
            return (val_by_call if k % 2 == 0 else (test_by_call or val_by_call))[k // 2]

    monkeypatch.setattr(tlp, "_build_task_splits", lambda **_kwargs: {})
    monkeypatch.setattr(tlp, "_tokenize_splits", lambda **_kwargs: loaders)
    monkeypatch.setattr(tlp, "_resolve_eval_loader", lambda loaders_obj, split: loaders_obj.test)
    if seen_heads is not None:
        real = tlp.train_linear_probe_head

        def spy(model_, *a, **kw):
            seen_heads.append(model_.classification_head.out_proj.weight.detach().clone())
            return real(model_, *a, **kw)

        monkeypatch.setattr(tlp, "train_linear_probe_head", spy)
    return tlp.probe_task(
        llm=_LLM(), task="snli", cfg={"num_labels": 3, "batch_size": 4}, device="cpu",
        seed=33, shots=2, epochs=3, lr=lr, dropout=False, log_every=None, support=support,
    )


def test_lr_list_reports_the_test_accuracy_of_the_val_selected_lr(monkeypatch) -> None:
    # val rises with the lr while test falls: selecting on test would pick 1e-4.
    row = _probe_with_fake_llm(
        monkeypatch, _tiny_t5_encoder(seed=0),
        val_by_call=[0.4, 0.5, 0.7], test_by_call=[0.9, 0.6, 0.3], lr=[1e-4, 1e-3, 1e-2],
    )

    assert row["selected_lr"] == 1e-2
    assert row["probed_val_accuracy"] == 0.7 and row["probed_test_accuracy"] == 0.3
    assert set(row["lr_sweep"]) == {"0.0001", "0.001", "0.01"}
    assert row["lr_sweep"]["0.0001"] == {"val": 0.4, "test": 0.9}
    assert row["init_test_accuracy"] == 0.1


def test_every_lr_starts_from_the_nearest_mean_head(monkeypatch) -> None:
    heads: list[torch.Tensor] = []
    _probe_with_fake_llm(
        monkeypatch, _tiny_t5_encoder(seed=0), val_by_call=[0.5, 0.5, 0.5], lr=[0.5, 0.6, 0.7], seen_heads=heads,
    )

    assert len(heads) == 3
    assert torch.equal(heads[0], heads[1]) and torch.equal(heads[0], heads[2])


def test_probe_on_roberta_trains_only_the_head() -> None:
    from merge_and_rebase.rebase.text import train_linear_probe_head

    loaders = _text_loaders(n=12, seed=0)
    model = _tiny_roberta_encoder(seed=0)
    before = {n: p.clone() for n, p in model.named_parameters()}

    train_linear_probe_head(
        model, loaders.train, device="cpu", mask_class=loaders.mask_class, lr=0.05, steps=3, dropout=False,
    )

    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved == {"classification_head.out_proj.weight", "classification_head.out_proj.bias"}
