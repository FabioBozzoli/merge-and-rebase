from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.eval import kd_rebase
from merge_and_rebase.rebase.methods import steer as steer_mod

transformers = pytest.importorskip("transformers")

VOCAB = 64
PAD = 0
EOS = 1
NUM_LABELS = 3


# --------------------------------------------------------------------------
# Loss and config
# --------------------------------------------------------------------------


def test_kd_loss_vanishes_when_student_matches_teacher() -> None:
    logits = torch.randn(8, 5)
    assert float(kd_rebase.kd_loss(logits, logits.clone())) == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_is_positive_and_scales_with_temperature_squared() -> None:
    s, t = torch.randn(8, 5), torch.randn(8, 5)
    assert float(kd_rebase.kd_loss(s, t)) > 0.0
    t2 = 2.0
    expected = (
        torch.nn.functional.kl_div(torch.log_softmax(s / t2, -1), torch.softmax(t / t2, -1), reduction="batchmean")
        * t2**2
    )
    assert float(kd_rebase.kd_loss(s, t, temperature=t2)) == pytest.approx(float(expected), rel=1e-5)


def test_kd_loss_cross_entropy_term_needs_labels() -> None:
    s, t = torch.randn(4, 3), torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    with_ce = kd_rebase.kd_loss(s, t, labels=labels, ce_weight=0.5)
    ce = torch.nn.functional.cross_entropy(s, labels)
    assert float(with_ce) == pytest.approx(float(kd_rebase.kd_loss(s, t)) + 0.5 * float(ce), rel=1e-5)
    with pytest.raises(ValueError, match="needs labels"):
        kd_rebase.kd_loss(s, t, ce_weight=0.5)


def test_kd_config_defaults_to_the_last_epoch() -> None:
    assert kd_rebase.resolve_kd_config({"kd": {"few_shot": 1}})["epoch_selection"] == "last"


def test_kd_config_falls_back_to_steer_method_params() -> None:
    kd = kd_rebase.resolve_kd_config({"method_params": {"few_shot": 5, "seed": 54}, "kd": {"lr": 1e-4}})
    assert (kd["few_shot"], kd["seed"], kd["lr"]) == (5, 54, 1e-4)
    assert kd["epochs"] == kd_rebase._KD_DEFAULTS["epochs"]


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        ({}, "few_shot"),
        ({"kd": {"few_shot": 1, "temperature": 0}}, "temperature"),
        ({"kd": {"few_shot": 1, "epochs": 0}}, "epochs"),
        ({"kd": {"few_shot": 1, "amp": "fp16"}}, "amp"),
        ({"kd": {"few_shot": 1, "train_mode": "x"}}, "train_mode"),
        ({"kd": {"few_shot": 1, "epoch_selection": "best"}}, "epoch_selection"),
    ],
)
def test_kd_config_rejects_bad_values(cfg: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kd_rebase.resolve_kd_config(cfg)


# --------------------------------------------------------------------------
# Vision: the support set must be steer's
# --------------------------------------------------------------------------


class _FakeHFVisionDataset:
    """Just the label plumbing ``steer._dataset_labels`` reads."""

    label_key = "label"

    def __init__(self, per_class: int, classes: int = 10) -> None:
        self.split = {"label": [c for c in range(classes) for _ in range(per_class)]}

    def _map_label(self, y: int) -> int:
        return int(y)

    def __len__(self) -> int:
        return len(self.split["label"])


def test_vision_support_matches_steer_for_plain_tasks() -> None:
    ds = _FakeHFVisionDataset(per_class=30)
    got = kd_rebase._vision_support_indices(ds, "DTD", 4, 33)
    want = steer_mod._few_shot(steer_mod._dataset_labels(ds), 4, 33).tolist()
    assert got == want
    labels = steer_mod._dataset_labels(ds)
    assert sorted(torch.bincount(labels[got]).tolist()) == [4] * 10


def test_vision_support_for_svhn_draws_from_the_50_per_class_pool() -> None:
    ds = _FakeHFVisionDataset(per_class=80)
    labels = steer_mod._dataset_labels(ds)
    pool = set(steer_mod._few_shot(labels, 50, seed=0).tolist())

    got = kd_rebase._vision_support_indices(ds, "SVHN", 5, 33)
    assert len(got) == 50 and set(got) <= pool
    assert torch.bincount(labels[got]).tolist() == [5] * 10
    # Same call, same draw (seed-deterministic), different seed, different draw.
    assert got == kd_rebase._vision_support_indices(ds, "SVHN", 5, 33)
    assert got != kd_rebase._vision_support_indices(ds, "SVHN", 5, 54)

    # And it is exactly what steer.prepare would select out of its cached subset.
    keep = steer_mod._few_shot(labels, 50, seed=0)
    want = keep[steer_mod._few_shot(labels[keep], 5, 33)].tolist()
    assert got == want


# --------------------------------------------------------------------------
# Text: end to end on tiny models (teacher T5, student wider T5 / RoBERTa)
# --------------------------------------------------------------------------


def _tiny_t5(d_model: int, seed: int):
    from transformers import T5Config, T5ForSequenceClassification

    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=VOCAB, d_model=d_model, d_ff=2 * d_model, d_kv=8, num_layers=2, num_decoder_layers=2,
        num_heads=2, num_labels=NUM_LABELS, pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
        dropout_rate=0.0,
    )  # fmt: skip
    return T5ForSequenceClassification(config).eval()


def _tiny_roberta(seed: int):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(seed)
    config = RobertaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_labels=NUM_LABELS, pad_token_id=PAD, max_position_embeddings=80, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )  # fmt: skip
    return RobertaForSequenceClassification(config).eval()


class _DictDataset(Dataset):
    """Same tokens for teacher and student, so the pairing is checkable; ids 0/1 reserved (pad/eos)."""

    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(2, VOCAB, (n, 6), generator=g)
        self.input_ids[:, -1] = EOS
        self.attention_mask = torch.ones_like(self.input_ids)
        self.local = torch.arange(n) % 2
        self.labels = torch.tensor([[0, 2][int(y)] for y in self.local])

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}


def _fake_loaders(seed: int = 0) -> SimpleNamespace:
    sets = {"train": _DictDataset(40, seed), "val": _DictDataset(12, seed + 1), "test": _DictDataset(20, seed + 2)}
    loaders = {k: DataLoader(v, batch_size=4, shuffle=False, collate_fn=_collate) for k, v in sets.items()}
    return SimpleNamespace(
        **loaders,
        mask_class=[0, 2],
        local_labels={k: v.local.tolist() for k, v in sets.items()},
    )


def test_converted_encoder_checkpoint_dirs_are_refused(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["T5EncoderForSequenceClassification"]}))
    with pytest.raises(ValueError, match="randomly initialised"):
        kd_rebase._check_hf_checkpoint_dir(str(tmp_path))

    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "config.json").write_text(json.dumps({"architectures": ["T5ForSequenceClassification"]}))
    kd_rebase._check_hf_checkpoint_dir(str(ok))  # no error

    with pytest.raises(ValueError, match="without config.json"):
        kd_rebase._check_hf_checkpoint_dir(str(tmp_path / "nowhere"))


@pytest.mark.parametrize("selection", ["last", "val"])
@pytest.mark.parametrize("student", ["t5_wider", "roberta"])
def test_text_kd_runs_end_to_end_and_reports_the_steer_summary_schema(
    monkeypatch, tmp_path, student, selection
) -> None:
    from merge_and_rebase.eval import llm_merge, text_rebase
    from merge_and_rebase.models.text_lm import TextBuildConfig, TextLM

    teacher_base, teacher_tuned = _tiny_t5(32, seed=0), _tiny_t5(32, seed=7)
    student_model = _tiny_t5(48, seed=1) if student == "t5_wider" else _tiny_roberta(seed=1)

    def build_llm(cfg, *, role, model_kind, device):  # noqa: ARG001
        model = teacher_base if role == "source" else student_model
        return TextLM(model, tokenizer=None), TextBuildConfig(model_name_or_path=role, model_arch="auto")

    monkeypatch.setattr(text_rebase, "_build_llm", build_llm)
    monkeypatch.setattr(
        text_rebase,
        "_build_task_splits",
        lambda **kw: {k: SimpleNamespace(labels=[0, 1]) for k in ("train", "val", "test")},
    )
    monkeypatch.setattr(text_rebase, "_tokenize_splits", lambda **kw: _fake_loaders())
    monkeypatch.setattr(llm_merge, "_head_class_ids_for_task", lambda **kw: [0, 2])
    monkeypatch.setattr(llm_merge, "_load_task_heads", lambda path: {"rte": {}})
    monkeypatch.setattr(llm_merge, "_inject_task_head", lambda **kw: None)
    monkeypatch.setattr(kd_rebase, "_load_teacher_state_dict_text", lambda *a, **kw: dict(teacher_tuned.state_dict()))

    cfg = {
        "suite": "nli6",
        "tasks": "rte",
        "device": "cpu",
        "num_labels": NUM_LABELS,
        "target_task_heads": "unused.pt",
        "tuned_ckpts": {"rte": "unused.pt"},
        "kd": {
            "few_shot": 3,
            "seed": 33,
            "epochs": 12,
            "lr": 3e-3,
            "batch_size": 4,
            "temperature": 2.0,
            "epoch_selection": selection,
        },
        "logging": {"local_log_dir": str(tmp_path), "run_name": "kd_test"},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    kd_rebase.main(["text", "--config", str(cfg_path)])

    summary = json.loads((tmp_path / "kd_test.json").read_text())
    assert summary["method"] == "kd" and summary["baseline_label"] == "target_zeroshot"
    results = summary["test_results"]
    for key in ("per_task_baseline_accuracy", "per_task_absolute_accuracy", "per_task_normalized_accuracy_ratio"):
        assert set(results[key]) == {"rte"}
    assert 0.0 <= results["per_task_absolute_accuracy"]["rte"] <= 1.0

    diag = summary["kd_diagnostics"]["rte"]
    assert diag["n_support"] == 6  # 3 shots x 2 classes
    assert len(diag["history"]) == 13  # epoch 0 (untouched student) + 12
    losses = [h["loss"] for h in diag["history"][1:]]
    assert losses[-1] < losses[0], "distillation loss should go down on the support set"
    assert diag["epoch_selection"] == selection
    if selection == "last":
        # Fixed epoch: always the final one, whatever val says.
        assert diag["best_epoch"] == 12
        assert diag["best_val"] == diag["history"][-1]["val"]
    else:
        # Epoch 0 is the baseline, so the selected epoch can never be worse on val.
        assert diag["best_val"] >= diag["baseline_val"]
