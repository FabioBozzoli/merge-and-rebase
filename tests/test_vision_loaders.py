from __future__ import annotations

import pytest
from datasets import Dataset

from merge_and_rebase.data.vision_loaders import load_hf_splits


def test_load_hf_splits_loads_only_requested_splits(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None]] = []
    datasets_by_split = {
        "train": Dataset.from_dict({"label": [0, 1]}),
        "test": Dataset.from_dict({"label": [1, 0]}),
    }

    def _fake_load_dataset(path, *args, split=None, **kwargs):
        config = args[0] if args else None
        calls.append((config, split))
        if split is None:
            raise AssertionError("Fallback loading should not run when requested splits load directly.")
        return datasets_by_split[split]

    monkeypatch.setattr("merge_and_rebase.data.vision_loaders.hf_load_dataset", _fake_load_dataset)

    ds = load_hf_splits("tanganke/sun397", requested_splits=("train", "test"))

    assert list(ds.keys()) == ["train", "test"]
    assert calls == [(None, "train"), (None, "test")]


def test_load_hf_splits_rejects_unsupported_requested_split() -> None:
    with pytest.raises(ValueError, match="Unsupported requested splits"):
        load_hf_splits("tanganke/sun397", requested_splits=("train", "foo"))
