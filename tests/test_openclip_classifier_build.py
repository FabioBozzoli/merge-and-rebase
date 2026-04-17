from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))


def test_build_uses_hf_hub_model_name_without_pretrained_tag(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["device"] = device
        calls["quick_gelu"] = quick_gelu
        return _DummyModel(), None, "preprocess"

    def _fake_get_tokenizer(model_name):
        calls["tokenizer_model_name"] = model_name
        return "tokenizer"

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=_fake_get_tokenizer,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    cfg = OpenClipBuildConfig(model_name="hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K", pretrained="openai")
    clf = OpenClipClassifier.build(cfg)

    assert isinstance(clf, OpenClipClassifier)
    assert calls == {
        "model_name": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "pretrained": None,
        "device": "cuda",
        "quick_gelu": False,
        "tokenizer_model_name": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    }


def test_build_accepts_hf_hub_reference_in_pretrained_field(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["device"] = device
        calls["quick_gelu"] = quick_gelu
        return _DummyModel(), None, "preprocess"

    def _fake_get_tokenizer(model_name):
        calls["tokenizer_model_name"] = model_name
        return "tokenizer"

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=_fake_get_tokenizer,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    OpenClipClassifier.build(cfg)

    assert calls == {
        "model_name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "pretrained": None,
        "device": "cuda",
        "quick_gelu": False,
        "tokenizer_model_name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    }
