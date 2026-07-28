import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from merge_and_rebase.eval.utils import load_vision_checkpoint_reference
from merge_and_rebase.io import peft_helpers


def test_load_vision_checkpoint_reference_accepts_local_adapter_dir(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    ref, obj = load_vision_checkpoint_reference(ckpt_ref=str(adapter_dir))

    assert ref == str(adapter_dir)
    assert obj == {"format": "peft", "peft_adapter_dir": str(adapter_dir)}


def test_load_vision_checkpoint_reference_accepts_hf_adapter_repo(monkeypatch) -> None:
    adapter_dir = "/tmp/fake-adapter"

    def _fake_resolve(ref: str) -> str:
        assert ref == "hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
        return adapter_dir

    def _fail_torch_load(*args, **kwargs):
        raise AssertionError("torch.load should not be used for HF adapter refs")

    monkeypatch.setattr("merge_and_rebase.eval.utils.resolve_peft_adapter_dir", _fake_resolve)
    monkeypatch.setattr(torch, "load", _fail_torch_load)

    ref, obj = load_vision_checkpoint_reference(
        ckpt_ref="hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
    )

    assert ref == "hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
    assert obj == {"format": "peft", "peft_adapter_dir": adapter_dir}


def test_resolve_peft_adapter_dir_downloads_hf_repo(monkeypatch, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "downloaded-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    peft_helpers._HF_ADAPTER_DIR_CACHE.clear()

    calls: list[dict[str, object]] = []

    def _snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(adapter_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_snapshot_download))

    resolved = peft_helpers.resolve_peft_adapter_dir("example-org/example-adapter")

    assert resolved == adapter_dir
    assert calls == [
        {
            "repo_id": "example-org/example-adapter",
            "repo_type": "model",
            "allow_patterns": (
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
                "merge_and_rebase_meta.json",
            ),
        }
    ]
