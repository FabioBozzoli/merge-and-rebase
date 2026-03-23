from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.pcb_merge import PCBMerge
from merge_and_rebase.merge.registry import get_method


def _toy_checkpoints() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    base = {
        "w": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "b": torch.tensor([0.0, 0.0]),
    }
    tuned = [
        {
            "w": torch.tensor([[1.0, -2.0], [0.5, -0.1]]),
            "b": torch.tensor([0.2, -0.4]),
        },
        {
            "w": torch.tensor([[0.4, 0.2], [-1.0, 0.3]]),
            "b": torch.tensor([-0.3, 0.1]),
        },
    ]
    return base, tuned


def test_pcb_registered() -> None:
    method = get_method("pcb")
    assert isinstance(method, PCBMerge)
    method_alias = get_method("pcb_merge")
    assert isinstance(method_alias, PCBMerge)


def test_pcb_merge_alpha_zero_is_base() -> None:
    base, tuned = _toy_checkpoints()
    method = PCBMerge()

    merged = method.merge(base=base, tuned=tuned, alpha=0.0, strict=True)

    assert torch.allclose(merged["w"], base["w"])
    assert torch.allclose(merged["b"], base["b"])


def test_pcb_merge_returns_finite_and_correct_shapes() -> None:
    base, tuned = _toy_checkpoints()
    method = PCBMerge()

    merged = method.merge(base=base, tuned=tuned, strict=True)

    assert merged["w"].shape == base["w"].shape
    assert merged["b"].shape == base["b"].shape
    assert torch.isfinite(merged["w"]).all()
    assert torch.isfinite(merged["b"]).all()


def test_pcb_lam_zero_returns_base() -> None:
    base, tuned = _toy_checkpoints()
    method = PCBMerge()

    merged = method.merge(base=base, tuned=tuned, strict=True, method_params={"lam": 0.0})

    assert torch.allclose(merged["w"], base["w"])
    assert torch.allclose(merged["b"], base["b"])


def test_pcb_ratio_validation() -> None:
    base, tuned = _toy_checkpoints()
    method = PCBMerge()

    with pytest.raises(ValueError, match="att_ratio"):
        method.merge(base=base, tuned=tuned, strict=True, method_params={"att_ratio": 0.0})
