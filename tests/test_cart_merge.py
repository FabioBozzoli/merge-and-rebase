from __future__ import annotations

import torch

from merge_and_rebase.merge.methods.cart import CARTMerge


def test_cart_fractional_rank_does_not_collapse_to_average() -> None:
    torch.manual_seed(0)
    base = {"layer.weight": torch.zeros((3, 3), dtype=torch.float32)}
    tuned = [{"layer.weight": torch.randn((3, 3), dtype=torch.float32)} for _ in range(3)]

    _, average_direction = CARTMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"pruning_rank": 1.0, "scaling_coeffs": 0.5},
    )
    _, truncated_direction = CARTMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"pruning_rank": 1.0 / 3.0, "scaling_coeffs": 0.5},
    )

    assert not torch.allclose(truncated_direction["layer.weight"], average_direction["layer.weight"])


def test_cart_accepts_absolute_rank_without_treating_it_as_a_fraction() -> None:
    torch.manual_seed(0)
    base = {"layer.weight": torch.zeros((3, 3), dtype=torch.float32)}
    tuned = [{"layer.weight": torch.randn((3, 3), dtype=torch.float32)} for _ in range(3)]

    _, rank_one = CARTMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"pruning_rank": 1, "scaling_coeffs": 0.5},
    )
    _, rank_two = CARTMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"pruning_rank": 2, "scaling_coeffs": 0.5},
    )

    assert not torch.allclose(rank_one["layer.weight"], rank_two["layer.weight"])
