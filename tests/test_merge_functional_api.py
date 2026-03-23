from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.functional import list_functional_methods, merge_functional, merge_raw_matrices

ALL_FUNCTIONAL_METHODS = [
    "task_arithmetic",
    "weighted_average",
    "tsv_merge",
    "isoc_merge",
    "isocts_merge",
    "dare_merge",
    "ties_merge",
    "pcb",
    "pcb_merge",
    "cart_merge",
]


def _toy_matrices_2d() -> list[torch.Tensor]:
    return [
        torch.tensor(
            [
                [1.0, -0.5, 0.2],
                [-0.3, 0.8, -0.7],
                [0.4, -0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.3, 0.6, -0.2],
                [0.9, -0.4, 0.5],
                [-0.8, 0.2, 0.7],
            ],
            dtype=torch.float32,
        ),
    ]


def _toy_matrices_1d() -> list[torch.Tensor]:
    return [
        torch.tensor([0.2, -0.4, 0.1, 0.5], dtype=torch.float32),
        torch.tensor([-0.3, 0.5, -0.2, 0.1], dtype=torch.float32),
    ]


def test_list_functional_methods_contains_all() -> None:
    methods = set(list_functional_methods())
    assert methods.issuperset(set(ALL_FUNCTIONAL_METHODS))


@pytest.mark.parametrize("method_name", ALL_FUNCTIONAL_METHODS)
def test_merge_functional_accepts_raw_matrices(method_name: str) -> None:
    mats = _toy_matrices_2d()

    merged = merge_functional(
        method_name,
        matrices=mats,
        svd_dtype="fp32",
        accum_dtype="fp32",
        topk=1.0,
        merging_type="mean",
        drop_rate=0.0,
        common_space_fraction=0.0,
        pruning_rank=1,
        scaling_coeffs=0.5,
        clamp_min_ratio=0.0,
        clamp_max_ratio=0.0,
        att_ratio=1.0,
        lam=1.0,
    )

    assert merged.shape == mats[0].shape
    assert torch.isfinite(merged).all()


def test_merge_raw_matrices_alias() -> None:
    mats = _toy_matrices_2d()
    merged = merge_raw_matrices("task_arithmetic", matrices=mats)
    assert merged.shape == mats[0].shape
    assert torch.isfinite(merged).all()


def test_merge_functional_unknown_method_raises() -> None:
    mats = _toy_matrices_2d()
    with pytest.raises(KeyError):
        merge_functional("not_a_method", matrices=mats)


def test_vector_methods_accept_1d() -> None:
    mats = _toy_matrices_1d()
    for name in ["task_arithmetic", "weighted_average", "dare_merge", "ties_merge", "pcb", "pcb_merge"]:
        merged = merge_functional(
            name,
            matrices=mats,
            topk=1.0,
            drop_rate=0.0,
            clamp_min_ratio=0.0,
            clamp_max_ratio=0.0,
            att_ratio=1.0,
        )
        assert merged.shape == mats[0].shape


def test_matrix_only_methods_reject_1d() -> None:
    mats = _toy_matrices_1d()
    for name in ["tsv_merge", "isoc_merge", "isocts_merge", "cart_merge"]:
        with pytest.raises(ValueError, match="requires 2D matrices"):
            merge_functional(name, matrices=mats)
