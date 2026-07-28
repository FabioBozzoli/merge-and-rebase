from __future__ import annotations

from merge_and_rebase.eval.utils import stable_method_params_cache_key


def test_stable_method_params_cache_key_is_order_invariant_for_nested_dicts() -> None:
    left = {
        "subspace_variant": "posterior",
        "nested": {
            "tau": 1.0,
            "support": "subspace",
        },
    }
    right = {
        "nested": {
            "support": "subspace",
            "tau": 1.0,
        },
        "subspace_variant": "posterior",
    }

    assert stable_method_params_cache_key(left) == stable_method_params_cache_key(right)


def test_stable_method_params_cache_key_changes_when_values_change() -> None:
    first = {
        "subspace_variant": "masked",
        "mask_lambda": 0.05,
    }
    second = {
        "subspace_variant": "masked",
        "mask_lambda": 0.10,
    }

    assert stable_method_params_cache_key(first) != stable_method_params_cache_key(second)
