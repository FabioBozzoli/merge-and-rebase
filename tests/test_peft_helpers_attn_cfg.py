from __future__ import annotations

import json
from pathlib import Path

from merge_and_rebase.io.peft_helpers import (
    get_attn_patch_cfg,
    normalize_attn_patch_cfg,
    state_dict_looks_patched_attn,
)


def test_get_attn_patch_cfg_prefers_checkpoint_payload() -> None:
    cfg = get_attn_patch_cfg(
        {
            "attn_patch_cfg": {
                "attn_impl": "linear",
                "kernel": "exp",
                "eps": 1e-4,
            }
        }
    )
    assert cfg["attn_impl"] == "linear"
    assert cfg["kernel"] == "exp"
    assert float(cfg["eps"]) == 1e-4
    assert cfg["linear_rule"] == "kernel"
    assert cfg["delta_learn_w0"] is False
    assert int(cfg["delta_w0_rank"]) == 0


def test_get_attn_patch_cfg_falls_back_to_defaults(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with (adapter_dir / "merge_and_rebase_meta.json").open("w", encoding="utf-8") as f:
        json.dump({}, f)

    cfg = get_attn_patch_cfg({"peft_adapter_dir": str(adapter_dir)})
    assert cfg["attn_impl"] == "softmax"
    assert cfg["kernel"] == "elu_plus_one"
    assert float(cfg["eps"]) == 1e-6
    assert cfg["linear_rule"] == "kernel"
    assert cfg["delta_learn_w0"] is False
    assert int(cfg["delta_w0_rank"]) == 0


def test_normalize_attn_patch_cfg_lowercases_impl() -> None:
    cfg = normalize_attn_patch_cfg({"attn_impl": "LINEAR", "kernel": "exp", "eps": "1e-4", "linear_rule": "DELTA"})
    assert cfg["attn_impl"] == "linear"
    assert cfg["kernel"] == "exp"
    assert float(cfg["eps"]) == 1e-4
    assert cfg["linear_rule"] == "delta"


def test_state_dict_looks_patched_attn_detects_split_qkv() -> None:
    sd = {
        "visual.transformer.resblocks.0.attn.q_proj.weight": 1,
        "visual.transformer.resblocks.0.attn.k_proj.weight": 1,
        "visual.transformer.resblocks.0.attn.v_proj.weight": 1,
    }
    assert state_dict_looks_patched_attn(sd)


def test_state_dict_looks_patched_attn_false_for_fused_qkv() -> None:
    sd = {"visual.transformer.resblocks.0.attn.in_proj_weight": 1}
    assert not state_dict_looks_patched_attn(sd)
