import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.io.utils import read_json_silent

_DEFAULT_ATTN_PATCH_CFG: dict[str, Any] = {
    "attn_impl": "softmax",
    "kernel": "elu_plus_one",
    "eps": 1e-6,
    "linear_rule": "kernel",
    "delta_eta": 1.0,
    "delta_exclude_cls_from_store": True,
    "delta_cls_only_readout": False,
    "delta_learn_w0": False,
    "delta_w0_rank": 0,
}
_SPLIT_ATTN_MARKERS = (
    ".attn.q_proj.",
    ".attn.k_proj.",
    ".attn.v_proj.",
    ".attn.out_proj.",
)
_FUSED_ATTN_MARKERS = (
    ".attn.in_proj_weight",
    ".attn.in_proj_bias",
)


def is_peft_adapter_dir_ckpt(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == "peft" and isinstance(obj.get("peft_adapter_dir"), str)


def load_peft_adapter_dir_components(adapter_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Returns (peft_state, peft_cfg_map) compatible with your existing helpers:
      - peft_state: state dict of adapter params (cpu tensors)
      - peft_cfg_map: dict like {"default": <adapter_config_dict>}
    """
    ad = Path(adapter_dir)
    if not ad.exists():
        raise FileNotFoundError(f"PEFT adapter_dir not found: {ad}")

    # 1) adapter config
    cfg_path = ad / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {ad}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"adapter_config.json is not a dict: {cfg_path}")

    # 2) adapter weights
    # PEFT commonly writes either:
    #  - adapter_model.safetensors
    #  - adapter_model.bin (older)
    st_path = ad / "adapter_model.safetensors"
    bin_path = ad / "adapter_model.bin"

    if st_path.exists():
        try:
            from safetensors.torch import load_file as _st_load_file  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Found adapter_model.safetensors but safetensors is not installed. "
                "Install `safetensors` or save adapters as .bin."
            ) from e
        peft_state = _st_load_file(str(st_path))
    elif bin_path.exists():
        peft_state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No adapter weights found in {ad} (expected adapter_model.safetensors or .bin)")

    if not isinstance(peft_state, dict):
        raise ValueError(f"Adapter weights are not a dict in {ad}")

    # ensure CPU tensors
    peft_state = {k: v.detach().cpu() for k, v in peft_state.items() if torch.is_tensor(v)}

    # Your downstream expects a map of adapter-name -> config-dict
    peft_cfg_map = {"default": cfg_dict}
    return peft_state, peft_cfg_map


def get_patched_attn_flag(ckpt_obj: dict[str, Any]) -> bool:
    # Prefer explicit key in the .pt payload
    if "patched_attn" in ckpt_obj:
        return bool(ckpt_obj["patched_attn"])

    # Fallback: read your meta json inside adapter dir
    ad = ckpt_obj.get("peft_adapter_dir", None)
    if isinstance(ad, str):
        meta = read_json_silent(str(Path(ad) / "merge_and_rebase_meta.json"))
        if "patched_attn" in meta:
            return bool(meta["patched_attn"])
    return False


def normalize_attn_patch_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(cfg or {})
    attn_impl = str(raw.get("attn_impl", _DEFAULT_ATTN_PATCH_CFG["attn_impl"])).strip().lower()
    if attn_impl not in {"softmax", "linear"}:
        raise ValueError(f"Unknown attn_impl '{attn_impl}'. Choose from: softmax, linear")
    linear_rule = str(raw.get("linear_rule", _DEFAULT_ATTN_PATCH_CFG["linear_rule"])).strip().lower()
    if linear_rule not in {"kernel", "delta"}:
        raise ValueError(f"Unknown linear_rule '{linear_rule}'. Choose from: kernel, delta")
    return {
        "attn_impl": attn_impl,
        "kernel": str(raw.get("kernel", _DEFAULT_ATTN_PATCH_CFG["kernel"])),
        "eps": float(raw.get("eps", _DEFAULT_ATTN_PATCH_CFG["eps"])),
        "linear_rule": linear_rule,
        "delta_eta": float(raw.get("delta_eta", _DEFAULT_ATTN_PATCH_CFG["delta_eta"])),
        "delta_exclude_cls_from_store": bool(
            raw.get("delta_exclude_cls_from_store", _DEFAULT_ATTN_PATCH_CFG["delta_exclude_cls_from_store"])
        ),
        "delta_cls_only_readout": bool(
            raw.get("delta_cls_only_readout", _DEFAULT_ATTN_PATCH_CFG["delta_cls_only_readout"])
        ),
        "delta_learn_w0": bool(raw.get("delta_learn_w0", _DEFAULT_ATTN_PATCH_CFG["delta_learn_w0"])),
        "delta_w0_rank": int(raw.get("delta_w0_rank", _DEFAULT_ATTN_PATCH_CFG["delta_w0_rank"])),
    }


def state_dict_looks_patched_attn(sd: Mapping[str, Any]) -> bool:
    keys = tuple(str(k) for k in sd.keys())
    has_split_attn = any(any(marker in k for marker in _SPLIT_ATTN_MARKERS) for k in keys)
    has_fused_attn = any(any(marker in k for marker in _FUSED_ATTN_MARKERS) for k in keys)
    return has_split_attn and not has_fused_attn


def get_attn_patch_cfg(ckpt_obj: dict[str, Any]) -> dict[str, Any]:
    cfg = ckpt_obj.get("attn_patch_cfg", None)
    if isinstance(cfg, dict):
        return normalize_attn_patch_cfg(cfg)

    ad = ckpt_obj.get("peft_adapter_dir", None)
    if isinstance(ad, str):
        meta = read_json_silent(str(Path(ad) / "merge_and_rebase_meta.json"))
        cfg2 = meta.get("attn_patch_cfg", None)
        if isinstance(cfg2, dict):
            return normalize_attn_patch_cfg(cfg2)

    return normalize_attn_patch_cfg(_DEFAULT_ATTN_PATCH_CFG)
