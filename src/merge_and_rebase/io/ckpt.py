from __future__ import annotations

from collections.abc import Mapping

import torch

TensorDict = dict[str, torch.Tensor]


def unwrap_state_dict(obj) -> TensorDict:
    """
    Accepts:
      - state_dict directly
      - {"state_dict": ...}
      - {"model": ...}
      - {"model_state_dict": ...}
    Returns dict[str, Tensor]
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    sd = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not sd:
        raise ValueError("No tensors found in checkpoint")
    return sd


def strip_prefix(sd: Mapping[str, torch.Tensor], prefix: str) -> TensorDict:
    if not prefix:
        return dict(sd)
    out: TensorDict = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def normalize_common_prefixes(sd: Mapping[str, torch.Tensor]) -> TensorDict:
    sd2 = dict(sd)

    # Trainer wrapper used by finetune/full checkpoints.
    # Example: clip_model.model.visual.* -> visual.*
    if any(k.startswith("clip_model.model.") for k in sd2):
        sd2 = strip_prefix(sd2, "clip_model.model.")

    # Generic trainer wrapper.
    if any(k.startswith("clip_model.") for k in sd2):
        sd2 = strip_prefix(sd2, "clip_model.")

    # DDP wrapper
    if any(k.startswith("module.") for k in sd2):
        sd2 = strip_prefix(sd2, "module.")

    # TSV / common wrapper: state_dict saved from an object with attribute .model
    if any(k.startswith("model.") for k in sd2):
        sd2 = strip_prefix(sd2, "model.")

    return sd2


def align_to_base_keys(sd: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor]) -> TensorDict:
    """
    Try to rename keys in sd so they match base's keyspace.
    Conservative: only renames when (new_key in base) and shapes match.
    """
    base_shapes = {k: tuple(v.shape) for k, v in base.items() if isinstance(v, torch.Tensor)}
    out: TensorDict = {}
    collisions: dict[str, str] = {}

    def try_add(old_k: str, new_k: str, v: torch.Tensor) -> bool:
        if new_k not in base_shapes:
            return False
        if tuple(v.shape) != base_shapes[new_k]:
            return False
        if new_k in out:
            collisions[new_k] = old_k
            return False
        out[new_k] = v
        return True

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        # 1) exact match
        if try_add(k, k, v):
            continue

        # 2) common wrapper strip already handled, but safe if sd wasn't normalized
        if k.startswith("model.") and try_add(k, k[len("model.") :], v):
            continue
        if k.startswith("module.") and try_add(k, k[len("module.") :], v):
            continue
        if k.startswith("clip_model.model.") and try_add(k, k[len("clip_model.model.") :], v):
            continue
        if k.startswith("clip_model.") and try_add(k, k[len("clip_model.") :], v):
            continue

        # 3) Visual prefix swap for transformer blocks only (your reported issue)
        # tuned has visual.transformer.* but base expects transformer.*
        if k.startswith("visual.transformer."):
            cand = "transformer." + k[len("visual.transformer.") :]
            if try_add(k, cand, v):
                continue

        # 4) opposite direction (base has visual.transformer.* but tuned has transformer.*)
        if k.startswith("transformer."):
            cand = "visual.transformer." + k[len("transformer.") :]
            if try_add(k, cand, v):
                continue

        # otherwise ignore unmatched keys (base stays base for those)
        # if you want, you can store them somewhere for debugging

    return out


def load_ckpt(path: str) -> TensorDict:
    # print(f"Loading checkpoint from {path}")
    obj = torch.load(path, map_location="cpu")
    sd = unwrap_state_dict(obj)
    return normalize_common_prefixes(sd)


def load_into_model(model, sd: Mapping[str, torch.Tensor], *, strict: bool = False) -> tuple[int, int]:
    """
    Loads sd into model with strict=False by default.
    Returns (missing, unexpected) counts.
    """
    missing, unexpected = model.load_state_dict(dict(sd), strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict load failed.\nmissing({len(missing)}): {missing[:20]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:20]}"
        )
    return len(missing), len(unexpected)
