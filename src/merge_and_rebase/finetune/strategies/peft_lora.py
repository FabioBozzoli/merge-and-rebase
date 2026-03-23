from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model

from merge_and_rebase.finetune.strategies.full import cosine_lr
from merge_and_rebase.models.patch_openclip_attention import patch_openclip_vit_attn

from .registry import register


def infer_linear_leaf_names(m: nn.Module) -> set[str]:
    leafs = set()
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            leafs.add(name.split(".")[-1])
    return leafs


def _resolve_attn_patch_cfg(
    *,
    peft_cfg: dict[str, Any],
    strategy_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if isinstance(strategy_cfg, dict):
        attention_cfg = strategy_cfg.get("attention", {})
        if attention_cfg is None:
            attention_cfg = {}
        if not isinstance(attention_cfg, dict):
            raise ValueError("strategy.attention must be a dict when provided.")
        raw.update(attention_cfg)

    # Backward-compat convenience: allow these keys under strategy.peft too.
    for k in (
        "attn_impl",
        "kernel",
        "eps",
        "ramp_fraction",
        "linear_rule",
        "delta_eta",
        "delta_exclude_cls_from_store",
        "delta_cls_only_readout",
        "delta_learn_w0",
        "delta_w0_rank",
    ):
        if k in peft_cfg and k not in raw:
            raw[k] = peft_cfg[k]

    attn_impl = str(raw.get("attn_impl", "softmax"))
    ramp_fraction_default = 0.2 if attn_impl == "linear" else 0.0
    ramp_fraction = float(raw.get("ramp_fraction", ramp_fraction_default))
    if ramp_fraction < 0.0 or ramp_fraction > 1.0:
        raise ValueError("attention.ramp_fraction must be in [0, 1].")

    return {
        "attn_impl": attn_impl,
        "kernel": str(raw.get("kernel", "elu_plus_one")),
        "eps": float(raw.get("eps", 1e-6)),
        "ramp_fraction": ramp_fraction,
        "linear_rule": str(raw.get("linear_rule", "kernel")),
        "delta_eta": float(raw.get("delta_eta", 1.0)),
        "delta_exclude_cls_from_store": bool(raw.get("delta_exclude_cls_from_store", True)),
        "delta_cls_only_readout": bool(raw.get("delta_cls_only_readout", False)),
        "delta_learn_w0": bool(raw.get("delta_learn_w0", False)),
        "delta_w0_rank": int(raw.get("delta_w0_rank", 0)),
    }


@dataclass(frozen=True)
class PeftLoraVision:
    """
    Apply LoRA to *vision encoder only* (OpenCLIP visual), keeping everything else frozen
    except optionally logit_scale and/or other explicitly allowed params.

    Expected model layout:
      model.clip_model.model.visual  -> vision module (patched with PEFT)
      model.clip_model.model.transformer -> text encoder (kept frozen)
      model.clip_model.logit_scale -> scalar parameter (optionally trainable)

    Config expected in vision yaml:
      strategy:
        name: peft_lora
        peft:
          r: 16
          lora_alpha: 16
          lora_dropout: 0.0
          bias: "none"
          target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]  # REQUIRED: list of module names to target with LoRA
          train_logit_scale: false                    # optional
    """

    name: str = "peft_lora"

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_length: int,
        optimizer: str = "adamw",
        steps: int,
        device: torch.device,
        peft_cfg: dict[str, Any] | None = None,
        strategy_cfg: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        if peft_cfg is None:
            peft_cfg = {}

        # --- locate OpenCLIP pieces ---
        clip = getattr(model, "clip_model", None)
        if clip is None:
            raise ValueError("PeftLoraVision expects `model.clip_model` (your OpenClipClassifier).")

        if not hasattr(clip, "model"):
            raise ValueError("PeftLoraVision expects `model.clip_model.model` (OpenCLIP model).")

        openclip_model = clip.model
        if not hasattr(openclip_model, "visual"):
            raise ValueError("PeftLoraVision expects `model.clip_model.model.visual` to exist.")

        visual = model.clip_model.model.visual

        # Optionally train logit_scale
        train_logit_scale = bool(peft_cfg.get("train_logit_scale", False))
        if train_logit_scale and hasattr(clip, "logit_scale"):
            try:
                clip.logit_scale.requires_grad_(True)
            except Exception:
                pass

        # --- build LoRA config ---
        target_modules = peft_cfg.get("target_modules", None)
        if target_modules is None:
            raise ValueError("strategy.peft.target_modules is required for PeftLoraVision.")
        if not isinstance(target_modules, list) or not all(isinstance(x, str) for x in target_modules):
            raise ValueError("strategy.peft.target_modules must be a list[str] (or omit it).")

        # leafs = infer_linear_leaf_names(clip.model.visual)
        # print("Linear leaf names:", sorted(leafs))

        attn_patch_cfg: dict[str, Any] | None = None
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            attn_patch_cfg = _resolve_attn_patch_cfg(peft_cfg=peft_cfg, strategy_cfg=strategy_cfg)

        modules_to_save = peft_cfg.get("modules_to_save", None)
        if modules_to_save is not None and (not isinstance(modules_to_save, list) or not all(isinstance(x, str) for x in modules_to_save)):
            raise ValueError("strategy.peft.modules_to_save must be a list[str] when provided.")
        mts = list(modules_to_save or [])
        if isinstance(attn_patch_cfg, dict) and bool(attn_patch_cfg.get("delta_learn_w0", False)):
            if "delta_mem" not in mts:
                mts.append("delta_mem")

        lora_cfg = LoraConfig(
            r=int(peft_cfg.get("r", 16)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
            target_modules=target_modules,
            bias=str(peft_cfg.get("bias", "none")),
            modules_to_save=mts if mts else None,
        )

        # if target_modules contains "qkv" or "proj", we need to patch attention modules
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            assert attn_patch_cfg is not None
            ramp_steps = int(round(float(attn_patch_cfg.get("ramp_fraction", 0.0)) * max(1, int(steps))))
            attn_patch_cfg["ramp_steps"] = int(ramp_steps)
            n = patch_openclip_vit_attn(
                visual,
                proj_dropout=0.0,
                attn_impl=attn_patch_cfg["attn_impl"],
                kernel=attn_patch_cfg["kernel"],
                eps=attn_patch_cfg["eps"],
                ramp_steps=ramp_steps,
                linear_rule=str(attn_patch_cfg.get("linear_rule", "kernel")),
                delta_eta=float(attn_patch_cfg.get("delta_eta", 1.0)),
                delta_exclude_cls_from_store=bool(attn_patch_cfg.get("delta_exclude_cls_from_store", True)),
                delta_cls_only_readout=bool(attn_patch_cfg.get("delta_cls_only_readout", False)),
                delta_learn_w0=bool(attn_patch_cfg.get("delta_learn_w0", False)),
                delta_w0_rank=int(attn_patch_cfg.get("delta_w0_rank", 0)),
            )
            if n == 0:
                raise RuntimeError(
                    "No OpenCLIP attention modules were patched. "
                    "Check that the model is a ViT and target_modules are correct."
                )
            model.peft_patched_attn = True  # type: ignore[attr-defined]
            model.peft_attn_patch_cfg = attn_patch_cfg  # type: ignore[attr-defined]

        # --- freeze everything by default ---
        for p in model.parameters():
            p.requires_grad = False

        # Ensure text encoder stays frozen
        if hasattr(openclip_model, "transformer"):
            for p in openclip_model.transformer.parameters():
                p.requires_grad = False

        # --- wrap ONLY visual with PEFT ---
        peft_visual = get_peft_model(visual, lora_cfg)
        openclip_model.visual = peft_visual
        model.to(device)

        # Ensure LoRA params exist and require_grad
        lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n.lower()]
        print("n_lora_params:", len(lora), "any_trainable:", any(p.requires_grad for _, p in lora))

        # Double-check we really injected something trainable
        lora_trainables = [p for p in model.clip_model.model.visual.parameters() if p.requires_grad]
        if len(lora_trainables) == 0:
            # If you hit this, target_modules likely didn't match the module names.
            names = [n for n, _ in model.clip_model.model.visual.named_modules()]
            raise RuntimeError(
                "No trainable LoRA parameters were created. "
                "Likely target_modules mismatch for OpenCLIP visual.\n"
                f"Example visual submodule names (first 40): {names[:40]}"
            )

        model.to(device)

        # Optimizer only over trainable params (LoRA + optional logit_scale)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found (after LoRA injection).")

        opt = self.get_optimizer(trainable_params, optimizer, lr, weight_decay)
        scheduler = cosine_lr(opt, lr, warmup_length, steps)

        info: dict[str, int] = {
            "trainable_params": sum(p.numel() for p in trainable_params),
            "lora_params": sum(
                p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()
            ),
        }
        if train_logit_scale and hasattr(clip, "logit_scale"):
            info["logit_scale_params"] = int(clip.logit_scale.numel())
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            info["attn_ramp_steps"] = int(attn_patch_cfg.get("ramp_steps", 0))

        return opt, scheduler, info

    def get_optimizer(self, params, opt: str, lr: float, weight_decay: float) -> optim.Optimizer:
        if opt.lower() == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt.lower() == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt.lower() == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")


register(PeftLoraVision())
