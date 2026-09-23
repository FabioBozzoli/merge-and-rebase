from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call, jvp

from merge_and_rebase.models.forward_modes import list_forward_modes
from merge_and_rebase.utils.linearization import forward_ad_safe_attention_context
from merge_and_rebase.utils.peft_materialization import is_lora_parameter_name


def resolve_training_forward_mode(strategy_cfg: dict[str, Any] | None) -> str:
    cfg = dict(strategy_cfg or {})
    name = str(cfg.get("forward_mode", "standard")).strip()
    if name not in list_forward_modes():
        raise ValueError(f"strategy.forward_mode must be one of: {list_forward_modes()}")
    return name


def _lora_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if hasattr(mod, "base_layer") and isinstance(getattr(mod, "lora_A", None), nn.ModuleDict)
    ]


def apply_training_forward_mode(
    *,
    model: nn.Module,
    forward_mode: str,
    device: torch.device,
    output_transform: Callable[[Any], torch.Tensor] | None = None,
    output_builder: Callable[[torch.Tensor], Any] | None = None,
) -> dict[str, int]:
    """Bind f(x; theta0) + J(x; theta0) . dtheta as `model.forward`.

    The update enters ONLY as the jvp tangent, the way the reference trainer
    (FFTMammoth `clip_ft_ntk_text`) does it -- never as `W_now - W0`:

      * LoRA host weights: primal = the model's own frozen base weight (it IS
        W0: nothing ever writes to it), tangent = s * B @ A, formed in the factors'
        dtype (fp32) and cast once, so bf16 rounding is relative to dW itself
        rather than to W0. The LoRA branch is switched off during the call via
        each layer's plain `_disable_adapters` flag (PEFT's `disable_adapter()`
        toggles requires_grad, which is illegal inside a functorch transform).
      * Any other trainable parameter (a sequence-classification head via
        modules_to_save, `strategy: full`, ...): primal = a snapshot taken here,
        tangent = p - p0. Only these are copied; bind before they are trained.
      * Everything frozen is read live from the model: no reference copy.

    Against a deepcopy-reference + theta0 + materialized-W_now implementation
    this holds one copy of the base weights instead of three, and no per-step
    W_now: measured on Llama-3.2-3B r=32 seq 1024, 23.3 vs 39.9 GiB peak and
    0.53 vs 0.80 s per micro-batch. Because the expansion point is the frozen
    base weights, binding after an adapter was trained is fine (resume).
    """
    if forward_mode == "standard":
        model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
        return {"linearized_params": 0, "linearized_buffers": 0}
    if forward_mode != "linearized_ntk":
        raise ValueError(f"Unsupported training forward mode: {forward_mode}")

    named = dict(model.named_parameters())

    hosts: list[tuple[str, nn.Module, list[str]]] = []  # (base weight name, lora layer, adapters)
    lora_layers = _lora_layers(model)
    for name, mod in lora_layers:
        adapters = [a for a in mod.active_adapters if a in mod.lora_A]
        if not adapters:
            continue
        if any(getattr(mod, "use_dora", {}).get(a, False) for a in adapters):
            raise NotImplementedError(f"linearized_ntk does not support DoRA ({name}).")
        for a in adapters:
            factors = (mod.lora_A[a].weight, mod.lora_B[a].weight)
            if not all(p.requires_grad for p in factors):
                raise NotImplementedError(
                    f"{name}: adapter '{a}' is active but frozen; its update would have to be part of the "
                    "expansion point, which this linearization does not model."
                )
        hosts.append((f"{name}.base_layer.weight" if name else "base_layer.weight", mod, adapters))

    generic = [n for n, p in named.items() if p.requires_grad and not is_lora_parameter_name(n)]
    if not hosts and not generic:
        raise RuntimeError("No trainable parameters found for linearized_ntk forward mode.")
    for host_name, _, _ in hosts:
        if named[host_name].requires_grad:
            raise RuntimeError(f"{host_name} is trainable; LoRA host weights must be frozen.")

    generic_p0 = {n: named[n].detach().clone() for n in generic}
    names = [h for h, _, _ in hosts] + generic

    def _lora_delta(mod: nn.Module, adapters: list[str], dtype: torch.dtype) -> torch.Tensor:
        delta = None
        for a in adapters:
            b, a_w = mod.lora_B[a].weight, mod.lora_A[a].weight
            d = (b * float(mod.scaling[a])) @ a_w  # scale the small factor, not the full product
            if getattr(mod, "fan_in_fan_out", False):
                d = d.t()
            delta = d if delta is None else delta + d
        return delta.to(dtype)

    original_forward = model.forward
    inside = {"active": False}

    def _linearized_forward(*args: Any, **kwargs: Any) -> Any:
        # functional_call re-enters model.forward: the inner call is the plain forward.
        if inside["active"]:
            return original_forward(*args, **kwargs)

        first = next((v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
        if first is None:
            raise ValueError("The linearized forward needs at least one tensor input.")

        live = dict(model.named_parameters())
        primals = tuple(live[h] for h, _, _ in hosts) + tuple(generic_p0[n] for n in generic)
        tangents = tuple(_lora_delta(mod, ad, live[h].dtype) for h, mod, ad in hosts) + tuple(
            live[n] - generic_p0[n] for n in generic
        )

        def _f(*params: torch.Tensor) -> torch.Tensor:
            out = functional_call(model, dict(zip(names, params, strict=True)), args=args, kwargs=kwargs, strict=False)
            return output_transform(out) if output_transform is not None else out

        previous = [(mod, mod._disable_adapters) for _, mod in lora_layers]
        inside["active"] = True
        try:
            for mod, _ in previous:
                mod._disable_adapters = True
            with forward_ad_safe_attention_context(first.device):
                f0, df = jvp(_f, primals, tangents)
        finally:
            for mod, flag in previous:
                mod._disable_adapters = flag
            inside["active"] = False
        out = f0 + df
        return output_builder(out) if output_builder is not None else out

    model.forward = _linearized_forward  # type: ignore[method-assign]
    model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
    model._ntk_linearized = True  # type: ignore[attr-defined]
    model._ntk_linearized_names = list(names)  # type: ignore[attr-defined]
    return {"linearized_params": len(names), "linearized_buffers": 0}
