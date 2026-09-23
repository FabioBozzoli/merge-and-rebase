from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call, jvp

from merge_and_rebase.models.forward_modes import list_forward_modes
from merge_and_rebase.utils.linearization import LinearizedModule, forward_ad_safe_attention_context
from merge_and_rebase.utils.peft_materialization import (
    is_lora_parameter_name,
    materialized_peft_param_map,
    training_linearization_param_names,
)

LINEARIZATION_IMPLS = ("dense", "lowrank")


def resolve_training_forward_mode(strategy_cfg: dict[str, Any] | None) -> str:
    cfg = dict(strategy_cfg or {})
    name = str(cfg.get("forward_mode", "standard")).strip()
    if name not in list_forward_modes():
        raise ValueError(f"strategy.forward_mode must be one of: {list_forward_modes()}")
    return name


def apply_training_forward_mode(
    *,
    model: nn.Module,
    forward_mode: str,
    device: torch.device,
    output_transform: Callable[[Any], torch.Tensor] | None = None,
    output_builder: Callable[[torch.Tensor], Any] | None = None,
    impl: str = "dense",
) -> dict[str, int]:
    if forward_mode == "standard":
        model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
        return {"linearized_params": 0, "linearized_buffers": 0}

    if forward_mode != "linearized_ntk":
        raise ValueError(f"Unsupported training forward mode: {forward_mode}")
    if impl not in LINEARIZATION_IMPLS:
        raise ValueError(f"linearization impl must be one of {LINEARIZATION_IMPLS}, got {impl!r}.")
    if impl == "lowrank":
        return _apply_lowrank_linearized_forward(
            model=model, output_transform=output_transform, output_builder=output_builder
        )

    param_names = training_linearization_param_names(model, trainable_only=True)
    if not param_names:
        raise RuntimeError("No trainable parameters found for linearized_ntk forward mode.")

    # LinearizedModule deepcopies the *PEFT-wrapped* model, so the frozen
    # reference keeps whatever the adapter held at bind time. It contributes 0
    # to f(x; theta0) only because PEFT zero-inits lora_B. Bind after loading a
    # trained adapter and the expansion point silently becomes
    # "pretrained + frozen adapter", double-counting the delta.
    warm_lora_b = [
        name
        for name, param in model.named_parameters()
        if "lora_B" in name and bool(param.detach().any())
    ]
    if warm_lora_b:
        raise RuntimeError(
            "linearized_ntk requires lora_B == 0 at bind time so the linearization point is the "
            f"pretrained weights; found {len(warm_lora_b)} nonzero lora_B tensors "
            f"(e.g. {warm_lora_b[0]}). Bind the forward mode before loading any adapter."
        )

    linearized = LinearizedModule.from_module(
        model,
        device=device,
        copy_module=True,
        param_names=param_names,
    )

    def _current_param_map() -> dict[str, torch.Tensor]:
        getter = getattr(model, "_current_param_map", None)
        raw = getter() if callable(getter) else None
        current_raw = None if raw is None else dict(raw)
        return materialized_peft_param_map(model, raw_current_params=current_raw)

    def _linearized_forward(*args: Any, **kwargs: Any) -> Any:
        out = linearized.forward(
            current_module=model,
            current_params=_current_param_map(),
            args=args,
            kwargs=kwargs,
            output_transform=output_transform,
        )
        return output_builder(out) if output_builder is not None else out

    model.forward = _linearized_forward  # type: ignore[method-assign]
    model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
    model._ntk_linearized = True  # type: ignore[attr-defined]
    model._linearized_module = linearized  # type: ignore[attr-defined]
    return {
        "linearized_params": len(linearized.param_names),
        "linearized_buffers": len(linearized.buffer_names),
    }


def _apply_lowrank_linearized_forward(
    *,
    model: nn.Module,
    output_transform: Callable[[Any], torch.Tensor] | None,
    output_builder: Callable[[torch.Tensor], Any] | None,
) -> dict[str, int]:
    """Same function as the dense path, without densifying the adapter.

    The dense path computes f(x; W0) + J_W(x; W0) . (s B A) by materializing
    W0 + s B A for every host weight and running jvp in weight space, so each
    linear pays a full-size dW . x. That tangent is exactly the derivative of
    the PEFT model itself along lora_B:

        g(eps) = f_peft(x; W0, A, eps * B)   =>   g(0) = f(x; W0),  g'(0) = J_W . (s B A)

    so jvp w.r.t. the scalar eps yields the identical output through PEFT's own
    W0 x + s B (A x), where the tangent product is rank r. No reference copy of
    the model and no theta0 snapshot are needed: the expansion point is the
    frozen base weights, whatever lora_B holds, so there is no bind-time
    lora_B == 0 requirement either.
    """
    params = dict(model.named_parameters())
    non_lora_trainable = [n for n, p in params.items() if p.requires_grad and not is_lora_parameter_name(n)]
    if non_lora_trainable:
        raise ValueError(
            "linearization impl 'lowrank' linearizes along the LoRA factors only, but these trainable "
            f"params are not LoRA factors: {non_lora_trainable[:3]}. Use impl='dense'."
        )
    lora_b_names = [n for n, p in params.items() if "lora_B" in n and p.requires_grad]
    if not lora_b_names:
        raise RuntimeError("No trainable lora_B parameters found for the lowrank linearized forward.")

    original_forward = model.forward
    inside = {"active": False}

    def _linearized_forward(*args: Any, **kwargs: Any) -> Any:
        # functional_call re-enters model.forward; the inner call is the plain PEFT forward.
        if inside["active"]:
            return original_forward(*args, **kwargs)

        first_tensor = next(
            (v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)),
            None,
        )
        if first_tensor is None:
            raise ValueError("The lowrank linearized forward needs at least one tensor input.")
        live = dict(model.named_parameters())
        lora_b = {n: live[n] for n in lora_b_names}

        def _g(eps: torch.Tensor) -> torch.Tensor:
            out = functional_call(
                model, {n: eps * b for n, b in lora_b.items()}, args=args, kwargs=kwargs, strict=False
            )
            return output_transform(out) if output_transform is not None else out

        eps0 = torch.zeros((), device=first_tensor.device, dtype=torch.float32)
        inside["active"] = True
        try:
            with forward_ad_safe_attention_context(first_tensor.device):
                f0, df = jvp(_g, (eps0,), (torch.ones_like(eps0),))
        finally:
            inside["active"] = False
        out = f0 + df
        return output_builder(out) if output_builder is not None else out

    model.forward = _linearized_forward  # type: ignore[method-assign]
    model.forward_mode_name = "linearized_ntk"  # type: ignore[attr-defined]
    model._ntk_linearized = True  # type: ignore[attr-defined]
    model._ntk_linearization_impl = "lowrank"  # type: ignore[attr-defined]
    return {"linearized_params": len(lora_b_names), "linearized_buffers": 0}
