from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def attach_full_rank_deltas(model: nn.Module) -> nn.ParameterDict:
    """Freeze every parameter and give each its own trainable dW (zeros, same dtype) in ``model.ntk_delta``.

    The full-rank counterpart of LoRA hosts: the linearized forward expands around the frozen W0 with dW as
    the jvp tangent, unscaled and uncast. Keys are the parameter names with '.' -> '/' (ParameterDict keys
    cannot contain dots); tied weights appear once, as in ``named_parameters``.
    """
    named = list(model.named_parameters())
    for _, p in named:
        p.requires_grad_(False)
    model.ntk_delta = nn.ParameterDict({n.replace(".", "/"): nn.Parameter(torch.zeros_like(p)) for n, p in named})
    return model.ntk_delta


class _LayerJvp(torch.autograd.Function):
    """One decoder layer, (h, h_t) -> (h', h'_t), with the jvp recomputed in backward (activation checkpointing).

    Only (h, h_t) are kept between forward and backward. Gradients flow to h_t and the layer's dW only: the
    primal h depends on the frozen W0 and the input alone, so nothing trainable sits upstream of it.
    """

    @staticmethod
    def _jvp(layer, kw, local, h, h_t, ds):
        def f(x, *p):
            return functional_call(layer, dict(zip(local, p, strict=True)), args=(x,), kwargs=kw, strict=False)

        with forward_ad_safe_attention_context(h.device):
            return jvp(f, (h, *(layer.get_parameter(n) for n in local)), (h_t, *ds))

    @staticmethod
    def forward(ctx, layer, kw, local, h, h_t, *ds):
        ctx.layer, ctx.kw, ctx.local = layer, kw, local
        ctx.save_for_backward(h, h_t, *ds)
        out, out_t = _LayerJvp._jvp(layer, kw, local, h, h_t, ds)
        ctx.mark_non_differentiable(out)
        return out, out_t

    @staticmethod
    def backward(ctx, _g_out, g_out_t):
        h, h_t, *ds = ctx.saved_tensors
        h_t = h_t.detach().requires_grad_(True)
        ds = [d.detach().requires_grad_(True) for d in ds]
        with torch.enable_grad():
            _, out_t = _LayerJvp._jvp(ctx.layer, ctx.kw, ctx.local, h, h_t, ds)
            grads = torch.autograd.grad(out_t, (h_t, *ds), g_out_t, allow_unused=True)
        return (None, None, None, None, *grads)


def _layerwise_linearized_logits(
    model: nn.Module, tangents: dict[str, torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """f(x; W0) + J(x; W0) . dW for a HF Llama, one checkpointed jvp per decoder layer.

    Mirrors ``LlamaModel.forward`` (embedding, causal mask, rotary embeddings, layers, norm, lm_head). The
    primals are the live (frozen) weights; ``tangents`` maps parameter names to dW. Activation memory is one
    layer's worth plus the (h, h_t) at each layer boundary, instead of the whole network's.
    """
    from transformers.masking_utils import create_causal_mask

    # shortcut: Llama only (one full-attention mask for every layer); other decoders need their own mask setup.
    if getattr(model.config, "model_type", None) != "llama":
        raise NotImplementedError("strategy.recompute supports Llama models only.")
    body = model.model
    emb_name = "model.embed_tokens.weight"
    head_name = "lm_head.weight" if "lm_head.weight" in dict(model.named_parameters()) else emb_name  # tied
    outside = {emb_name, "model.norm.weight", head_name}
    stray = [n for n in tangents if n not in outside and not n.startswith("model.layers.")]
    if stray:
        raise NotImplementedError(f"strategy.recompute: no layer-wise jvp for {stray[:3]}")

    def tangent_or_zeros(name: str) -> torch.Tensor:
        return tangents[name] if name in tangents else torch.zeros_like(model.get_parameter(name))

    emb = model.get_parameter(emb_name)
    h = F.embedding(input_ids, emb)
    h_t = F.embedding(input_ids, tangent_or_zeros(emb_name))
    pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    kw = {
        "attention_mask": create_causal_mask(
            config=model.config, inputs_embeds=h, attention_mask=attention_mask, past_key_values=None, position_ids=pos
        ),
        "position_embeddings": body.rotary_emb(h, position_ids=pos),
        "position_ids": pos,
    }
    for i, layer in enumerate(body.layers):
        prefix = f"model.layers.{i}."
        local = [n[len(prefix) :] for n in tangents if n.startswith(prefix)]
        h, h_t = _LayerJvp.apply(layer, kw, local, h, h_t, *(tangents[prefix + n] for n in local))

    def head(x: torch.Tensor, norm_w: torch.Tensor, head_w: torch.Tensor) -> torch.Tensor:
        return F.linear(functional_call(body.norm, {"weight": norm_w}, (x,)), head_w)

    norm_w, head_w = model.get_parameter("model.norm.weight"), model.get_parameter(head_name)
    with forward_ad_safe_attention_context(h.device):
        f0, df = jvp(
            head, (h, norm_w, head_w), (h_t, tangent_or_zeros("model.norm.weight"), tangent_or_zeros(head_name))
        )
    return f0 + df


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
    recompute: bool = False,
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
      * Full-rank dW (`attach_full_rank_deltas`, causal `strategy: full`): primal =
        the frozen W0, tangent = the bf16 dW parameter itself.
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

    # Full-rank dW (attach_full_rank_deltas): (base weight name, dW param).
    full_rank = [(k.replace("/", "."), d) for k, d in getattr(model, "ntk_delta", nn.ParameterDict()).items()]
    generic = [
        n for n, p in named.items() if p.requires_grad and not is_lora_parameter_name(n) and not n.startswith("ntk_delta.")
    ]
    if not hosts and not generic and not full_rank:
        raise RuntimeError("No trainable parameters found for linearized_ntk forward mode.")
    for host_name in [h for h, _, _ in hosts] + [h for h, _ in full_rank]:
        if named[host_name].requires_grad:
            raise RuntimeError(f"{host_name} is trainable; the weights a delta is added to must be frozen.")

    if recompute and (hosts or generic or not full_rank):
        raise NotImplementedError("strategy.recompute is implemented for full-rank dW (strategy: full) only.")
    generic_p0 = {n: named[n].detach().clone() for n in generic}
    names = [h for h, _, _ in hosts] + generic + [h for h, _ in full_rank]

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
        if recompute:
            out = _layerwise_linearized_logits(
                model, dict(full_rank), kwargs["input_ids"], kwargs.get("attention_mask")
            )
            return output_builder(out) if output_builder is not None else out

        live = dict(model.named_parameters())
        primals = (
            tuple(live[h] for h, _, _ in hosts)
            + tuple(generic_p0[n] for n in generic)
            + tuple(live[h] for h, _ in full_rank)
        )
        tangents = (
            tuple(_lora_delta(mod, ad, live[h].dtype) for h, mod, ad in hosts)
            + tuple(live[n] - generic_p0[n] for n in generic)
            + tuple(d for _, d in full_rank)
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
