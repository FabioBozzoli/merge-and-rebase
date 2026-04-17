from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockExtensionConfig:
    blocks_to_add: int | None = None
    insertion_order: str = "bottom-top"
    extension_density: str = "spread"
    extension_strategy: str = "interpolate"
    dampening_factor: float = 1.0
    n_batches_act: int = 2
    calibration_split: str = "test"
    skip_ln_correction: bool = False
    skip_final_ln: bool = False
    eval_before_extension: bool = False
    first_n_eval_batches: int | None = None
    verbose: bool = True
    show_progress: bool = True


def resolve_block_extension_config(cfg: Mapping[str, Any]) -> tuple[bool, BlockExtensionConfig]:
    raw_params = cfg.get("block_extension_params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raise ValueError("config['block_extension_params'] must be a dict when provided.")

    params = dict(raw_params)
    enabled_raw = cfg.get("block_extension_enabled", None)
    enabled = bool(enabled_raw) if enabled_raw is not None else bool(params)

    return enabled, BlockExtensionConfig(
        blocks_to_add=_as_optional_int(params.get("blocks_to_add", None)),
        insertion_order=str(params.get("insertion_order", "bottom-top")),
        extension_density=str(params.get("extension_density", "spread")),
        extension_strategy=str(params.get("extension_strategy", "interpolate")),
        dampening_factor=float(params.get("dampening_factor", 1.0)),
        n_batches_act=max(1, int(params.get("n_batches_act", 2))),
        calibration_split=str(params.get("calibration_split", "test")),
        skip_ln_correction=bool(params.get("skip_ln_correction", False)),
        skip_final_ln=bool(params.get("skip_final_ln", False)),
        eval_before_extension=bool(params.get("eval_before_extension", False)),
        first_n_eval_batches=_as_optional_int(params.get("first_n_eval_batches", None)),
        verbose=bool(params.get("verbose", True)),
        show_progress=bool(params.get("show_progress", True)),
    )


def select_loader(split: str, train_loader: Iterable[Any], test_loader: Iterable[Any], val_loader: Iterable[Any] | None):
    if split == "train":
        return train_loader
    if split == "val" and val_loader is not None:
        return val_loader
    return test_loader


class InputAlignedBlock(nn.Module):
    def __init__(self, original_block: nn.Module, dim: int):
        super().__init__()
        ref_weight = _extract_ln_weight(original_block)
        self.aligner = nn.Linear(dim, dim, bias=True).to(ref_weight.device)
        self.reset_aligner()

        object.__setattr__(self, "_orig_block", original_block)

        for name, module in original_block.named_children():
            setattr(self, name, module)

        for name, param in original_block.named_parameters(recurse=False):
            self.register_parameter(name, param)
        for name, buf in original_block.named_buffers(recurse=False):
            self.register_buffer(name, buf)

    @torch.no_grad()
    def reset_aligner(self):
        eye = torch.eye(self.aligner.weight.shape[0], device=self.aligner.weight.device, dtype=self.aligner.weight.dtype)
        self.aligner.weight.copy_(eye)
        self.aligner.bias.zero_()

    def forward(self, x, attn_mask=None, **kwargs):
        x = self.aligner(x)
        self._orig_block.train(self.training)
        return self._orig_block(x, attn_mask=attn_mask, **kwargs)


class InputAlignedFinalLayer(nn.Module):
    def __init__(self, original_ln: nn.Module, dim: int):
        super().__init__()
        self.aligner = nn.Linear(dim, dim, bias=True).to(original_ln.weight.device)
        self.reset_aligner()

        self.normalized_shape = original_ln.normalized_shape
        self.eps = original_ln.eps
        self.elementwise_affine = original_ln.elementwise_affine
        if self.elementwise_affine:
            self.weight = original_ln.weight
            self.bias = original_ln.bias

    @torch.no_grad()
    def reset_aligner(self):
        eye = torch.eye(self.aligner.weight.shape[0], device=self.aligner.weight.device, dtype=self.aligner.weight.dtype)
        self.aligner.weight.copy_(eye)
        self.aligner.bias.zero_()

    def forward(self, x):
        x = self.aligner(x)
        if self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps)


class BlockExtender:
    def __init__(
        self,
        model_base: nn.Module,
        model_ft: nn.Module,
        device: str | torch.device,
        *,
        verbose: bool = True,
        show_progress: bool = True,
    ):
        self.model_base = model_base
        self.model_ft = model_ft
        self.device = device
        self.reference_inputs: dict[str, dict[str, torch.Tensor]] = {"base": {}, "ft": {}}
        self.verbose = bool(verbose)
        self.show_progress = bool(show_progress)

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"[block_extension] {message}")

    @staticmethod
    def _inner_block(block: nn.Module) -> nn.Module:
        return block.block if hasattr(block, "block") else block

    @staticmethod
    def _store_input_hook(store: dict[str, list[torch.Tensor]], key: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                store[key].append(inputs[0].detach().cpu())

        return hook

    @staticmethod
    def _fit_ridge(A: torch.Tensor, T: torch.Tensor, lambda_reg: float = 1e-6):
        A = A.float()
        T = T.float()

        mu_A = A.mean(dim=0)
        mu_T = T.mean(dim=0)

        A_c = A - mu_A
        T_c = T - mu_T

        dim_in = A.shape[1]
        cov = A_c.T @ A_c
        cov = cov + lambda_reg * torch.eye(dim_in, device=A.device, dtype=A.dtype)
        rhs = A_c.T @ T_c

        try:
            W_T = torch.linalg.solve(cov, rhs)
        except RuntimeError:
            W_T = torch.linalg.pinv(cov) @ rhs

        b = mu_T - mu_A @ W_T
        return W_T.T, b

    @staticmethod
    def _match_rows(A: torch.Tensor, T: torch.Tensor):
        n = min(A.shape[0], T.shape[0])
        return A[:n], T[:n]

    @torch.no_grad()
    def wrap_with_aligners(self):
        for model in [self.model_base, self.model_ft]:
            wrapped_blocks: list[nn.Module] = []
            for block in model.visual.transformer.resblocks:
                if isinstance(block, InputAlignedBlock):
                    wrapped_blocks.append(block)
                    continue
                inner = self._inner_block(block)
                dim = _extract_ln_weight(inner).shape[0]
                wrapped_blocks.append(InputAlignedBlock(block, dim))
            model.visual.transformer.resblocks = nn.ModuleList(wrapped_blocks)

            if not isinstance(model.visual.ln_post, InputAlignedFinalLayer):
                ln = model.visual.ln_post
                dim = ln.weight.shape[0]
                model.visual.ln_post = InputAlignedFinalLayer(ln, dim)

    @torch.no_grad()
    def capture_reference_inputs(self, loader: Iterable[Any], n_batches: int):
        for name, model in [("base", self.model_base), ("ft", self.model_ft)]:
            self._vprint(f"capture reference inputs ({name}) with n_batches={n_batches}")
            model.eval()
            store: dict[str, list[torch.Tensor]] = defaultdict(list)
            hooks = []

            for i, block in enumerate(model.visual.transformer.resblocks):
                hooks.append(block.register_forward_hook(self._store_input_hook(store, f"{i}.input")))
            hooks.append(model.visual.ln_post.register_forward_hook(self._store_input_hook(store, "final.input")))

            it = iter(loader)
            for _ in _iter_with_progress(
                range(n_batches),
                total=n_batches,
                desc=f"block_extension.capture.{name}",
                enabled=self.show_progress,
            ):
                try:
                    images, _ = next(it)
                except StopIteration:
                    break
                _encode_image(model, images.to(self.device))

            for h in hooks:
                h.remove()

            refs: dict[str, torch.Tensor] = {}
            for key, tensors in store.items():
                refs[key] = torch.cat(tensors, dim=0).flatten(0, 1)
            self.reference_inputs[name] = refs

    @torch.no_grad()
    def _capture_single_input(self, model: nn.Module, target: int | str, loader: Iterable[Any], n_batches: int):
        model.eval()
        buffers: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any):
            if inputs and inputs[0] is not None:
                buffers.append(inputs[0].detach().cpu())

        if target == "final":
            handle = model.visual.ln_post.register_forward_hook(hook)
        else:
            handle = model.visual.transformer.resblocks[target].register_forward_hook(hook)

        it = iter(loader)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            _encode_image(model, images.to(self.device))

        handle.remove()

        if not buffers:
            return torch.empty(0)
        return torch.cat(buffers, dim=0).flatten(0, 1)

    @torch.no_grad()
    def _interpolate_block_weights(self, target_block: nn.Module, source_block: nn.Module, alpha: float = 0.5):
        target_inner = self._inner_block(target_block)
        source_inner = self._inner_block(source_block)
        source_params = dict(source_inner.named_parameters())
        for name_t, p_t in target_inner.named_parameters():
            if name_t.startswith("aligner."):
                continue
            p_s = source_params.get(name_t)
            if p_s is not None and p_t.shape == p_s.shape:
                p_t.copy_((1.0 - alpha) * p_t + alpha * p_s)

    @torch.no_grad()
    def _dampen_block_output(self, block: nn.Module, factor: float):
        inner = self._inner_block(block)
        if hasattr(inner, "attn") and hasattr(inner.attn, "out_proj"):
            inner.attn.out_proj.weight.mul_(factor)
        if hasattr(inner, "mlp") and hasattr(inner.mlp, "c_proj"):
            inner.mlp.c_proj.weight.mul_(factor)

    @staticmethod
    def _build_duplication_schedule(
        curr_layers: int,
        n_needed: int,
        insertion_order: str,
        extension_density: str,
    ):
        if n_needed <= 0:
            return []

        if insertion_order == "bottom-top":
            priority = list(range(curr_layers))
        elif insertion_order == "top-bottom":
            priority = list(range(curr_layers - 1, -1, -1))
        elif insertion_order == "random":
            priority = list(range(curr_layers))
            np.random.shuffle(priority)
        else:
            raise ValueError(
                "Unsupported insertion_order. Expected one of: bottom-top, top-bottom, random. "
                f"Got: {insertion_order}"
            )

        if not priority:
            return []

        if extension_density == "clump":
            return [priority[0]] * n_needed
        if extension_density != "spread":
            raise ValueError(
                "Unsupported extension_density. Expected one of: spread, clump. " f"Got: {extension_density}"
            )

        schedule = []
        while len(schedule) < n_needed:
            if insertion_order == "random":
                cycle = list(range(curr_layers))
                np.random.shuffle(cycle)
            else:
                cycle = priority
            need = n_needed - len(schedule)
            schedule.extend(cycle[:need])

        return schedule

    @torch.no_grad()
    def _collect_inner_means(
        self,
        model: nn.Module,
        targets: list[tuple[int | str, int | str]],
        loader: Iterable[Any],
        n_batches: int,
    ):
        model.eval()
        store: dict[str, list[torch.Tensor]] = defaultdict(list)
        hooks = []

        for tgt, _ in targets:
            if tgt == "final":
                hooks.append(model.visual.ln_post.register_forward_hook(self._store_input_hook(store, "final_inner")))
            else:
                block = model.visual.transformer.resblocks[tgt]
                hooks.append(self._inner_block(block).register_forward_hook(self._store_input_hook(store, f"{tgt}_inner")))

        it = iter(loader)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            _encode_image(model, images.to(self.device))

        for h in hooks:
            h.remove()

        means = {}
        for key, tensors in store.items():
            if tensors:
                means[key] = torch.cat(tensors, dim=0).mean().item()
        return means

    def _debug_log_alignment_stats(
        self,
        model_name: str,
        model: nn.Module,
        targets: list[tuple[int | str, int | str]],
        loader: Iterable[Any],
        n_batches: int,
    ):
        if not logger.isEnabledFor(logging.DEBUG):
            return

        means = self._collect_inner_means(model, targets, loader, n_batches)
        logger.debug("%s stats (act vs ref):", model_name)
        logger.debug("%-10s | %-10s | %-12s | %-12s", "Target", "Ref", "Mean(Act)", "Mean(Ref)")

        refs = self.reference_inputs[model_name]
        for tgt, src_ref in targets:
            act_key = "final_inner" if tgt == "final" else f"{tgt}_inner"
            ref_key = "final.input" if src_ref == "final" else f"{src_ref}.input"

            if act_key not in means or ref_key not in refs:
                continue

            logger.debug(
                "%-10s | %-10s | %-12.6f | %-12.6f",
                str(tgt),
                str(src_ref),
                means[act_key],
                refs[ref_key].mean().item(),
            )

    @torch.no_grad()
    def extend_and_calibrate(
        self,
        *,
        loader: Iterable[Any],
        n_batches: int,
        strategy: str,
        dampening_factor: float,
        blocks_to_add: int | None,
        target_layers_total: int | None,
        insertion_order: str,
        extension_density: str,
        skip_ln_correction: bool,
        skip_final_ln: bool,
    ) -> int:
        self._vprint("starting extension and calibration")
        self.wrap_with_aligners()
        self._vprint("wrapping with aligners completed")
        self.capture_reference_inputs(loader, n_batches)
        self._vprint("reference activation capture completed")

        curr_layers = len(self.model_base.visual.transformer.resblocks)
        if blocks_to_add is not None:
            n_needed = int(blocks_to_add)
        elif target_layers_total is not None:
            n_needed = int(target_layers_total) - curr_layers
        else:
            n_needed = 0

        if n_needed <= 0:
            logger.info("Block extension: no extension needed.")
            self._vprint("no extension needed")
            return curr_layers

        schedule = self._build_duplication_schedule(
            curr_layers=curr_layers,
            n_needed=n_needed,
            insertion_order=insertion_order,
            extension_density=extension_density,
        )

        logger.info("Block extension planned duplications: %s", schedule)
        self._vprint(f"planned duplications: {schedule}")

        orig_base = list(self.model_base.visual.transformer.resblocks)
        orig_ft = list(self.model_ft.visual.transformer.resblocks)

        chain_base = [{"mod": b, "orig_idx": i} for i, b in enumerate(orig_base)]
        chain_ft = [{"mod": b, "orig_idx": i} for i, b in enumerate(orig_ft)]

        step_iter = _iter_with_progress(
            enumerate(schedule, start=1),
            total=len(schedule),
            desc="block_extension.extend",
            enabled=self.show_progress,
        )
        for step, src_idx in step_iter:
            logger.info("Block extension step %d/%d. Source block: %d", step, len(schedule), src_idx)
            self._vprint(f"step {step}/{len(schedule)} source_block={src_idx}")

            dup_base = deepcopy(orig_base[src_idx])
            dup_ft = deepcopy(orig_ft[src_idx])
            dup_base.reset_aligner()
            dup_ft.reset_aligner()

            if strategy == "interpolate":
                src_next = min(src_idx + 1, len(orig_base) - 1)
                self._interpolate_block_weights(dup_base, orig_base[src_next], alpha=0.5)
                self._interpolate_block_weights(dup_ft, orig_ft[src_next], alpha=0.5)
            elif strategy != "duplicate":
                raise ValueError(
                    "Unsupported extension_strategy. Expected one of: duplicate, interpolate. " f"Got: {strategy}"
                )

            if dampening_factor < 1.0:
                self._dampen_block_output(dup_base, dampening_factor)
                self._dampen_block_output(dup_ft, dampening_factor)

            insert_pos = -1
            for i, item in enumerate(chain_base):
                if item["orig_idx"] == src_idx:
                    insert_pos = i
            insert_pos += 1

            chain_base.insert(insert_pos, {"mod": dup_base, "orig_idx": src_idx})
            chain_ft.insert(insert_pos, {"mod": dup_ft, "orig_idx": src_idx})

            self.model_base.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_base])
            self.model_ft.visual.transformer.resblocks = nn.ModuleList([x["mod"] for x in chain_ft])

            if skip_ln_correction:
                self._vprint("skip_ln_correction enabled, skipping aligner fit for this step")
                continue

            targets = [(i, chain_base[i]["orig_idx"]) for i in range(insert_pos + 1, len(chain_base))]
            if not skip_final_ln:
                targets.append(("final", "final"))
            self._vprint(f"calibrating {len(targets)} targets")

            for model_name, model in [("base", self.model_base), ("ft", self.model_ft)]:
                self._debug_log_alignment_stats(model_name, model, targets, loader, n_batches)

            for model_name, model in [("base", self.model_base), ("ft", self.model_ft)]:
                for tgt, src_ref in targets:
                    current = self._capture_single_input(model, tgt, loader, n_batches)
                    if current.numel() == 0:
                        continue

                    ref_key = "final.input" if src_ref == "final" else f"{src_ref}.input"
                    ref = self.reference_inputs[model_name].get(ref_key)
                    if ref is None:
                        continue

                    A, T = self._match_rows(current, ref)
                    if A.numel() == 0 or T.numel() == 0:
                        continue

                    W, b = self._fit_ridge(A, T)
                    module = model.visual.ln_post if tgt == "final" else model.visual.transformer.resblocks[tgt]
                    module.aligner.weight.copy_(W.to(module.aligner.weight.device, dtype=module.aligner.weight.dtype))
                    module.aligner.bias.copy_(b.to(module.aligner.bias.device, dtype=module.aligner.bias.dtype))

            for model_name, model in [("base", self.model_base), ("ft", self.model_ft)]:
                self._debug_log_alignment_stats(model_name, model, targets, loader, n_batches)

        final_depth = len(self.model_base.visual.transformer.resblocks)
        self._vprint(f"extension completed. final_depth={final_depth}")
        return final_depth


@torch.no_grad()
def run_block_extension(
    *,
    source_base_model: nn.Module,
    source_ft_model: nn.Module,
    calibration_loader: Iterable[Any],
    target_layers_total: int | None,
    config: BlockExtensionConfig,
    device: str | torch.device,
) -> int:
    extender = BlockExtender(
        source_base_model,
        source_ft_model,
        device,
        verbose=bool(config.verbose),
        show_progress=bool(config.show_progress),
    )
    return extender.extend_and_calibrate(
        loader=calibration_loader,
        n_batches=config.n_batches_act,
        strategy=config.extension_strategy,
        dampening_factor=float(config.dampening_factor),
        blocks_to_add=config.blocks_to_add,
        target_layers_total=target_layers_total,
        insertion_order=config.insertion_order,
        extension_density=config.extension_density,
        skip_ln_correction=bool(config.skip_ln_correction),
        skip_final_ln=bool(config.skip_final_ln),
    )


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _extract_ln_weight(block: nn.Module) -> torch.Tensor:
    if hasattr(block, "ln_1") and hasattr(block.ln_1, "weight"):
        return block.ln_1.weight
    raise AttributeError("Cannot infer transformer block width from ln_1.weight.")


def _encode_image(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_image(images)
    if hasattr(model, "visual") and callable(model.visual):
        return model.visual(images)
    return model(images)


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
