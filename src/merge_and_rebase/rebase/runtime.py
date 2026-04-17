from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..eval.utils import build_grad_dataloader
from ..models.grad_recipes import clip_contrastive_recipe
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from .base import TensorDict


def average_scores(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class PerTaskAlphaTracker:
    task_names: list[str]
    initial_alpha: float = 0.0
    best_alpha: list[float] = field(init=False)
    best_baseline_acc: list[float] = field(init=False)
    best_rebase_acc: list[float] = field(init=False)
    active: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.task_names)
        self.best_alpha = [float(self.initial_alpha)] * n
        self.best_baseline_acc = [0.0] * n
        self.best_rebase_acc = [float("-inf")] * n
        self.active = [True] * n

    def active_indices(self) -> list[int]:
        return [i for i, is_active in enumerate(self.active) if is_active]

    def update(
        self,
        *,
        alpha: float,
        indices: list[int],
        baseline_accs: list[float],
        rebase_accs: list[float],
    ) -> list[int]:
        if len(indices) != len(baseline_accs) or len(indices) != len(rebase_accs):
            raise ValueError("indices, baseline_accs, and rebase_accs must have the same length.")

        stopped: list[int] = []
        eps = 1e-12
        for idx, baseline_acc, rebase_acc in zip(indices, baseline_accs, rebase_accs, strict=True):
            baseline_acc = float(baseline_acc)
            rebase_acc = float(rebase_acc)
            best_rebase_acc = float(self.best_rebase_acc[idx])
            if rebase_acc > best_rebase_acc + eps:
                self.best_alpha[idx] = float(alpha)
                self.best_baseline_acc[idx] = baseline_acc
                self.best_rebase_acc[idx] = rebase_acc
                continue
            if rebase_acc + eps >= best_rebase_acc:
                continue
            self.active[idx] = False
            stopped.append(idx)
        return stopped

    def best_avg(self) -> float:
        vals = [v for v in self.best_rebase_acc if v != float("-inf")]
        return average_scores(vals)


def resolve_rebase_method_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    method_name = str(cfg.get("method", "gradfix"))
    method_params = cfg.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("config['method_params'] must be a dict when provided.")

    out = dict(method_params)
    if method_name == "gradfix":
        for legacy_key in ("mask_mode", "vote"):
            if legacy_key not in out and cfg.get(legacy_key) is not None:
                out[legacy_key] = cfg[legacy_key]
    return method_name, out


def format_rebase_method_label(method_name: str, method_params: dict[str, Any]) -> str:
    if method_name == "gradfix":
        mask_mode = str(method_params.get("mask_mode", "normal"))
        vote = str(method_params.get("vote", "mean"))
        return f"gradfix(mask={mask_mode}, vote={vote})"
    if method_name == "theseus":
        batches = int(method_params.get("num_batches", 1))
        seq_align = str(method_params.get("seq_align", "interpolate2d"))
        return f"theseus(batches={batches}, align={seq_align})"
    if method_name == "theseus_reference":
        batches = int(method_params.get("num_batches", 1))
        token_strategy = str(method_params.get("token_strategy", "interpolate_2d"))
        transport = str(method_params.get("method", "svd"))
        return f"theseus_reference(batches={batches}, token={token_strategy}, transport={transport})"
    return method_name


def transport_vision_task_vector(
    *,
    method: Any,
    source_base: dict[str, torch.Tensor],
    target_base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    clf_source: OpenClipClassifier,
    clf_target: OpenClipClassifier,
    task_name: str,
    source_loaders: Any | None,
    loaders: Any,
    classnames: list[str],
    build_cfg_task: OpenClipBuildConfig,
    device: str,
    strict: bool,
    method_params: dict[str, Any],
    grad_batch_size: int | None,
    grad_imgs_per_class: int | None,
    grad_num_batches: int | None,
    num_workers: int,
    seed: int,
) -> TensorDict:
    method_name = getattr(method, "name", None)
    if method_name not in {"gradfix", "theseus", "theseus_reference"}:
        return method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            strict=strict,
            **method_params,
        )

    if method_name in {"theseus", "theseus_reference"}:
        if source_loaders is None:
            raise ValueError("Theseus vision transport requires source_loaders.")

        theseus_batch_size = int(
            method_params.get("batch_size", grad_batch_size or getattr(loaders.train, "batch_size", 32))
        )
        theseus_num_batches = int(method_params.get("num_batches", grad_num_batches or 1))
        source_loader = source_loaders.train
        target_loader = loaders.train
        n_support = min(len(source_loader.dataset), len(target_loader.dataset))

        theseus_kwargs = dict(method_params)
        if theseus_kwargs.get("batch_size") is None:
            theseus_kwargs["batch_size"] = theseus_batch_size
        if theseus_kwargs.get("num_batches") is None:
            theseus_kwargs["num_batches"] = theseus_num_batches
        if theseus_kwargs.get("seed") is None:
            theseus_kwargs["seed"] = int(seed)

        print(
            f"  {task_name}: Theseus support set - {n_support} samples, "
            f"batch_size={theseus_batch_size}, num_batches={theseus_num_batches}, "
            f"mode={method_name}, "
            f"align={method_params.get('seq_align', method_params.get('token_strategy', 'interpolate_2d'))}"
        )
        return method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            strict=strict,
            source_model=clf_source.model,
            target_model=clf_target.model,
            source_dataloader=source_loader,
            target_dataloader=target_loader,
            device=device,
            **theseus_kwargs,
        )
    elif method_name == "gradfix":
        vote = str(method_params.get("vote", "mean"))
        grad_loader = build_grad_dataloader(
            loaders.train,
            loaders.train.dataset,
            grad_batch_size=grad_batch_size,
            grad_imgs_per_class=grad_imgs_per_class,
            grad_num_batches=grad_num_batches,
            num_workers=num_workers,
            seed=seed,
        )
        n_grad_samples = len(grad_loader.dataset) if hasattr(grad_loader, "dataset") else "?"
        print(
            f"  {task_name}: grad dataloader - {n_grad_samples} samples ({grad_imgs_per_class} per class), "
            f"batch_size={getattr(grad_loader, 'batch_size', '?')}"
        )

        recipe = clip_contrastive_recipe(
            clf_target,
            classnames,
            build_cfg_task,
            device=device,
            reduction="none" if vote in {"majority", "max"} else "mean",
        )
        return method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            strict=strict,
            target_model=clf_target.model,
            target_dataloader=grad_loader,
            recipe=recipe,
            device=device,
            **method_params,
        )
