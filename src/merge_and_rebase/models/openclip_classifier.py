from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OpenClipBuildConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"  # examples: "openai", "laion2b_s34b_b79k", depends on model
    device: str = "cuda"
    dtype: str | None = None  # "fp16" | "bf16" | "fp32" | None
    normalize: bool = True
    logit_scale: float = 100.0
    prompt_template: str = "a photo of a {}"
    prompt_templates: list[str] | None = None


def _template_id(t) -> str:
    """
    Turn a template into a stable-ish identifier for caching.
    Supports:
      - string templates
      - callables like lambda/class/function
    """
    if isinstance(t, str):
        return t
    if callable(t):
        mod = getattr(t, "__module__", "<?>")
        name = getattr(t, "__qualname__", getattr(t, "__name__", "<callable>"))
        code = getattr(t, "__code__", None)
        # include location if available, helps distinguish lambdas
        if code is not None:
            return f"{mod}:{name}@{code.co_filename}:{code.co_firstlineno}"
        return f"{mod}:{name}"
    return repr(t)


def _cfg_to_stable_dict(cfg: OpenClipBuildConfig) -> dict[str, Any]:
    d = asdict(cfg)

    # prompt_template may be callable in your usage; store identifier
    pt = d.get("prompt_template", None)
    if pt is not None and callable(pt):
        d["prompt_template"] = _template_id(pt)

    # prompt_templates may contain callables; store identifiers
    pts = d.get("prompt_templates", None)
    if pts is not None:
        d["prompt_templates"] = [_template_id(t) for t in list(pts)]

    return d


def _fingerprint(cfg_dict: dict[str, Any], classnames: Sequence[str], normalize: bool) -> str:
    payload = {
        "cfg": cfg_dict,
        "classnames": list(classnames),
        "normalize": bool(normalize),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class OpenClipClassifier(nn.Module):
    """
    Zero-shot classifier using open_clip.
    Forward expects already-preprocessed images: [B, 3, H, W]
    and returns logits [B, C].
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        preprocess,  # callable PIL->Tensor (or any transform)
        *,
        normalize: bool = True,
        logit_scale: float = 100.0,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.normalize = normalize
        self.logit_scale = float(logit_scale)
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)
        self._zs_text_fingerprint: str | None = None

    @staticmethod
    def build(cfg: OpenClipBuildConfig) -> OpenClipClassifier:
        try:
            import open_clip
        except Exception as e:
            raise ImportError("open_clip support requires: pip install -e '.[openclip]'") from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.pretrained,
            device=cfg.device,
            quick_gelu=(cfg.pretrained == "openai"),
        )
        tokenizer = open_clip.get_tokenizer(cfg.model_name)

        # dtype casting
        if cfg.dtype is not None:
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            if cfg.dtype not in dtype_map:
                raise ValueError(f"Unknown dtype {cfg.dtype}. Choose from {sorted(dtype_map)}")
            model = model.to(dtype=dtype_map[cfg.dtype])

        model.eval()
        return OpenClipClassifier(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            normalize=cfg.normalize,
            logit_scale=cfg.logit_scale,
        )

    @torch.no_grad()
    def build_zeroshot_text_features(
        self,
        classnames: list[str],
        cfg: OpenClipBuildConfig,
        *,
        cache_dir: str | None = None,
        cache_tag: str = "zs_text",
        force_rebuild: bool = False,
    ) -> None:
        """
        Build or load zero-shot text features for (cfg, classnames).

        If cache_dir is provided, features are cached to disk under:
        {cache_dir}/{cache_tag}_{fingerprint}.pt
        """
        cfg_dict = _cfg_to_stable_dict(cfg)
        fp = _fingerprint(cfg_dict, classnames, self.normalize)

        if self._zs_text_features.numel() > 0 and not force_rebuild and self._zs_text_fingerprint == fp:
            return  # already built for the same request

        if cache_dir is None:
            self._zs_text_features = self._compute_zeroshot_text_features(classnames, cfg)
            self._zs_text_fingerprint = fp
            return

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        feat_file = cache_path / f"{cache_tag}_{fp}.pt"

        device = next(self.model.parameters()).device

        if feat_file.exists() and not force_rebuild:
            obj = torch.load(feat_file, map_location="cpu")
            feats = obj.get("feats", None)
            meta = obj.get("meta", {})

            if not isinstance(feats, torch.Tensor):
                raise ValueError(f"Invalid cache file (missing 'feats' tensor): {feat_file}")

            if feats.ndim != 2 or feats.shape[0] != len(classnames):
                raise ValueError(
                    f"Cached feats shape mismatch: got {tuple(feats.shape)}, expected ({len(classnames)}, D)"
                )

            if bool(meta.get("normalize", True)) != bool(self.normalize):
                raise ValueError("Cached feats were built with different normalize setting.")

            self._zs_text_features = feats.to(device)
            self._zs_text_fingerprint = fp
            # print(f"Loaded zero-shot text features cache from {feat_file}")
            return

        feats = self._compute_zeroshot_text_features(classnames, cfg).to("cpu")
        meta = {
            "cfg": cfg_dict,
            "classnames": list(classnames),
            "normalize": bool(self.normalize),
            "shape": list(feats.shape),
            "fingerprint": fp,
        }
        torch.save({"feats": feats, "meta": meta}, feat_file)
        print(f"Saved zero-shot text features cache to {feat_file}")

        self._zs_text_features = feats.to(device)
        self._zs_text_fingerprint = fp

    @torch.no_grad()
    def _compute_zeroshot_text_features(self, classnames: list[str], cfg: OpenClipBuildConfig) -> torch.Tensor:
        templates = cfg.prompt_templates or [cfg.prompt_template]
        device = next(self.model.parameters()).device

        all_text_feats = []
        for c in classnames:
            texts = [t(c) for t in templates]

            tokens = self.tokenizer(texts).to(device)
            text_feats = self.model.encode_text(tokens)  # [T, D]

            if self.normalize:
                text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

            text_feats = text_feats.mean(dim=0, keepdim=True)  # [1, D]
            all_text_feats.append(text_feats)

        feats = torch.cat(all_text_feats, dim=0)  # [C, D]
        if self.normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

        return feats

    @staticmethod
    def extract_tuned_text_features_from_checkpoint(
        *,
        obj: Any,
        ckpt_path: str,
    ) -> torch.Tensor | None:
        if not isinstance(obj, dict):
            return None
        feats = obj.get("tuned_text_features", None)
        if feats is None:
            return None
        if not isinstance(feats, torch.Tensor):
            raise ValueError(f"Checkpoint '{ckpt_path}' has non-tensor tuned_text_features.")
        if feats.ndim != 2:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has invalid tuned_text_features shape={tuple(feats.shape)} (expected 2D [C, D])."
            )
        return feats.detach().cpu()

    @torch.no_grad()
    def resolve_eval_text_features(
        self,
        *,
        text_features_source: str,
        classnames: list[str],
        build_cfg: OpenClipBuildConfig,
        tuned_text_features: torch.Tensor | None,
        cache_dir: str | None = "src/.cache/zs_cache",
        force_rebuild_zeroshot: bool = False,
        task_name: str | None = None,
        ckpt_path: str | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor | None, str]:
        source = str(text_features_source).strip().lower()
        if source not in {"auto", "zero_shot", "tuned_ckpt"}:
            raise ValueError("text_features_source must be one of: auto, zero_shot, tuned_ckpt")

        if source == "tuned_ckpt":
            if tuned_text_features is None:
                label = str(task_name or "<task>")
                path = str(ckpt_path or "<unknown>")
                raise ValueError(
                    f"Task '{label}' checkpoint '{path}' has no tuned_text_features. "
                    "Use text_features_source='auto' or 'zero_shot', or provide checkpoints from two-stage finetuning."
                )
            return tuned_text_features.detach().cpu(), "tuned_ckpt"

        if source == "auto" and tuned_text_features is not None:
            print(f"Using tuned_text_features from checkpoint for task '{task_name or '<task>'}'.")
            return tuned_text_features.detach().cpu(), "tuned_ckpt"

        if source == "auto" and verbose and tuned_text_features is None:
            label = str(task_name or "<task>")
            print(f"{label}: tuned_text_features not found in checkpoint, falling back to zero-shot text features.")

        self.build_zeroshot_text_features(
            classnames,
            build_cfg,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild_zeroshot,
        )
        return None, "zero_shot"

    @torch.no_grad()
    def top1_with_text_features(
        self,
        loader,
        *,
        device: str,
        text_features: torch.Tensor,
        expected_num_classes: int | None = None,
    ) -> float:
        if text_features.ndim != 2:
            raise ValueError("text_features must be a 2D matrix [C, D].")
        if expected_num_classes is not None and int(text_features.shape[0]) != int(expected_num_classes):
            raise ValueError(
                "text_features row count mismatch: "
                f"got={int(text_features.shape[0])}, expected={int(expected_num_classes)}"
            )

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        text_feats = text_features.to(device=dev)
        if self.normalize:
            text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

        prev_text = self._zs_text_features
        prev_fingerprint = self._zs_text_fingerprint
        self._zs_text_features = text_feats
        self._zs_text_fingerprint = None
        try:
            return float(self.top1(loader, device=device))
        finally:
            self._zs_text_features = prev_text
            self._zs_text_fingerprint = prev_fingerprint

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        img_feats = self.model.encode_image(images)  # [B, D]
        if self.normalize:
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)

        if self._zs_text_features.numel() == 0:
            raise RuntimeError("Call build_zeroshot_text_features() before forward in zero-shot mode.")

        return self.logit_scale * (img_feats @ self._zs_text_features.t())

    @torch.no_grad()
    def top1(self, loader, device: str) -> float:
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.to(dev)
        self.eval()

        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = self(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(1, total))
