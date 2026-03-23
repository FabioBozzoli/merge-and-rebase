from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Protocol

import torch

from merge_and_rebase.models.openclip_classifier import OpenClipClassifier
from merge_and_rebase.utils.linearization import LinearizedModule


class ForwardMode(Protocol):
    name: str

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
    ) -> None: ...


@dataclass(frozen=True)
class StandardForwardMode:
    name: str = "standard"

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
    ) -> None:
        del base_sd, strict_load
        clf.forward = clf.__class__.forward.__get__(clf, clf.__class__)  # type: ignore[method-assign]
        clf.forward_mode_name = self.name  # type: ignore[attr-defined]


@dataclass(frozen=True)
class LinearizedNtkForwardMode:
    name: str = "linearized_ntk"

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
    ) -> None:
        # Ensure we always linearize from the true class forward, not a previous patch.
        clf.forward = clf.__class__.forward.__get__(clf, clf.__class__)  # type: ignore[method-assign]

        device = next(clf.model.parameters()).device
        ref_visual = deepcopy(clf.model.visual).to(device)
        ref_visual.eval()
        for p in ref_visual.parameters():
            p.requires_grad = False

        visual_base_sd = {
            k[len("visual.") :]: v.to(device=device) for k, v in base_sd.items() if k.startswith("visual.")
        }
        if not visual_base_sd:
            raise ValueError("linearized_ntk forward mode requires base state_dict keys prefixed with 'visual.'.")

        miss, unexp = ref_visual.load_state_dict(visual_base_sd, strict=strict_load)
        if strict_load and (miss or unexp):
            raise RuntimeError(
                f"Failed to load base visual weights for linearized mode. missing={len(miss)}, unexpected={len(unexp)}"
            )

        linearized = LinearizedModule.from_module(ref_visual, copy_module=False)

        def _linearized_forward(self: OpenClipClassifier, images: torch.Tensor) -> torch.Tensor:
            if self._zs_text_features.numel() == 0:
                raise RuntimeError("Call build_zeroshot_text_features() before forward in zero-shot mode.")

            text_feats = self._zs_text_features

            def _postprocess(img_feats: torch.Tensor) -> torch.Tensor:
                if self.normalize:
                    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
                return self.logit_scale * (img_feats @ text_feats.t())

            return linearized.forward(
                current_module=self.model.visual,
                images=images,
                output_transform=_postprocess,
            )

        clf.forward = MethodType(_linearized_forward, clf)  # type: ignore[method-assign]
        clf.forward_mode_name = self.name  # type: ignore[attr-defined]


_FORWARD_MODES: dict[str, ForwardMode] = {
    "standard": StandardForwardMode(),
    "linearized_ntk": LinearizedNtkForwardMode(),
}


def list_forward_modes() -> list[str]:
    return sorted(_FORWARD_MODES.keys())


def get_forward_mode(name: str) -> ForwardMode:
    if name not in _FORWARD_MODES:
        raise KeyError(f"Unknown forward mode '{name}'. Available: {sorted(_FORWARD_MODES)}")
    return _FORWARD_MODES[name]
