"""
Custom SVHN preprocessing ported from ``rebasin_linear`` (``src/datasets/svhn.py``).

SVHN digits are natively 32x32. The stock CLIP pipeline resizes them straight to the
backbone input size (224) and center-crops, which blurs the digit badly. This pipeline
instead resizes to a smaller ``target_size`` (96) and zero-pads it back to the backbone
input size, keeping the digit sharp on a black canvas, and applies an augmentation policy
to the train split only.

The override is **opt-in** and off by default: with ``SVHN_CUSTOM_PREPROCESS`` disabled,
:func:`maybe_svhn_transforms` returns its arguments unchanged, so behaviour is identical
to a repo without this module.

It is wired into ``build_vision_loaders``, which is the single place where the dataset
identity and the transform meet, so training and every eval entrypoint share the same
pipeline. Because it changes input *geometry*, a checkpoint trained with the override on
must also be evaluated with it on.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode

Transform = Callable[[Any], Any]

SVHN_HF_PATH = "ufldl-stanford/svhn"

# Master switch. Env override: MR_SVHN_PREPROCESS=1
SVHN_CUSTOM_PREPROCESS = False
# Size the digit is resized to before being padded back to the backbone input size.
# Env override: MR_SVHN_TARGET_SIZE
SVHN_TARGET_SIZE = 96
# Augmentation applied to the train split. Env override: MR_SVHN_AUGMENTATION
SVHN_AUGMENTATION = "autoaugment"

# open_clip / OpenAI CLIP normalization, used only when the incoming preprocess
# does not expose a Normalize we can reuse.
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_INPUT_SIZE = 224

_BICUBIC = InterpolationMode.BICUBIC

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_ANNOUNCED: set[str] = set()


@dataclass(frozen=True)
class ConvertRGB:
    """
    PIL ``convert("RGB")`` as a picklable callable.

    The original code used ``transforms.Lambda``, which cannot be pickled and therefore
    breaks DataLoader workers under the spawn start method. Mirrors the existing
    ``EMNISTFixTransform`` pattern in ``vision_loaders``.
    """

    def __call__(self, img: Any) -> Any:
        return img.convert("RGB")


def _augmentation_affine_jitter_strong() -> list[Transform]:
    return [
        transforms.RandomAffine(
            degrees=8,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            interpolation=_BICUBIC,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    ]


def _augmentation_affine_jitter_light() -> list[Transform]:
    return [
        transforms.RandomAffine(
            degrees=4,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
            shear=(-5, 5),
            interpolation=_BICUBIC,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.075, hue=0.015),
    ]


def _augmentation_autoaugment() -> list[Transform]:
    return [
        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.SVHN,
            interpolation=_BICUBIC,
            fill=0,
        )
    ]


def _augmentation_randaugment_2_5() -> list[Transform]:
    return [transforms.RandAugment(num_ops=2, magnitude=5, interpolation=_BICUBIC, fill=0)]


def _augmentation_randaugment_3_5() -> list[Transform]:
    return [transforms.RandAugment(num_ops=3, magnitude=5, interpolation=_BICUBIC, fill=0)]


def _augmentation_randaugment_3_7() -> list[Transform]:
    return [transforms.RandAugment(num_ops=3, magnitude=7, interpolation=_BICUBIC, fill=0)]


def _augmentation_photometric() -> list[Transform]:
    return [
        transforms.RandomAffine(
            degrees=8,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            shear=(-10, 10),
            interpolation=_BICUBIC,
            fill=0,
        ),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomInvert(p=0.1),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    ]


def _augmentation_none() -> list[Transform]:
    return []


# Names map onto AUGMENTATION_CONFIG 0..6 of the original svhn.py, plus "none".
AUGMENTATIONS: dict[str, Callable[[], list[Transform]]] = {
    "affine_jitter_strong": _augmentation_affine_jitter_strong,  # AUGMENTATION_CONFIG = 0
    "affine_jitter_light": _augmentation_affine_jitter_light,  # AUGMENTATION_CONFIG = 1
    "autoaugment": _augmentation_autoaugment,  # AUGMENTATION_CONFIG = 2
    "randaugment_2_5": _augmentation_randaugment_2_5,  # AUGMENTATION_CONFIG = 3
    "randaugment_3_5": _augmentation_randaugment_3_5,  # AUGMENTATION_CONFIG = 4
    "randaugment_3_7": _augmentation_randaugment_3_7,  # AUGMENTATION_CONFIG = 5
    "photometric": _augmentation_photometric,  # AUGMENTATION_CONFIG = 6
    "none": _augmentation_none,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(f"{name} must be one of: {sorted(_TRUTHY | _FALSY)}. Got {raw!r}.")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Got {raw!r}.") from exc


def svhn_preprocess_enabled() -> bool:
    return _env_bool("MR_SVHN_PREPROCESS", SVHN_CUSTOM_PREPROCESS)


def svhn_target_size() -> int:
    size = _env_int("MR_SVHN_TARGET_SIZE", SVHN_TARGET_SIZE)
    if size <= 0:
        raise ValueError(f"SVHN target size must be positive. Got {size}.")
    return size


def svhn_augmentation_name() -> str:
    name = str(os.environ.get("MR_SVHN_AUGMENTATION", SVHN_AUGMENTATION)).strip()
    if name not in AUGMENTATIONS:
        raise ValueError(f"SVHN augmentation must be one of: {sorted(AUGMENTATIONS)}. Got {name!r}.")
    return name


def _iter_transforms(preprocess: Transform | None) -> list[Any]:
    return list(getattr(preprocess, "transforms", []) or [])


def _extract_normalize(preprocess: Transform | None) -> transforms.Normalize:
    """Reuse the backbone's own normalization (laion/datacomp differ from OpenAI stats)."""
    for transform in _iter_transforms(preprocess):
        if isinstance(transform, transforms.Normalize):
            return transform
    return transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)


def _as_single_size(size: Any) -> int | None:
    if isinstance(size, int):
        return size
    if isinstance(size, (list, tuple)) and size:
        first = size[0]
        if isinstance(first, int):
            return first
    return None


def _infer_input_size(preprocess: Transform | None) -> int:
    """Backbone input resolution, read off the incoming preprocess (224, 336, ...)."""
    found = _iter_transforms(preprocess)
    for transform in found:
        if isinstance(transform, transforms.CenterCrop):
            size = _as_single_size(transform.size)
            if size is not None:
                return size
    for transform in found:
        if isinstance(transform, transforms.Resize):
            size = _as_single_size(transform.size)
            if size is not None:
                return size
    return DEFAULT_INPUT_SIZE


def build_svhn_transforms(
    *,
    preprocess: Transform | None,
    augmentation: str,
    target_size: int,
) -> tuple[Transform, Transform]:
    """
    Build the (train, eval) SVHN transforms.

    Operation order matters and matches the original: convert RGB -> resize to
    ``target_size`` -> augment -> pad back to the backbone input size -> ToTensor ->
    normalize. The augmentation therefore only ever sees the resized digit, never the
    padded border, and the border is normalized like any other pixel (it ends up at
    ``-mean/std``, not 0).
    """
    if augmentation not in AUGMENTATIONS:
        raise ValueError(f"SVHN augmentation must be one of: {sorted(AUGMENTATIONS)}. Got {augmentation!r}.")

    input_size = _infer_input_size(preprocess)
    if target_size > input_size:
        raise ValueError(f"SVHN target size must be <= backbone input size {input_size}. Got {target_size}.")

    total_padding = input_size - target_size
    padding_left = total_padding // 2
    padding_top = total_padding // 2
    padding = (
        padding_left,
        padding_top,
        total_padding - padding_left,
        total_padding - padding_top,
    )

    normalize = _extract_normalize(preprocess)
    prefix: list[Transform] = [
        ConvertRGB(),
        transforms.Resize((target_size, target_size), interpolation=_BICUBIC, antialias=True),
    ]
    suffix: list[Transform] = [
        transforms.Pad(padding, fill=0),
        transforms.ToTensor(),
        normalize,
    ]

    train_transform = transforms.Compose(prefix + AUGMENTATIONS[augmentation]() + suffix)
    eval_transform = transforms.Compose(prefix + suffix)
    return train_transform, eval_transform


def maybe_svhn_transforms(
    hf_path: str | None,
    train_transform: Transform | None,
    eval_preprocess: Transform | None,
) -> tuple[Transform | None, Transform | None]:
    """
    Swap in the custom SVHN pipeline, or pass the arguments straight through.

    Returns ``(train_transform, eval_preprocess)`` unchanged for every dataset other than
    SVHN, and for SVHN when the override is disabled.
    """
    if hf_path != SVHN_HF_PATH or not svhn_preprocess_enabled():
        return train_transform, eval_preprocess

    target_size = svhn_target_size()
    augmentation = svhn_augmentation_name()
    svhn_train, svhn_eval = build_svhn_transforms(
        preprocess=eval_preprocess,
        augmentation=augmentation,
        target_size=target_size,
    )

    announcement = (
        f"[SVHN] custom preprocess: target={target_size} "
        f"input={_infer_input_size(eval_preprocess)} augmentation={augmentation}"
    )
    if announcement not in _ANNOUNCED:
        _ANNOUNCED.add(announcement)
        print(announcement)

    return svhn_train, svhn_eval
