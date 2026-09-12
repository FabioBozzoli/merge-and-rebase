from __future__ import annotations

import pickle

import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from merge_and_rebase.data.svhn_preprocess import (
    SVHN_HF_PATH,
    build_svhn_transforms,
    maybe_svhn_transforms,
)

INPUT_SIZE = 224
TARGET_SIZE = 96
FAKE_MEAN = (0.1, 0.2, 0.3)
FAKE_STD = (0.4, 0.5, 0.6)


def _fake_clip_preprocess() -> transforms.Compose:
    """Shaped like the open_clip eval preprocess, with deliberately non-OpenAI stats."""
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=FAKE_MEAN, std=FAKE_STD),
        ]
    )


def _sample_image() -> Image.Image:
    torch.manual_seed(0)
    pixels = (torch.rand(32, 32, 3) * 255).to(torch.uint8).numpy()
    return Image.fromarray(pixels, mode="RGB")


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MR_SVHN_PREPROCESS", "1")


def test_maybe_svhn_transforms_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MR_SVHN_PREPROCESS", raising=False)
    preprocess = _fake_clip_preprocess()
    train_out, eval_out = maybe_svhn_transforms(SVHN_HF_PATH, preprocess, preprocess)
    assert train_out is preprocess
    assert eval_out is preprocess


def test_maybe_svhn_transforms_is_noop_for_other_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    preprocess = _fake_clip_preprocess()
    train_out, eval_out = maybe_svhn_transforms("tanganke/emnist_mnist", preprocess, preprocess)
    assert train_out is preprocess
    assert eval_out is preprocess


def test_maybe_svhn_transforms_replaces_both_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    preprocess = _fake_clip_preprocess()
    train_out, eval_out = maybe_svhn_transforms(SVHN_HF_PATH, preprocess, preprocess)
    assert train_out is not preprocess
    assert eval_out is not preprocess
    assert eval_out(_sample_image()).shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_svhn_eval_transform_pads_to_input_size() -> None:
    _, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="autoaugment",
        target_size=TARGET_SIZE,
    )
    assert eval_transform(_sample_image()).shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_svhn_eval_transform_border_is_constant() -> None:
    _, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="autoaugment",
        target_size=TARGET_SIZE,
    )
    out = eval_transform(_sample_image())

    pad = (INPUT_SIZE - TARGET_SIZE) // 2
    mask = torch.ones(INPUT_SIZE, INPUT_SIZE, dtype=torch.bool)
    mask[pad : pad + TARGET_SIZE, pad : pad + TARGET_SIZE] = False

    # fill=0 happens before ToTensor/Normalize, so the border lands on -mean/std.
    for channel, (mean, std) in enumerate(zip(FAKE_MEAN, FAKE_STD, strict=True)):
        border = out[channel][mask]
        assert border.min().item() == pytest.approx(border.max().item())
        assert border[0].item() == pytest.approx((0.0 - mean) / std, abs=1e-6)


def test_svhn_transforms_reuse_backbone_normalize() -> None:
    """Stats come from the incoming preprocess, not hardcoded OpenAI CLIP values."""
    _, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="none",
        target_size=TARGET_SIZE,
    )
    normalize = eval_transform.transforms[-1]
    assert tuple(normalize.mean) == FAKE_MEAN
    assert tuple(normalize.std) == FAKE_STD


def test_svhn_augmentation_none_matches_eval_transform() -> None:
    train_transform, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="none",
        target_size=TARGET_SIZE,
    )
    image = _sample_image()
    assert torch.allclose(train_transform(image), eval_transform(image))


def test_svhn_autoaugment_only_changes_the_train_transform() -> None:
    train_transform, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="autoaugment",
        target_size=TARGET_SIZE,
    )
    assert len(train_transform.transforms) == len(eval_transform.transforms) + 1


def test_svhn_unknown_augmentation_raises() -> None:
    with pytest.raises(ValueError, match=r"augmentation must be one of"):
        build_svhn_transforms(
            preprocess=_fake_clip_preprocess(),
            augmentation="does_not_exist",
            target_size=TARGET_SIZE,
        )


def test_svhn_target_size_larger_than_input_raises() -> None:
    with pytest.raises(ValueError, match=r"target size must be <="):
        build_svhn_transforms(
            preprocess=_fake_clip_preprocess(),
            augmentation="none",
            target_size=INPUT_SIZE + 1,
        )


def test_svhn_transforms_are_picklable() -> None:
    """DataLoader workers under spawn need this; a transforms.Lambda would fail here."""
    train_transform, eval_transform = build_svhn_transforms(
        preprocess=_fake_clip_preprocess(),
        augmentation="autoaugment",
        target_size=TARGET_SIZE,
    )
    assert pickle.loads(pickle.dumps(train_transform)) is not None
    assert pickle.loads(pickle.dumps(eval_transform)) is not None
