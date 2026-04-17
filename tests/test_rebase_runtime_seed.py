from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.runtime import transport_vision_task_vector


class _DummyMethod:
    name = "theseus"

    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def transport(self, **kwargs):
        self.last_kwargs = kwargs
        return {}


def _build_loaders(batch_size: int = 4):
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 2, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return SimpleNamespace(train=loader)


def test_theseus_runtime_propagates_run_seed_when_missing_from_method_params():
    method = _DummyMethod()
    loaders = _build_loaders()
    source_loaders = _build_loaders()

    clf_source = SimpleNamespace(model=torch.nn.Identity())
    clf_target = SimpleNamespace(model=torch.nn.Identity())

    out = transport_vision_task_vector(
        method=method,
        source_base={},
        target_base={},
        delta={},
        clf_source=clf_source,
        clf_target=clf_target,
        task_name="dummy",
        source_loaders=source_loaders,
        loaders=loaders,
        classnames=["a", "b"],
        build_cfg_task=SimpleNamespace(),
        device="cpu",
        strict=False,
        method_params={"num_batches": 1, "batch_size": 4},
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        num_workers=0,
        seed=89,
    )

    assert out == {}
    assert method.last_kwargs is not None
    assert method.last_kwargs["seed"] == 89


def test_theseus_runtime_respects_explicit_method_seed():
    method = _DummyMethod()
    loaders = _build_loaders()
    source_loaders = _build_loaders()

    clf_source = SimpleNamespace(model=torch.nn.Identity())
    clf_target = SimpleNamespace(model=torch.nn.Identity())

    transport_vision_task_vector(
        method=method,
        source_base={},
        target_base={},
        delta={},
        clf_source=clf_source,
        clf_target=clf_target,
        task_name="dummy",
        source_loaders=source_loaders,
        loaders=loaders,
        classnames=["a", "b"],
        build_cfg_task=SimpleNamespace(),
        device="cpu",
        strict=False,
        method_params={"seed": 7, "num_batches": 1, "batch_size": 4},
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        num_workers=0,
        seed=89,
    )

    assert method.last_kwargs is not None
    assert method.last_kwargs["seed"] == 7
