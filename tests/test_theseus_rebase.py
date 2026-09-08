from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.methods import theseus as theseus_mod
from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.runtime import format_rebase_method_label


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        return self.fc2(x)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, hid_dim: int = 8, out_dim: int = 5) -> None:
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_theseus_registered() -> None:
    assert "theseus" in list_methods()
    assert get_method("theseus").name == "theseus"


def test_theseus_transport_smoke() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=1,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_partial_whitening_changes_alignment_map() -> None:
    store = theseus_mod.ActivationStore(store_a_gram=True, store_b_gram=True)
    source_rows = torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_rows = torch.tensor([[1.0, 2.0], [2.0, 0.0], [0.0, 1.0]])
    store.update(source_rows, target_rows)

    raw_map = theseus_mod._compute_alignment_map(
        store,
        center=False,
        whiten_power=0.0,
        whiten_eps=1e-6,
    )
    whitened_map = theseus_mod._compute_alignment_map(
        store,
        center=False,
        whiten_power=0.5,
        whiten_eps=1e-6,
    )

    assert raw_map is not None
    assert whitened_map is not None
    assert raw_map.shape == whitened_map.shape == (2, 2)
    assert not torch.allclose(raw_map, whitened_map)


def test_theseus_transport_with_partial_whitening_smoke() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    loader = _make_loader(in_dim=6)
    method = get_method("theseus")

    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        whiten_power=0.25,
        num_batches=1,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_theseus_data_free_transport_smoke_without_dataloaders() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)

    source_base = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
    target_base = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

    delta = {
        key: torch.randn_like(tensor)
        for key, tensor in source_base.items()
        if key.startswith("visual.") and tensor.is_floating_point()
    }

    method = get_method("theseus")
    transported = method.transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        device="cpu",
        covariance_mode="data_free",
        whiten_power=0.25,
        strict=True,
    )

    assert transported
    assert set(transported.keys()) == set(delta.keys())
    for key, tensor in transported.items():
        assert tensor.shape == target_base[key].shape
        assert tensor.dtype == target_base[key].dtype


def test_data_free_covariance_map_uses_weight_proxies() -> None:
    source = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    target = source @ rotation

    t_in = theseus_mod._compute_alignment_map_from_matrix_proxies(
        source,
        target,
        side="input",
        whiten_power=0.0,
        whiten_eps=1e-6,
    )

    assert t_in.shape == (2, 2)
    source_cov = source.T @ source
    target_cov = target.T @ target
    aligned_cov = t_in.T @ source_cov @ t_in
    assert torch.allclose(aligned_cov, target_cov, atol=1e-5, rtol=1e-5)


def test_theseus_runtime_label_includes_covariance_mode() -> None:
    label = format_rebase_method_label(
        "theseus",
        {"num_batches": 3, "seq_align": "mean", "covariance_mode": "data_free", "whiten_power": 0.25},
    )
    assert label == "theseus(batches=3, align=mean, cov=data_free, whiten=0.25)"


def test_data_free_proj_transform_uses_swapped_axes() -> None:
    source_ref = torch.randn(8, 5)
    target_ref = torch.randn(7, 6)
    visual_delta = {"proj": torch.randn_like(source_ref)}
    transforms = theseus_mod._precompute_transforms_data_free(
        source_visual_base={"proj": source_ref},
        target_visual_base={"proj": target_ref},
        visual_delta=visual_delta,
        whiten_power=0.0,
        whiten_eps=1e-6,
        show_progress=False,
        method_name="theseus",
    )

    transform = transforms["proj"]
    assert transform.t_in is not None
    assert transform.t_out is not None
    assert transform.t_in.shape == (8, 7)
    assert transform.t_out.shape == (5, 6)


def test_fused_qkv_split_merge_roundtrip() -> None:
    w = torch.randn(12, 4)
    b = torch.randn(12)
    sd = {
        "transformer.resblocks.0.attn.in_proj_weight": w,
        "transformer.resblocks.0.attn.in_proj_bias": b,
        "transformer.resblocks.0.attn.out_proj.weight": torch.randn(4, 4),
    }

    split = theseus_mod._split_fused_qkv_state(sd)
    assert "transformer.resblocks.0.attn.q_proj.weight" in split
    assert "transformer.resblocks.0.attn.k_proj.weight" in split
    assert "transformer.resblocks.0.attn.v_proj.weight" in split
    assert "transformer.resblocks.0.attn.in_proj_weight" not in split

    merged = theseus_mod._merge_split_qkv_state(split, reference=sd)
    assert "transformer.resblocks.0.attn.in_proj_weight" in merged
    assert "transformer.resblocks.0.attn.in_proj_bias" in merged
    assert torch.allclose(merged["transformer.resblocks.0.attn.in_proj_weight"], w)
    assert torch.allclose(merged["transformer.resblocks.0.attn.in_proj_bias"], b)


def test_random_dataset_subsampling_uses_randperm_seed() -> None:
    x = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    iterator = theseus_mod._iter_random_dataset_batches(
        loader,
        loader,
        n_batches=3,
        seed=123,
        batch_size=4,
    )
    assert iterator is not None

    seen: list[int] = []
    for source_batch, _ in iterator:
        inputs = source_batch[0]
        seen.extend(int(v) for v in inputs.squeeze(1).tolist())

    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    expected = torch.randperm(20, generator=g)[:12].tolist()
    assert seen == expected

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(124)
    expected_other_seed = torch.randperm(20, generator=g2)[:12].tolist()
    assert seen != expected_other_seed


def _make_class_balanced_loader(
    *, n_classes: int = 5, per_class: int = 6, in_dim: int = 6, batch_size: int = 4
) -> DataLoader:
    x = torch.randn(n_classes * per_class, in_dim)
    y = torch.arange(n_classes).repeat_interleave(per_class)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_dataset_labels_reads_tensor_dataset() -> None:
    loader = _make_class_balanced_loader(n_classes=3, per_class=4)
    labels = theseus_mod._dataset_labels(loader.dataset)
    assert labels is not None
    assert labels.tolist() == loader.dataset.tensors[1].tolist()


def test_class_balanced_indices_exact_shots_per_class() -> None:
    loader = _make_class_balanced_loader(n_classes=5, per_class=6)
    indices = theseus_mod._class_balanced_indices(loader.dataset, shots_per_class=2, seed=0)
    assert indices.numel() == 5 * 2

    labels = loader.dataset.tensors[1]
    selected_labels = labels[indices]
    for class_id in range(5):
        assert int((selected_labels == class_id).sum()) == 2


def test_class_balanced_indices_raises_when_a_class_is_too_small() -> None:
    loader = _make_class_balanced_loader(n_classes=2, per_class=3)
    with pytest.raises(ValueError, match="shots_per_class"):
        theseus_mod._class_balanced_indices(loader.dataset, shots_per_class=4, seed=0)


def test_iter_random_dataset_batches_shots_per_class_is_class_balanced() -> None:
    loader = _make_class_balanced_loader(n_classes=4, per_class=5, batch_size=3)

    iterator = theseus_mod._iter_random_dataset_batches(
        loader,
        loader,
        n_batches=None,
        seed=7,
        batch_size=3,
        shots_per_class=2,
    )
    assert iterator is not None

    seen_labels: list[int] = []
    for source_batch, target_batch in iterator:
        seen_labels.extend(int(v) for v in source_batch[1].tolist())
        assert source_batch[1].tolist() == target_batch[1].tolist()

    assert len(seen_labels) == 4 * 2
    for class_id in range(4):
        assert seen_labels.count(class_id) == 2


def test_theseus_prepare_shots_per_class_smoke() -> None:
    source_model = _TinyModel(in_dim=6, hid_dim=8, out_dim=5)
    target_model = _TinyModel(in_dim=6, hid_dim=7, out_dim=5)
    loader = _make_class_balanced_loader(n_classes=5, per_class=4, in_dim=6, batch_size=4)
    method = get_method("theseus")

    prepared = method.prepare(
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        shots_per_class=2,
        verbose=False,
        show_progress=False,
    )

    assert prepared["shots_per_class"] == 2
    assert prepared["activation_registry"]
