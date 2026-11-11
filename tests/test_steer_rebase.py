from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.rebase.methods import steer as steer_mod
from merge_and_rebase.rebase.methods.steer import steer_correction_context
from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.runtime import format_rebase_method_label


def test_steer_registered() -> None:
    assert "steer" in list_methods()
    assert get_method("steer").name == "steer"
    label = format_rebase_method_label("steer", {"feature_regime": "linear", "stage_2_strategy": "block_ridge"})
    assert "steer" in label and "block_ridge" in label


def _synthetic_dataset(seed: int = 0, n: int = 64, d_a: int = 5, d_b: int = 7, classes: int = 4):
    g = torch.Generator().manual_seed(seed)
    w_a = torch.randn(classes, d_a, generator=g, dtype=torch.float64)
    w_b = torch.randn(classes, d_b, generator=g, dtype=torch.float64)
    f_a = torch.randn(n, d_a, generator=g, dtype=torch.float64)
    delta_a = torch.randn(n, d_a, generator=g, dtype=torch.float64)
    f_b = torch.randn(n, d_b, generator=g, dtype=torch.float64)
    labels = torch.randint(0, classes, (n,), generator=g)
    return w_a, w_b, f_a, delta_a, f_b, labels


def test_stage1_target_corrections_shapes() -> None:
    w_a, w_b, f_a, delta_a, f_b, labels = _synthetic_dataset()
    selected = torch.arange(16)
    train_target, test_target = steer_mod._stage1_target_corrections(
        f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, delta_a_test=delta_a, selected=selected, regularization=1.0
    )
    assert train_target.shape == (16, f_b.shape[1])
    assert test_target.shape == (f_a.shape[0], f_b.shape[1])
    assert torch.isfinite(train_target).all()
    assert torch.isfinite(test_target).all()


def _matrix_with_singular_values(values: torch.Tensor, rows: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(rows, values.numel(), generator=g, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(values.numel(), values.numel(), generator=g, dtype=torch.float64))
    return left @ torch.diag(values) @ right.T


def test_energy_truncated_pinv_keeps_leading_singular_values() -> None:
    # 1000 singular values summing to 10000; the first 600 carry 9995 of it (99.95%).
    values = torch.cat([torch.full((600,), 9995.0 / 600), torch.full((400,), 5.0 / 400)]).double()
    matrix = torch.diag(values)
    _, kept, rank = steer_mod._energy_truncated_pinv(matrix, 0.999)
    assert (kept, rank) == (600, 1000)
    # Default 0.99: smallest k with k * 9995/600 >= 9900, i.e. ceil(594.30) = 595.
    _, kept, _ = steer_mod._energy_truncated_pinv(matrix, 0.99)
    assert kept == 595


def test_energy_truncated_pinv_full_energy_matches_pinv() -> None:
    values = torch.linspace(5.0, 0.1, 12, dtype=torch.float64)
    matrix = _matrix_with_singular_values(values, rows=20)
    pinv, kept, rank = steer_mod._energy_truncated_pinv(matrix, 1.0)
    assert (kept, rank) == (12, 12)
    assert torch.allclose(pinv, torch.linalg.pinv(matrix), atol=1e-10)
    truncated, kept, _ = steer_mod._energy_truncated_pinv(matrix, 0.5)
    assert kept < 12
    # The truncated inverse is the exact inverse restricted to the kept singular directions.
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    expected = vh[:kept].T @ torch.diag(1.0 / s[:kept]) @ u[:, :kept].T
    assert torch.allclose(truncated, expected, atol=1e-10)


def test_energy_truncated_pinv_ignores_numerically_zero_singular_values() -> None:
    values = torch.cat([torch.tensor([3.0, 2.0, 1.0]), torch.zeros(5)]).double()
    matrix = _matrix_with_singular_values(values, rows=10)
    pinv, kept, rank = steer_mod._energy_truncated_pinv(matrix, 1.0)
    assert (kept, rank) == (3, 3)
    assert torch.allclose(pinv, torch.linalg.pinv(matrix), atol=1e-10)


def test_energy_truncated_pinv_rejects_bad_thresholds() -> None:
    import pytest

    matrix = torch.eye(4, dtype=torch.float64)
    for threshold in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError):
            steer_mod._energy_truncated_pinv(matrix, threshold)
    with pytest.raises(ValueError):
        steer_mod._energy_truncated_pinv(torch.zeros(4, 4, dtype=torch.float64), 0.99)


def test_stage1_projection_energy_threshold_default_and_exact() -> None:
    w_a, w_b, f_a, delta_a, f_b, _ = _synthetic_dataset()
    selected = torch.arange(16)
    kwargs = dict(f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=selected, regularization=1.0)
    default_map, default_kept, rank = steer_mod._stage1_projection(**kwargs)
    explicit_map, explicit_kept, _ = steer_mod._stage1_projection(**kwargs, energy_threshold=0.99)
    assert default_kept == explicit_kept and torch.equal(default_map, explicit_map)
    exact_map, exact_kept, _ = steer_mod._stage1_projection(**kwargs, energy_threshold=1.0)
    residual = (f_a[selected] + delta_a[selected]) @ w_a.T - f_b[selected] @ w_b.T
    expected = (torch.linalg.pinv(delta_a[selected]) @ residual / 2.0).T
    assert exact_kept == rank
    assert torch.allclose(exact_map, expected, atol=1e-10)


def test_global_ridge_recovers_linear_map() -> None:
    g = torch.Generator().manual_seed(1)
    n, d, out_dim = 40, 6, 3
    features = torch.randn(n, d, generator=g, dtype=torch.float64)
    true_map = torch.randn(d, out_dim, generator=g, dtype=torch.float64)
    targets = features @ true_map
    coefficient = steer_mod._ridge(features, targets, regularization=1e-6)
    pred = features @ coefficient
    assert torch.allclose(pred, targets, atol=1e-3)


def test_global_mlp_fits_and_predicts() -> None:
    g = torch.Generator().manual_seed(2)
    train_features = torch.randn(20, 5, generator=g, dtype=torch.float64)
    train_target = torch.randn(20, 3, generator=g, dtype=torch.float64)
    model = steer_mod._fit_global_mlp(train_features, train_target, seed=0, epochs=5, hidden_dim=8)
    with torch.no_grad():
        out = model(train_features)
    assert out.shape == train_target.shape
    assert torch.isfinite(out).all()


def test_block_ridge_fit_predict_roundtrip() -> None:
    g = torch.Generator().manual_seed(3)
    n_sel, num_blocks, d_block, d_out = 10, 4, 6, 3
    blocks_train = {b: torch.randn(n_sel, d_block, generator=g, dtype=torch.float64) for b in range(num_blocks)}
    train_targets = torch.randn(n_sel, num_blocks, d_out, generator=g, dtype=torch.float64)
    selected = torch.arange(n_sel)

    for mode in ("independent", "smoothed_residual"):
        coefficients = steer_mod._fit_block_ridge(
            blocks_train, train_targets, selected=selected, regularization=1.0, mode=mode, rho=0.9
        )
        assert len(coefficients) == num_blocks
        pred = steer_mod._predict_block_ridge(coefficients, blocks_train)
        assert pred.shape == (n_sel, d_out)
        assert torch.isfinite(pred).all()


def test_concat_ridge_matches_scaled_concat() -> None:
    """Folded per-block coefficients reproduce a ridge on the hand-scaled concatenation."""
    g = torch.Generator().manual_seed(4)
    n, d_out = 12, 3
    widths = [6, 6, 4]
    raw_scales = [50.0, 0.1, 1.0]  # deep raw blocks vs L2-normalized output
    blocks = {
        b: raw_scales[b] * torch.randn(n, w, generator=g, dtype=torch.float64) for b, w in enumerate(widths)
    }
    target = torch.randn(n, d_out, generator=g, dtype=torch.float64)

    for scaling in ("none", "trace"):
        coefficients, info = steer_mod._fit_concat_ridge(
            blocks, target, regularization=0.5, regularization_scaling=scaling
        )
        assert [c.shape for c in coefficients] == [(w, d_out) for w in widths]
        scales = [float(v.square().sum().div(n).sqrt()) for v in blocks.values()]
        x_scaled = torch.cat([v / s for v, s in zip(blocks.values(), scales)], dim=1)
        expected_lambda = 0.5 * (len(widths) if scaling == "trace" else 1.0)
        assert abs(info["effective_lambda"] - expected_lambda) < 1e-10
        expected = x_scaled @ steer_mod._ridge(x_scaled, target, expected_lambda)
        pred = steer_mod._predict_block_ridge(coefficients, blocks)
        assert torch.allclose(pred, expected, atol=1e-10)

        # Per-block scaling makes the fit invariant to rescaling any one block.
        rescaled = dict(blocks)
        rescaled[1] = 1e3 * rescaled[1]
        rescaled_coefficients, _ = steer_mod._fit_concat_ridge(
            rescaled, target, regularization=0.5, regularization_scaling=scaling
        )
        assert torch.allclose(steer_mod._predict_block_ridge(rescaled_coefficients, rescaled), pred, atol=1e-8)


def test_concat_ridge_recovers_linear_map() -> None:
    g = torch.Generator().manual_seed(5)
    n, d_out = 60, 3
    blocks = {b: torch.randn(n, 4, generator=g, dtype=torch.float64) * (10.0**b) for b in range(3)}
    true_maps = [torch.randn(4, d_out, generator=g, dtype=torch.float64) / (10.0**b) for b in range(3)]
    target = sum(blocks[b] @ true_maps[b] for b in range(3))
    coefficients, _ = steer_mod._fit_concat_ridge(blocks, target, regularization=1e-8)
    for coefficient, true_map in zip(coefficients, true_maps):
        assert torch.allclose(coefficient, true_map, atol=1e-5)


def test_block_ridge_fit_predict_roundtrip_with_heterogeneous_block_widths() -> None:
    """Residual blocks 3x wider than the output block (steer_text's segment pooling
    triples only the residual blocks' width, never the output block's) -- the
    structural assumption the whole change rests on: _fit_block_ridge/
    _predict_block_ridge index per block and never assume a shared width."""
    g = torch.Generator().manual_seed(4)
    n_sel, num_residual, d_residual, d_output, d_out = 10, 3, 12, 4, 3
    blocks_train = {b: torch.randn(n_sel, d_residual, generator=g, dtype=torch.float64) for b in range(num_residual)}
    blocks_train[num_residual] = torch.randn(n_sel, d_output, generator=g, dtype=torch.float64)
    train_targets = torch.randn(n_sel, num_residual + 1, d_out, generator=g, dtype=torch.float64)
    selected = torch.arange(n_sel)

    coefficients = steer_mod._fit_block_ridge(
        blocks_train, train_targets, selected=selected, regularization=1.0, mode="independent"
    )
    assert len(coefficients) == num_residual + 1
    for b in range(num_residual):
        assert coefficients[b].shape == (d_residual, d_out)
    assert coefficients[num_residual].shape == (d_output, d_out)

    pred = steer_mod._predict_block_ridge(coefficients, blocks_train)
    assert pred.shape == (n_sel, d_out)
    assert torch.isfinite(pred).all()


def test_group_blocks_concat_and_sum_avg() -> None:
    blocks = {b: torch.full((2, 3), float(b)) for b in range(4)}
    concat = steer_mod._group_blocks_concat(blocks, 2)
    assert set(concat) == {0, 1}
    assert concat[0].shape == (2, 6)

    avg = steer_mod._group_blocks_sum_avg(blocks, 2)
    assert set(avg) == {0, 1}
    assert avg[0].shape == (2, 3)
    assert torch.allclose(avg[0], torch.full((2, 3), 0.5))


def test_group_blocks_last_keeps_the_deepest_member_of_each_group() -> None:
    import pytest

    blocks = {b: torch.full((2, 3), float(b)) for b in range(4)}
    last = steer_mod._group_blocks_last(blocks, 2)
    assert set(last) == {0, 1}
    assert last[0].shape == (2, 3) and last[1].shape == (2, 3)  # not 2x wide, unlike concat
    assert torch.equal(last[0], blocks[1]) and torch.equal(last[1], blocks[3])

    # Same number of blocks and groups: nothing to group, returned as-is.
    assert set(steer_mod._group_blocks_last(blocks, 4)) == {0, 1, 2, 3}
    assert "last" in steer_mod._BLOCK_GROUP_STRATEGIES

    with pytest.raises(ValueError):
        steer_mod._group_blocks_last(blocks, 0)
    with pytest.raises(ValueError):
        steer_mod._group_blocks_last(blocks, 5)
    with pytest.raises(ValueError):
        steer_mod._group_blocks_last({}, 2)


def test_clip_vit_parameter_blocks() -> None:
    names = [
        "class_embedding",
        "positional_embedding",
        "conv1.weight",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.1.attn.in_proj_weight",
        "proj",
        "ln_post.weight",
    ]
    block_ids, num_blocks = steer_mod._clip_vit_parameter_blocks(names)
    assert num_blocks == 3  # 2 resblocks (0,1) + 1 output block
    assert block_ids[names.index("class_embedding")] == 0
    assert block_ids[names.index("transformer.resblocks.1.attn.in_proj_weight")] == 1
    assert block_ids[names.index("proj")] == 2


def test_clip_resnet_parameter_blocks() -> None:
    names = ["conv1.weight", "bn1.weight"]
    for stage, count in steer_mod._RESNET_STAGE_BLOCKS.items():
        for index in range(count):
            names.append(f"layer{stage}.{index}.conv1.weight")
    names += ["attnpool.in_proj_weight", "proj"]

    block_ids, num_blocks = steer_mod._clip_resnet_parameter_blocks(names)
    total_bottlenecks = sum(steer_mod._RESNET_STAGE_BLOCKS.values())
    assert num_blocks == total_bottlenecks + 1
    assert block_ids[names.index("conv1.weight")] == 0
    assert block_ids[names.index("layer4.2.conv1.weight")] == total_bottlenecks - 1
    assert block_ids[names.index("proj")] == total_bottlenecks
    assert block_ids[names.index("attnpool.in_proj_weight")] == total_bottlenecks

    # Round-trip block_id -> (stage, index) matches the forward mapping.
    for stage, count in steer_mod._RESNET_STAGE_BLOCKS.items():
        for index in range(count):
            bid = steer_mod._resnet_block_id_for(stage, index)
            assert steer_mod._resnet_block_id_to_stage_index(bid) == (stage, index)


def test_feature_cache_roundtrip(tmp_path) -> None:
    calls = {"n": 0}

    def compute_fn():
        calls["n"] += 1
        return {
            "features_A": torch.randn(4, 3),
            "delta_A": torch.randn(4, 3),
            "features_B": torch.randn(4, 5),
            "y_A": torch.randint(0, 2, (4,)),
        }

    kwargs = dict(
        feature_cache_dir=str(tmp_path),
        source_tag="srcA",
        target_tag="tgtB",
        task="TaskX",
        feature_regime="standard",
        split="train",
        need_source_blocks=False,
        need_target_blocks=False,
        compute_fn=compute_fn,
        verbose=False,
    )

    first = steer_mod._load_or_compute_split(force_recompute_features=False, **kwargs)
    assert calls["n"] == 1

    second = steer_mod._load_or_compute_split(force_recompute_features=False, **kwargs)
    assert calls["n"] == 1  # cache hit, no recompute
    assert torch.equal(first["features_A"], second["features_A"])
    assert torch.equal(first["delta_A"], second["delta_A"])

    steer_mod._load_or_compute_split(force_recompute_features=True, **kwargs)
    assert calls["n"] == 2  # forced recompute


class _TinyResblock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class _TinyTransformer(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_TinyResblock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class _TinyVisual(nn.Module):
    """Minimal CLIP-ViT-shaped module: `transformer.resblocks.N.*`, `proj`, etc.

    Deliberately gives the residual-block hidden width (``dim``) a different
    size from the projected output width (``out_dim``) -- this is the real
    CLIP situation (e.g. ViT-B-16's 768 hidden dim vs 512 joint embed dim) and
    is what exposes the dimension-mismatch bug in block grouping if the final
    projected feature is ever concatenated/averaged together with residual
    block activations instead of standing alone as its own group.
    """

    def __init__(self, dim: int = 8, depth: int = 3, out_dim: int = 5) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(dim))
        self.positional_embedding = nn.Parameter(torch.zeros(5, dim))
        self.conv1 = nn.Conv2d(3, dim, kernel_size=8, stride=8, bias=False)
        self.ln_pre = nn.LayerNorm(dim)
        self.transformer = _TinyTransformer(dim, depth)
        self.ln_post = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, out_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        b, c, h, w = feat.shape
        feat = feat.reshape(b, c, h * w).permute(0, 2, 1)
        cls = self.class_embedding.expand(b, 1, -1)
        feat = torch.cat([cls, feat], dim=1)
        feat = feat + self.positional_embedding[: feat.shape[1]]
        feat = self.ln_pre(feat)
        feat = self.transformer(feat)
        feat = self.ln_post(feat[:, 0])
        return feat @ self.proj


class _TinyCLIP(nn.Module):
    def __init__(self, dim: int = 8, depth: int = 3, out_dim: int = 5) -> None:
        super().__init__()
        self.visual = _TinyVisual(dim=dim, depth=depth, out_dim=out_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.randn(tokens.shape[0], 5)


def _tiny_tokenizer(texts):
    return torch.zeros(len(texts), 4, dtype=torch.long)


def _make_tiny_clf(depth: int, dim: int = 8, out_dim: int = 5) -> OpenClipClassifier:
    return OpenClipClassifier(
        model=_TinyCLIP(dim=dim, depth=depth, out_dim=out_dim),
        tokenizer=_tiny_tokenizer,
        preprocess=None,
        normalize=True,
        logit_scale=100.0,
    )


class _Loaders:
    def __init__(self, loader: DataLoader) -> None:
        self.train = loader
        self.test = loader


_BUILD_CFG = OpenClipBuildConfig(model_name="tiny", pretrained="none", prompt_templates=[lambda c: f"a photo of a {c}"])
_CLASSNAMES = ["cat", "dog", "car", "tree"]


def _tiny_loaders(n: int = 16, dim_hw: int = 16) -> _Loaders:
    x = torch.randn(n, 3, dim_hw, dim_hw)
    y = torch.arange(n) % len(_CLASSNAMES)
    return _Loaders(DataLoader(TensorDataset(x, y), batch_size=4))


def _finetune(clf: OpenClipClassifier) -> OpenClipClassifier:
    with torch.no_grad():
        for p in clf.model.visual.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return clf


def test_steer_prepare_and_correction_end_to_end(tmp_path) -> None:
    """Full prepare()+correction round trip for each stage_2_strategy on a tiny fake CLIP."""
    torch.manual_seed(0)
    loaders = _tiny_loaders()

    for strategy, regime in [
        ("global_ridge", "standard"),
        ("global_mlp", "standard"),
        ("block_ridge", "linear"),
        ("concat_ridge", "standard"),
        ("concat_ridge", "linear"),
    ]:
        clf_source_pretrained = _make_tiny_clf(depth=3)
        clf_source_finetuned = _make_tiny_clf(depth=3)
        clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
        _finetune(clf_source_finetuned)
        clf_target = _make_tiny_clf(depth=3)

        method = get_method("steer")
        prepared = method.prepare(
            clf_source=clf_source_finetuned,
            clf_source_pretrained=clf_source_pretrained,
            clf_target=clf_target,
            source_loaders=loaders,
            target_loaders=loaders,
            classnames=_CLASSNAMES,
            task=f"tiny_{strategy}_{regime}",
            source_build_cfg_task=_BUILD_CFG,
            build_cfg_task=_BUILD_CFG,
            device="cpu",
            feature_regime=regime,
            stage_2_strategy=strategy,
            few_shot=2,
            feature_cache_dir=str(tmp_path / "features"),
            force_recompute_features=True,
            mlp_epochs=3,
            verbose=False,
        )
        x = torch.randn(4, 3, 16, 16)
        with steer_correction_context(clf_target, prepared, alpha=1.0):
            logits = clf_target(x)
        assert logits.shape == (4, len(_CLASSNAMES))
        assert torch.isfinite(logits).all()


def test_steer_block_ridge_cross_arch_dimension_safety(tmp_path) -> None:
    """
    Regression test: the target's final projected feature (width=out_dim) must
    never be grouped together with residual-block activations (width=dim) when
    dim != out_dim and depths differ, for either block_group_strategy.
    """
    torch.manual_seed(1)
    loaders = _tiny_loaders()

    clf_source_pretrained = _make_tiny_clf(depth=3, dim=8, out_dim=5)
    clf_source_finetuned = _make_tiny_clf(depth=3, dim=8, out_dim=5)
    clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
    _finetune(clf_source_finetuned)
    clf_target = _make_tiny_clf(depth=6, dim=8, out_dim=5)  # deeper target: 6 residual blocks vs source's 3

    for strategy in ("concat", "sum_avg"):
        method = get_method("steer")
        prepared = method.prepare(
            clf_source=clf_source_finetuned,
            clf_source_pretrained=clf_source_pretrained,
            clf_target=clf_target,
            source_loaders=loaders,
            target_loaders=loaders,
            classnames=_CLASSNAMES,
            task=f"crossarch_{strategy}",
            source_build_cfg_task=_BUILD_CFG,
            build_cfg_task=_BUILD_CFG,
            device="cpu",
            feature_regime="linear",
            stage_2_strategy="block_ridge",
            block_group_strategy=strategy,
            few_shot=2,
            feature_cache_dir=str(tmp_path / f"features_{strategy}"),
            force_recompute_features=True,
            verbose=False,
        )
        assert prepared["num_source_blocks"] == 4  # 3 residual + 1 output
        assert prepared["block_group_size"] == 2.0  # 6 residual target blocks / 3 residual source blocks

        x = torch.randn(4, 3, 16, 16)
        with steer_correction_context(clf_target, prepared, alpha=1.0):
            logits = clf_target(x)
        assert logits.shape == (4, len(_CLASSNAMES))
        assert torch.isfinite(logits).all()


def test_block_ridge_trace_scaling_matches_hand_scaled_lambda() -> None:
    """``regularization_scaling="trace"`` fits block b with beta * tr(X_b X_b^T) / n, per block,
    and leaves the default ("none") exactly as before -- including through the smoothed carry."""
    import pytest

    g = torch.Generator().manual_seed(5)
    n_sel, num_blocks, d_block, d_out, beta, rho = 10, 3, 6, 3, 0.1, 0.9
    # Block scales span two orders of magnitude, like an un-normalized residual stream.
    blocks = {b: (10.0**b) * torch.randn(n_sel, d_block, generator=g, dtype=torch.float64) for b in range(num_blocks)}
    targets = torch.randn(n_sel, num_blocks, d_out, generator=g, dtype=torch.float64)
    selected = torch.arange(n_sel)
    ridge = steer_mod._ridge

    trace = steer_mod._fit_block_ridge(
        blocks, targets, selected=selected, regularization=beta, mode="independent", regularization_scaling="trace"
    )
    plain = steer_mod._fit_block_ridge(blocks, targets, selected=selected, regularization=beta, mode="independent")
    for b in range(num_blocks):
        lam = beta * float(blocks[b].square().sum()) / n_sel
        assert torch.allclose(trace[b], ridge(blocks[b], targets[:, b], lam))
        assert torch.allclose(plain[b], ridge(blocks[b], targets[:, b], beta))
    assert not torch.allclose(trace[2], plain[2])  # the deepest (largest-scale) block is regularized far more

    # smoothed_residual: the carry uses each block's own (trace) fit.
    carried = steer_mod._fit_block_ridge(
        blocks, targets, selected=selected, regularization=beta, mode="smoothed_residual", rho=rho,
        regularization_scaling="trace",
    )
    state = torch.zeros(n_sel, d_out, dtype=torch.float64)
    for b in range(num_blocks):
        compensation = rho * state
        lam = beta * float(blocks[b].square().sum()) / n_sel
        want = ridge(blocks[b], targets[:, b] + compensation, lam)
        assert torch.allclose(carried[b], want)
        state = compensation + targets[:, b] - blocks[b] @ want

    with pytest.raises(ValueError, match="regularization_scaling"):
        steer_mod._fit_block_ridge(
            blocks, targets, selected=selected, regularization=beta, mode="independent", regularization_scaling="mean"
        )


def test_steer_block_ridge_trace_scaling_end_to_end(tmp_path) -> None:
    """The recipe used for the vision runs -- residual (end-of-block) features, trace lambda,
    smoothed_residual with rho=0.9, ridge_lambda=0.1 -- runs through prepare() and the live
    correction, records the option, and changes the fit relative to the unscaled one."""
    import pytest

    torch.manual_seed(2)
    loaders = _tiny_loaders()
    clf_source_pretrained = _make_tiny_clf(depth=3)
    clf_source_finetuned = _make_tiny_clf(depth=3)
    clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
    _finetune(clf_source_finetuned)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(scaling: str):
        return get_method("steer").prepare(
            clf_source=clf_source_finetuned,
            clf_source_pretrained=clf_source_pretrained,
            clf_target=clf_target,
            source_loaders=loaders,
            target_loaders=loaders,
            classnames=_CLASSNAMES,
            task="tiny_trace",
            source_build_cfg_task=_BUILD_CFG,
            build_cfg_task=_BUILD_CFG,
            device="cpu",
            feature_regime="linear",
            stage_2_strategy="block_ridge",
            block_granularity="residual",
            block_ridge_mode="smoothed_residual",
            rho=0.9,
            ridge_lambda=0.1,
            block_ridge_lambda_scaling=scaling,
            few_shot=2,
            feature_cache_dir=str(tmp_path / "features"),
            force_recompute_features=(scaling == "none"),  # the second call reuses the first one's cache
            verbose=False,
        )

    plain, trace = _prepare("none"), _prepare("trace")
    assert plain["block_ridge_lambda_scaling"] == "none" and trace["block_ridge_lambda_scaling"] == "trace"

    x = torch.randn(4, 3, 16, 16)
    outputs = []
    for prepared in (plain, trace):
        with steer_correction_context(clf_target, prepared, alpha=1.0):
            logits = clf_target(x)
        assert logits.shape == (4, len(_CLASSNAMES)) and torch.isfinite(logits).all()
        outputs.append(logits)
    assert not torch.allclose(outputs[0], outputs[1]), "trace scaling must change the fitted correction"

    with pytest.raises(ValueError, match="block_ridge_lambda_scaling"):
        _prepare("median")


def test_pool_block_output_token_pooling() -> None:
    tokens = torch.randn(2, 5, 4)
    assert torch.equal(steer_mod._pool_block_output(tokens), tokens.mean(dim=1))
    assert torch.equal(steer_mod._pool_block_output(tokens, "mean"), tokens.mean(dim=1))
    assert torch.equal(steer_mod._pool_block_output((tokens, None), "cls"), tokens[:, 0])
    # Conv maps (the ViT patch embedding) have no class token: always GAP.
    conv = torch.randn(2, 3, 4, 4)
    assert torch.equal(steer_mod._pool_block_output(conv), conv.mean(dim=(2, 3)))
    assert torch.equal(steer_mod._pool_block_output(conv, "cls"), conv.mean(dim=(2, 3)))


def test_cache_split_dir_token_pooling() -> None:
    args = ("cache", "a", "b", "SVHN", "linear", "train")
    assert steer_mod._cache_split_dir(*args) == steer_mod._cache_split_dir(*args, "residual", "mean")
    assert str(steer_mod._cache_split_dir(*args)).endswith("SVHN/linear/train")
    assert str(steer_mod._cache_split_dir(*args, "residual", "cls")).endswith("SVHN/linear_clspool/train")
    assert str(steer_mod._cache_split_dir(*args, "attention", "cls")).endswith(
        "SVHN/linear_attentiongranularity_clspool/train"
    )


def _reference_block_tokens(visual: nn.Module, x: torch.Tensor) -> dict[int, torch.Tensor]:
    """Unpooled [B, L, D] output of every resblock, captured independently of steer."""
    tokens: dict[int, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(lambda _m, _i, out, b=b: tokens.__setitem__(b, out.detach()))
        for b, block in enumerate(visual.transformer.resblocks)
    ]
    try:
        with torch.no_grad():
            visual(x)
    finally:
        for handle in handles:
            handle.remove()
    return tokens


def test_residual_capture_cls_pooling_matches_class_token() -> None:
    torch.manual_seed(3)
    visual = _TinyVisual(depth=3).eval()
    x = torch.randn(4, 3, 16, 16)
    reference = _reference_block_tokens(visual, x)

    pooled = {}
    for pooling in ("mean", "cls"):
        capture, final_ids, expected = steer_mod._activation_capture_for_visual(visual, "residual", pooling)
        with capture as c, torch.no_grad():
            out = visual(x)
        pooled[pooling] = steer_mod._complete_captured_activations(
            c, final_output=out, final_ids=final_ids, expected_blocks=expected
        )
    for b, tok in reference.items():
        assert torch.allclose(pooled["cls"][b], tok[:, 0])
        assert torch.allclose(pooled["mean"][b], tok.mean(dim=1))
        assert not torch.allclose(pooled["cls"][b], pooled["mean"][b])
    # The final projected feature does not depend on the token pooling.
    assert torch.equal(pooled["cls"][3], pooled["mean"][3])


def test_steer_block_ridge_cls_pooling_end_to_end(tmp_path) -> None:
    """token_pooling='cls' is used both for the cached fit features and for the live eval capture."""
    import pytest

    torch.manual_seed(4)
    loaders = _tiny_loaders()
    clf_source_pretrained = _make_tiny_clf(depth=3)
    clf_source_finetuned = _make_tiny_clf(depth=3)
    clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
    _finetune(clf_source_finetuned)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(pooling: str):
        return get_method("steer").prepare(
            clf_source=clf_source_finetuned,
            clf_source_pretrained=clf_source_pretrained,
            clf_target=clf_target,
            source_loaders=loaders,
            target_loaders=loaders,
            classnames=_CLASSNAMES,
            task="tiny_cls",
            source_build_cfg_task=_BUILD_CFG,
            build_cfg_task=_BUILD_CFG,
            device="cpu",
            feature_regime="linear",
            stage_2_strategy="block_ridge",
            block_granularity="residual",
            token_pooling=pooling,
            few_shot=2,
            feature_cache_dir=str(tmp_path / "features"),
            force_recompute_features=True,
            verbose=False,
        )

    mean_prepared, cls_prepared = _prepare("mean"), _prepare("cls")
    assert mean_prepared["token_pooling"] == "mean" and cls_prepared["token_pooling"] == "cls"

    # Fit side: cached target block features are the class token of each resblock.
    visual = clf_target.model.visual
    x_all = loaders.test.dataset.tensors[0]
    reference = _reference_block_tokens(visual, x_all)
    for pooling, reduce in (("cls", lambda t: t[:, 0]), ("mean", lambda t: t.mean(dim=1))):
        cache_dir = steer_mod._cache_split_dir(
            str(tmp_path / "features"), "source", "target", "tiny_cls", "linear", "test", "residual", pooling
        )
        cached_blocks = torch.load(cache_dir / "features_B_blocks.pt", weights_only=True)
        for b, tok in reference.items():
            assert torch.allclose(cached_blocks[b], reduce(tok), atol=1e-6)

    # Eval side: the live wrapper must feed the predictor CLS-pooled activations.
    x = torch.randn(4, 3, 16, 16)
    capture, final_ids, expected = steer_mod._activation_capture_for_visual(visual, "residual", "cls")
    with capture as c, torch.no_grad():
        out_norm = steer_mod._l2_normalize(clf_target.model.encode_image(x))
    blocks = steer_mod._complete_captured_activations(c, final_output=out_norm, final_ids=final_ids, expected_blocks=expected)
    expected_feature = out_norm + cls_prepared["correction_fn"]({"global": out_norm, "blocks": blocks})
    with steer_correction_context(clf_target, cls_prepared, alpha=1.0), torch.no_grad():
        live_feature = clf_target.model.encode_image(x)
    assert torch.allclose(live_feature, expected_feature, atol=1e-6)

    with pytest.raises(ValueError, match="token_pooling"):
        _prepare("max")


def _prepare_tiny(models, loaders, cache_dir, *, strategy="block_ridge", pooling="mean", force=False):
    source_finetuned, source_pretrained, target = models
    return get_method("steer").prepare(
        clf_source=source_finetuned,
        clf_source_pretrained=source_pretrained,
        clf_target=target,
        source_loaders=loaders,
        target_loaders=loaders,
        classnames=_CLASSNAMES,
        task="tiny_reuse",
        source_build_cfg_task=_BUILD_CFG,
        build_cfg_task=_BUILD_CFG,
        device="cpu",
        feature_regime="linear",
        stage_2_strategy=strategy,
        token_pooling=pooling,
        few_shot=2,
        feature_cache_dir=str(cache_dir),
        force_recompute_features=force,
        verbose=False,
    )


def _tiny_models(seed: int):
    torch.manual_seed(seed)
    source_pretrained = _make_tiny_clf(depth=3)
    source_finetuned = _make_tiny_clf(depth=3)
    source_finetuned.load_state_dict(source_pretrained.state_dict())
    _finetune(source_finetuned)
    return source_finetuned, source_pretrained, _make_tiny_clf(depth=3)


def test_cls_pooling_reuses_mean_cache_and_matches_full_extraction(tmp_path, monkeypatch) -> None:
    """With a mean-pooled cache present, cls only recomputes B's blocks, and the result
    (cached tensors and fitted correction) equals a from-scratch cls extraction."""
    models = _tiny_models(5)
    loaders = _tiny_loaders()
    reuse_dir, full_dir = tmp_path / "reuse", tmp_path / "full"

    _prepare_tiny(models, loaders, reuse_dir, pooling="mean", force=True)
    full = _prepare_tiny(models, loaders, full_dir, pooling="cls", force=True)

    def _no_source_extraction(**_kwargs):
        raise AssertionError("A-side extraction must be skipped when the mean cache is reusable")

    with monkeypatch.context() as patch:
        patch.setattr(steer_mod, "_collect_linear_split", _no_source_extraction)
        reused = _prepare_tiny(models, loaders, reuse_dir, pooling="cls")

    for split in ("train", "test"):
        a = steer_mod._cache_split_dir(str(reuse_dir), "source", "target", "tiny_reuse", "linear", split, "residual", "cls")
        b = steer_mod._cache_split_dir(str(full_dir), "source", "target", "tiny_reuse", "linear", split, "residual", "cls")
        for name in ("features_A", "delta_A", "delta_A_blocks", "features_B", "y_A"):
            assert torch.allclose(torch.load(a / f"{name}.pt"), torch.load(b / f"{name}.pt"), atol=1e-6), name
        blocks_a, blocks_b = torch.load(a / "features_B_blocks.pt"), torch.load(b / "features_B_blocks.pt")
        assert blocks_a.keys() == blocks_b.keys()
        assert all(torch.allclose(blocks_a[k], blocks_b[k], atol=1e-6) for k in blocks_a)
    assert reused["diagnostics"]["stage2_test_acc"] == full["diagnostics"]["stage2_test_acc"]


def test_cls_pooling_falls_back_when_mean_cache_is_for_another_target(tmp_path) -> None:
    models = _tiny_models(6)
    loaders = _tiny_loaders()
    _prepare_tiny(models, loaders, tmp_path, pooling="mean", force=True)
    other_target = _make_tiny_clf(depth=3)  # same tags, different weights: stale cache
    reused = _prepare_tiny((models[0], models[1], other_target), loaders, tmp_path, pooling="cls")
    full = _prepare_tiny((models[0], models[1], other_target), loaders, tmp_path / "fresh", pooling="cls", force=True)
    assert reused["diagnostics"]["stage0_test_acc"] == full["diagnostics"]["stage0_test_acc"]


def test_global_strategies_ignore_token_pooling_in_cache(tmp_path) -> None:
    models = _tiny_models(7)
    loaders = _tiny_loaders()
    mean = _prepare_tiny(models, loaders, tmp_path, strategy="global_ridge", pooling="mean", force=True)
    cls = _prepare_tiny(models, loaders, tmp_path, strategy="global_ridge", pooling="cls")
    assert not list(tmp_path.rglob("*_clspool"))
    assert mean["diagnostics"] == cls["diagnostics"]


def _prepare_tiny_steer(tmp_path, clfs, *, task: str, loaders: _Loaders | None = None, **params):
    source_finetuned, source_pretrained, target = clfs
    loaders = loaders or _tiny_loaders()
    return get_method("steer").prepare(
        clf_source=source_finetuned,
        clf_source_pretrained=source_pretrained,
        clf_target=target,
        source_loaders=loaders,
        target_loaders=loaders,
        classnames=_CLASSNAMES,
        task=task,
        source_build_cfg_task=_BUILD_CFG,
        build_cfg_task=_BUILD_CFG,
        device="cpu",
        few_shot=2,
        feature_cache_dir=str(tmp_path / "features"),
        verbose=False,
        **params,
    )


def _tiny_steer_models(target_depth: int = 3):
    source_pretrained = _make_tiny_clf(depth=3)
    source_finetuned = _make_tiny_clf(depth=3)
    source_finetuned.load_state_dict(source_pretrained.state_dict())
    _finetune(source_finetuned)
    return source_finetuned, source_pretrained, _make_tiny_clf(depth=target_depth)


def test_concat_ridge_adds_target_blocks_to_standard_cache(tmp_path) -> None:
    """A standard cache built by global_ridge gains only features_B_blocks.pt for concat_ridge."""
    torch.manual_seed(6)
    clfs = _tiny_steer_models()
    loaders = _tiny_loaders()
    _prepare_tiny_steer(
        tmp_path, clfs, task="TaskC", loaders=loaders, feature_regime="standard", stage_2_strategy="global_ridge"
    )
    split_dirs = [p.parent for p in (tmp_path / "features").rglob("features_A.pt")]
    assert split_dirs and all(not (d / "features_B_blocks.pt").exists() for d in split_dirs)
    mtimes = {p: p.stat().st_mtime_ns for d in split_dirs for p in d.glob("*.pt")}

    prepared = _prepare_tiny_steer(
        tmp_path, clfs, task="TaskC", loaders=loaders, feature_regime="standard", stage_2_strategy="concat_ridge"
    )
    assert all((d / "features_B_blocks.pt").exists() for d in split_dirs)
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes  # A-side and final features not rewritten

    # The cached blocks are the live ones: cached-space and live corrections agree.
    loader = loaders.test
    target = clfs[2]
    captured = []
    with steer_correction_context(target, prepared, alpha=1.0):
        for x, _ in loader:
            captured.append(target.model.encode_image(x))
    cache_dir = next(d for d in split_dirs if d.name == "test")
    f_b = torch.load(cache_dir / "features_B.pt").double()
    blocks = {b: v.double() for b, v in torch.load(cache_dir / "features_B_blocks.pt").items()}
    cached = f_b + prepared["correction_fn"]({"global": f_b, "blocks": blocks})
    assert torch.allclose(torch.cat(captured).double(), cached, atol=1e-5)


def test_concat_ridge_cross_arch_deeper_target(tmp_path) -> None:
    """No block grouping: a deeper target simply contributes more concatenated blocks."""
    torch.manual_seed(7)
    clfs = _tiny_steer_models(target_depth=6)
    for regime in ("standard", "linear"):
        prepared = _prepare_tiny_steer(
            tmp_path,
            clfs,
            task=f"crossarch_concat_{regime}",
            feature_regime=regime,
            stage_2_strategy="concat_ridge",
            force_recompute_features=True,
        )
        x = torch.randn(4, 3, 16, 16)
        with steer_correction_context(clfs[2], prepared, alpha=1.0):
            logits = clfs[2](x)
        assert logits.shape == (4, len(_CLASSNAMES))
        assert torch.isfinite(logits).all()


def test_concat_ridge_standard_reads_target_blocks_from_linear_cache(tmp_path) -> None:
    """target_blocks_cache_dir: B's blocks come from the linear cache; the standard cache is never written to."""
    import pytest

    torch.manual_seed(8)
    clfs = _tiny_steer_models()
    loaders = _tiny_loaders()
    common = dict(task="TaskE", loaders=loaders, stage_2_strategy="concat_ridge")
    _prepare_tiny_steer(tmp_path, clfs, feature_regime="linear", **common)
    _prepare_tiny_steer(
        tmp_path, clfs, task="TaskE", loaders=loaders, feature_regime="standard", stage_2_strategy="global_ridge"
    )
    standard_dirs = [p.parent for p in (tmp_path / "features").rglob("standard/*/features_A.pt")]
    mtimes = {p: p.stat().st_mtime_ns for d in standard_dirs for p in d.glob("*.pt")}

    prepared = _prepare_tiny_steer(
        tmp_path,
        clfs,
        feature_regime="standard",
        target_blocks_cache_dir=str(tmp_path / "features"),
        **common,
    )
    assert all(not (d / "features_B_blocks.pt").exists() for d in standard_dirs)
    assert {p: p.stat().st_mtime_ns for d in standard_dirs for p in d.glob("*.pt")} == mtimes
    x = torch.randn(4, 3, 16, 16)
    with steer_correction_context(clfs[2], prepared, alpha=1.0):
        assert torch.isfinite(clfs[2](x)).all()

    # A linear cache built for a different target must be rejected, not silently paired.
    other = (clfs[0], clfs[1], _make_tiny_clf(depth=3))
    with pytest.raises(ValueError, match="do not match"):
        _prepare_tiny_steer(
            tmp_path / "other",
            other,
            feature_regime="standard",
            target_blocks_cache_dir=str(tmp_path / "features"),
            **common,
        )


def test_concat_ridge_standard_reads_target_blocks_from_linear_cache(tmp_path) -> None:
    """target_blocks_cache_dir: B's blocks come from the linear cache; the standard cache is never written to."""
    torch.manual_seed(8)
    clfs = _tiny_steer_models()
    loaders = _tiny_loaders()
    _prepare_tiny_steer(tmp_path, clfs, task="TaskE", loaders=loaders, feature_regime="linear", stage_2_strategy="concat_ridge")
    _prepare_tiny_steer(tmp_path, clfs, task="TaskE", loaders=loaders, feature_regime="standard", stage_2_strategy="global_ridge")
    standard_files = {p: p.stat().st_mtime_ns for p in (tmp_path / "features").rglob("standard/*/*.pt")}

    prepared = _prepare_tiny_steer(
        tmp_path,
        clfs,
        task="TaskE",
        loaders=loaders,
        feature_regime="standard",
        stage_2_strategy="concat_ridge",
        target_blocks_cache_dir=str(tmp_path / "features"),
    )
    assert {p: p.stat().st_mtime_ns for p in (tmp_path / "features").rglob("standard/*/*.pt")} == standard_files
    with steer_correction_context(clfs[2], prepared, alpha=1.0):
        assert torch.isfinite(clfs[2](torch.randn(4, 3, 16, 16))).all()

    # A linear cache of a different target must be rejected, not silently paired with these images.
    other_target = (clfs[0], clfs[1], _make_tiny_clf(depth=3))
    _prepare_tiny_steer(
        tmp_path / "other", other_target, task="TaskE", loaders=loaders, feature_regime="standard",
        stage_2_strategy="global_ridge",
    )
    try:
        _prepare_tiny_steer(
            tmp_path / "other",
            other_target,
            task="TaskE",
            loaders=loaders,
            feature_regime="standard",
            stage_2_strategy="concat_ridge",
            target_blocks_cache_dir=str(tmp_path / "features"),
        )
    except ValueError as err:
        assert "do not match" in str(err)
    else:
        raise AssertionError("mismatched target_blocks_cache_dir was accepted")


def test_block_ridge_error_profile_matches_hand_computation() -> None:
    g = torch.Generator().manual_seed(11)
    n, num_blocks, d_block, d_out = 12, 3, 5, 4
    blocks = {b: torch.randn(n, d_block, generator=g, dtype=torch.float64) for b in range(num_blocks)}
    targets = torch.randn(n, num_blocks, d_out, generator=g, dtype=torch.float64)
    coefs = steer_mod._fit_block_ridge(
        blocks, targets, selected=torch.arange(n), regularization=0.1, mode="smoothed_residual", rho=0.9
    )
    prof = steer_mod._block_ridge_error_profile(coefs, blocks, targets)
    err_sum = torch.zeros(n, d_out, dtype=torch.float64)
    target_sum = torch.zeros(n, d_out, dtype=torch.float64)
    assert set(prof) == {"local", "cumulative", "target_cumulative"}
    for b in range(num_blocks):
        err = targets[:, b] - blocks[b] @ coefs[b]
        err_sum = err_sum + err
        target_sum = target_sum + targets[:, b]
        assert abs(prof["local"][b] - float(err.norm())) < 1e-9
        assert abs(prof["cumulative"][b] - float(err_sum.norm())) < 1e-9
        assert abs(prof["target_cumulative"][b] - float(target_sum.norm())) < 1e-9

    # Optional logit-space profile: same quantities projected with the target head (K x D_B).
    head = torch.randn(3, d_out, generator=g, dtype=torch.float64)
    prof_logit = steer_mod._block_ridge_error_profile(coefs, blocks, targets, head=head)
    assert set(prof_logit) == {
        "local", "cumulative", "target_cumulative", "logit_local", "logit_cumulative", "logit_target_cumulative",
    }
    for key in ("local", "cumulative", "target_cumulative"):
        assert prof_logit[key] == prof[key]  # default keys unchanged by the head
    err_sum = torch.zeros(n, d_out, dtype=torch.float64)
    target_sum = torch.zeros(n, d_out, dtype=torch.float64)
    for b in range(num_blocks):
        err = targets[:, b] - blocks[b] @ coefs[b]
        err_sum = err_sum + err
        target_sum = target_sum + targets[:, b]
        assert abs(prof_logit["logit_local"][b] - float((err @ head.T).norm())) < 1e-9
        assert abs(prof_logit["logit_cumulative"][b] - float((err_sum @ head.T).norm())) < 1e-9
        assert abs(prof_logit["logit_target_cumulative"][b] - float((target_sum @ head.T).norm())) < 1e-9


def test_steer_block_ridge_diagnostics_flag_is_opt_in(tmp_path) -> None:
    torch.manual_seed(2)
    loaders = _tiny_loaders()
    clf_pre = _make_tiny_clf(depth=3)
    clf_ft = _make_tiny_clf(depth=3)
    clf_ft.load_state_dict(clf_pre.state_dict())
    _finetune(clf_ft)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(diagnostics: bool):
        return get_method("steer").prepare(
            clf_source=clf_ft, clf_source_pretrained=clf_pre, clf_target=clf_target,
            source_loaders=loaders, target_loaders=loaders, classnames=_CLASSNAMES, task="tiny_diag",
            source_build_cfg_task=_BUILD_CFG, build_cfg_task=_BUILD_CFG, device="cpu",
            feature_regime="linear", stage_2_strategy="block_ridge", block_granularity="residual",
            block_ridge_mode="smoothed_residual", rho=0.9, ridge_lambda=0.1, block_ridge_lambda_scaling="trace",
            block_ridge_diagnostics=diagnostics, few_shot=2, feature_cache_dir=str(tmp_path / "features"),
            force_recompute_features=not diagnostics, verbose=False,
        )

    off, on = _prepare(False), _prepare(True)
    assert "block_error_profile" not in off["diagnostics"]
    prof = on["diagnostics"]["block_error_profile"]
    assert prof["mode"] == "smoothed_residual"
    assert len(prof["train"]["local"]) == len(prof["test"]["cumulative"]) == 4  # 3 residual blocks + output
    assert len(prof["train"]["logit_cumulative"]) == len(prof["test"]["logit_local"]) == 4
    assert off["diagnostics"]["stage2_test_acc"] == on["diagnostics"]["stage2_test_acc"]


def test_block_ridge_shared_input_independent_equals_global_ridge() -> None:
    """With the same input X for every block, independent block ridges sum to one ridge on the summed
    target (ridge is linear in its target); the smoothed carry breaks this equivalence."""
    g = torch.Generator().manual_seed(7)
    n_sel, num_blocks, d_in, d_out, lam = 12, 4, 6, 3, 0.5
    x = torch.randn(n_sel, d_in, generator=g, dtype=torch.float64)
    targets = torch.randn(n_sel, num_blocks, d_out, generator=g, dtype=torch.float64)
    shared = {b: x for b in range(num_blocks)}
    selected = torch.arange(n_sel)
    global_pred = x @ steer_mod._ridge(x, targets.sum(dim=1), lam)

    independent = steer_mod._fit_block_ridge(shared, targets, selected=selected, regularization=lam, mode="independent")
    assert torch.allclose(steer_mod._predict_block_ridge(independent, shared), global_pred, atol=1e-8)

    smoothed = steer_mod._fit_block_ridge(
        shared, targets, selected=selected, regularization=lam, mode="smoothed_residual", rho=0.9
    )
    assert not torch.allclose(steer_mod._predict_block_ridge(smoothed, shared), global_pred, atol=1e-6)


def test_steer_block_ridge_final_input_end_to_end(tmp_path) -> None:
    """block_ridge_input='final': every block ridge reads B's final feature. Runs through prepare()
    without B's block activations, applies global @ sum_b W_b live, and in independent mode matches
    global_ridge up to the linearized per-block decomposition error."""
    import pytest

    torch.manual_seed(4)
    loaders = _tiny_loaders()
    clf_source_pretrained = _make_tiny_clf(depth=3)
    clf_source_finetuned = _make_tiny_clf(depth=3)
    clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
    _finetune(clf_source_finetuned)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(stage_2_strategy: str = "block_ridge", **overrides):
        params = dict(
            clf_source=clf_source_finetuned,
            clf_source_pretrained=clf_source_pretrained,
            clf_target=clf_target,
            source_loaders=loaders,
            target_loaders=loaders,
            classnames=_CLASSNAMES,
            task="tiny_final_input",
            source_build_cfg_task=_BUILD_CFG,
            build_cfg_task=_BUILD_CFG,
            device="cpu",
            feature_regime="linear",
            stage_2_strategy=stage_2_strategy,
            block_granularity="residual",
            ridge_lambda=0.1,
            few_shot=2,
            feature_cache_dir=str(tmp_path / "features"),
            verbose=False,
        )
        params.update(overrides)
        return get_method("steer").prepare(**params)

    smoothed = _prepare(
        block_ridge_input="final", block_ridge_mode="smoothed_residual", rho=0.9, block_ridge_lambda_scaling="trace"
    )
    assert smoothed["block_ridge_input"] == "final" and smoothed["needs_target_blocks"] is False
    assert smoothed["diagnostics"]["stage2_global_equiv_test_acc"] is not None

    x = torch.randn(4, 3, 16, 16)
    base = clf_target.model.encode_image(x)
    base = base / base.norm(dim=-1, keepdim=True)
    coef = smoothed["correction_fn"].__kwdefaults__["_coef"]
    with steer_correction_context(clf_target, smoothed, alpha=1.0):
        corrected = clf_target.model.encode_image(x)
    assert torch.allclose(corrected, base + (base.double() @ coef).to(base.dtype), atol=1e-5)

    independent = _prepare(block_ridge_input="final", block_ridge_mode="independent")
    global_ridge = _prepare("global_ridge")
    feats = torch.randn(6, base.shape[1])
    got = independent["correction_fn"]({"global": feats})
    want = global_ridge["correction_fn"]({"global": feats})
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-4)
    assert not torch.allclose(smoothed["correction_fn"]({"global": feats}), got, atol=1e-6)

    with pytest.raises(ValueError, match="block_ridge_input"):
        _prepare(block_ridge_input="last")
    with pytest.raises(ValueError, match="block_ridge_input"):
        _prepare("concat_ridge", block_ridge_input="final")


def test_per_class_subset_matches_few_shot_and_caps_small_classes() -> None:
    """Classes with >= per_class examples get exactly _few_shot's draw (existing caches keep their rows);
    smaller classes are kept whole instead of raising."""
    labels = torch.tensor([0] * 60 + [1] * 55 + [2] * 70)
    assert torch.equal(steer_mod._per_class_subset(labels, 50, seed=0), steer_mod._few_shot(labels, 50, seed=0))

    small = torch.tensor([0] * 60 + [1] * 12 + [2] * 70)
    picked = steer_mod._per_class_subset(small, 50, seed=0)
    counts = torch.bincount(small[picked])
    assert counts.tolist() == [50, 12, 50]
    assert picked.unique().numel() == picked.numel()


def test_stage1_target_corrections_logit_space_skips_head_pinv() -> None:
    w_a, w_b, f_a, delta_a, f_b, labels = _synthetic_dataset(d_b=7, classes=4)
    selected = torch.arange(16)
    kwargs = dict(f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, delta_a_test=delta_a, selected=selected, regularization=1.0)
    logit_train, logit_test = steer_mod._stage1_target_corrections(correction_space="logit", **kwargs)
    feature_train, feature_test = steer_mod._stage1_target_corrections(**kwargs)
    assert logit_train.shape == (16, w_b.shape[0])
    assert logit_test.shape == (f_a.shape[0], w_b.shape[0])
    # w_b [C, D_B] with C < D_B has full row rank, so w_b @ pinv(w_b) = I: the feature-space target
    # read through the head is exactly the logit-space one.
    assert torch.allclose(feature_train @ w_b.T, logit_train, atol=1e-10)
    assert torch.allclose(feature_test @ w_b.T, logit_test, atol=1e-10)


def _prepare_tiny_logit(tmp_path, *, strategy: str, regime: str, **overrides):
    torch.manual_seed(0)
    loaders = _tiny_loaders()
    clf_source_pretrained = _make_tiny_clf(depth=3)
    clf_source_finetuned = _make_tiny_clf(depth=3)
    clf_source_finetuned.load_state_dict(clf_source_pretrained.state_dict())
    _finetune(clf_source_finetuned)
    clf_target = _make_tiny_clf(depth=3)
    kwargs = dict(
        clf_source=clf_source_finetuned,
        clf_source_pretrained=clf_source_pretrained,
        clf_target=clf_target,
        source_loaders=loaders,
        target_loaders=loaders,
        classnames=_CLASSNAMES,
        task=f"tiny_logit_{strategy}_{regime}",
        source_build_cfg_task=_BUILD_CFG,
        build_cfg_task=_BUILD_CFG,
        device="cpu",
        feature_regime=regime,
        stage_2_strategy=strategy,
        few_shot=2,
        feature_cache_dir=str(tmp_path / "features"),
        force_recompute_features=True,
        mlp_epochs=3,
        verbose=False,
    )
    kwargs.update(overrides)
    return get_method("steer").prepare(**kwargs), clf_target, loaders


def test_steer_logit_space_end_to_end(tmp_path) -> None:
    """correction_space='logit': Stage 2 predicts [C] logit offsets, added after the head at eval."""
    import pytest

    cases = [
        ("global_ridge", "standard", {}),
        ("global_mlp", "standard", {}),
        ("block_ridge", "linear", {}),
        ("block_ridge", "linear", {"block_ridge_input": "final", "block_ridge_mode": "smoothed_residual"}),
        ("concat_ridge", "standard", {}),
        ("concat_ridge", "linear", {}),
    ]
    for strategy, regime, extra in cases:
        prepared, clf_target, loaders = _prepare_tiny_logit(
            tmp_path, strategy=strategy, regime=regime, correction_space="logit", **extra
        )
        assert prepared["correction_space"] == "logit"
        assert prepared["logit_unit_scale"] == pytest.approx(1.0)
        x = torch.randn(4, 3, 16, 16)
        plain = clf_target(x)
        with steer_correction_context(clf_target, prepared, alpha=0.0):
            at_zero = clf_target(x)
            # encode_image is untouched: the correction never enters feature space.
            assert torch.equal(clf_target.model.encode_image(x), clf_target.model.visual(x))
        assert torch.allclose(at_zero, plain, atol=1e-5)
        with steer_correction_context(clf_target, prepared, alpha=1.0):
            logits = clf_target(x)
        assert "forward" not in vars(clf_target)  # restored
        assert logits.shape == (4, len(_CLASSNAMES))
        assert torch.isfinite(logits).all()

        # logits == logit_scale * (norm(f) @ W^T + c), with c the raw Stage 2 prediction.
        global_feat = torch.nn.functional.normalize(clf_target.model.encode_image(x), dim=-1)
        activations = {"global": global_feat}
        if prepared["needs_target_blocks"]:
            capture_context, final_ids, expected = steer_mod._activation_capture_for_visual(
                clf_target.model.visual, prepared["block_granularity"], prepared["token_pooling"]
            )
            with capture_context as capture:
                out = clf_target.model.encode_image(x)
            activations["blocks"] = steer_mod._complete_captured_activations(
                capture, final_output=torch.nn.functional.normalize(out, dim=-1), final_ids=final_ids, expected_blocks=expected
            )
        correction = prepared["correction_fn"](activations)
        assert correction.shape == (4, len(_CLASSNAMES))
        want = plain + clf_target.logit_scale * correction
        assert torch.allclose(logits, want, rtol=1e-4, atol=1e-3)

        # Live accuracy through the wrapper matches the cached-space Stage 2 diagnostic.
        correct = total = 0
        with steer_correction_context(clf_target, prepared, alpha=1.0):
            for images, labels in loaders.test:
                correct += int(clf_target(images).argmax(dim=-1).eq(labels).sum())
                total += int(labels.numel())
        assert correct / total == pytest.approx(prepared["diagnostics"]["stage2_test_acc"])

    with pytest.raises(ValueError, match="correction_space"):
        _prepare_tiny_logit(tmp_path, strategy="global_ridge", regime="standard", correction_space="weights")


def test_block_ridge_diagnostics_in_logit_space(tmp_path) -> None:
    """correction_space='logit' + block_ridge_diagnostics: residuals are already [C] logits, so the profile
    is computed without pinv(w_b) and without the extra head projection, and the correction is unchanged."""
    common = dict(
        strategy="block_ridge", regime="linear", correction_space="logit", block_granularity="residual",
        block_ridge_mode="smoothed_residual", rho=0.9, ridge_lambda=0.1,
    )
    off, _, _ = _prepare_tiny_logit(tmp_path, **common, block_ridge_diagnostics=False)
    on, _, _ = _prepare_tiny_logit(tmp_path, **common, block_ridge_diagnostics=True)
    assert "block_error_profile" not in off["diagnostics"]
    prof = on["diagnostics"]["block_error_profile"]
    assert prof["correction_space"] == "logit"
    for split_name in ("train", "test"):
        assert set(prof[split_name]) == {"local", "cumulative", "target_cumulative"}
        assert len(prof[split_name]["local"]) == 4  # 3 residual blocks + output
        assert torch.isfinite(torch.tensor(prof[split_name]["cumulative"])).all()
    assert off["diagnostics"]["stage2_test_acc"] == on["diagnostics"]["stage2_test_acc"]


def test_block_pairing_permutation() -> None:
    import pytest

    assert steer_mod._block_pairing_permutation(5, "identity").tolist() == [0, 1, 2, 3, 4]
    assert steer_mod._block_pairing_permutation(5, "reversed").tolist() == [4, 3, 2, 1, 0]
    a = steer_mod._block_pairing_permutation(13, "random", seed=7)
    b = steer_mod._block_pairing_permutation(13, "random", seed=7)
    c = steer_mod._block_pairing_permutation(13, "random", seed=8)
    assert sorted(a.tolist()) == list(range(13))
    assert torch.equal(a, b) and not torch.equal(a, c)
    # The private generator must not consume or depend on the global RNG.
    torch.manual_seed(0)
    before = torch.rand(1)
    torch.manual_seed(0)
    steer_mod._block_pairing_permutation(13, "random", seed=7)
    assert torch.equal(torch.rand(1), before)
    with pytest.raises(ValueError, match="block_pairing_seed"):
        steer_mod._block_pairing_permutation(4, "random")
    with pytest.raises(ValueError, match="block_pairing"):
        steer_mod._block_pairing_permutation(4, "shuffle")


def test_steer_block_ridge_block_pairing_end_to_end(tmp_path) -> None:
    """block_pairing only re-pairs targets with input blocks: identity is bit-exact with the default,
    a non-identity pairing equals _fit_block_ridge on hand-permuted targets (carry in input order)."""
    import pytest

    torch.manual_seed(5)
    loaders = _tiny_loaders()
    clf_pre = _make_tiny_clf(depth=3)
    clf_ft = _make_tiny_clf(depth=3)
    clf_ft.load_state_dict(clf_pre.state_dict())
    _finetune(clf_ft)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(**overrides):
        params = dict(
            clf_source=clf_ft, clf_source_pretrained=clf_pre, clf_target=clf_target,
            source_loaders=loaders, target_loaders=loaders, classnames=_CLASSNAMES, task="tiny_pairing",
            source_build_cfg_task=_BUILD_CFG, build_cfg_task=_BUILD_CFG, device="cpu",
            feature_regime="linear", stage_2_strategy="block_ridge", block_granularity="residual",
            block_ridge_mode="smoothed_residual", rho=0.9, ridge_lambda=0.1, block_ridge_lambda_scaling="trace",
            block_ridge_diagnostics=True, few_shot=2, feature_cache_dir=str(tmp_path / "features"), verbose=False,
        )
        params.update(overrides)
        return get_method("steer").prepare(**params)

    default = _prepare()
    identity = _prepare(block_pairing="identity")
    assert "block_pairing" not in default["diagnostics"]
    c_default = default["correction_fn"].__kwdefaults__["_coefficients"]
    c_identity = identity["correction_fn"].__kwdefaults__["_coefficients"]
    assert all(torch.equal(x, y) for x, y in zip(c_default, c_identity))

    for pairing, seed in (("reversed", None), ("random", 3)):
        shuffled = _prepare(block_pairing=pairing, block_pairing_seed=seed)
        diag = shuffled["diagnostics"]
        perm = torch.tensor(diag["block_pairing_perm"])
        assert diag["block_pairing"] == pairing and sorted(perm.tolist()) == list(range(4))
        # Profile targets are the re-paired ones: target_cumulative of the last block is a sum over
        # all blocks, hence invariant to the pairing.
        for split in ("train", "test"):
            assert abs(
                diag["block_error_profile"][split]["target_cumulative"][-1]
                - default["diagnostics"]["block_error_profile"][split]["target_cumulative"][-1]
            ) < 1e-8
        c_shuffled = shuffled["correction_fn"].__kwdefaults__["_coefficients"]
        assert not all(torch.allclose(x, y) for x, y in zip(c_default, c_shuffled))

    # Pairing is a target re-indexing in front of an unchanged fit.
    g = torch.Generator().manual_seed(1)
    blocks = {b: torch.randn(8, 5, generator=g, dtype=torch.float64) for b in range(4)}
    targets = torch.randn(8, 4, 3, generator=g, dtype=torch.float64)
    perm = steer_mod._block_pairing_permutation(4, "random", seed=11)
    manual = torch.stack([targets[:, int(p)] for p in perm], dim=1)
    fit = lambda t: steer_mod._fit_block_ridge(
        blocks, t, selected=torch.arange(8), regularization=0.1, mode="smoothed_residual", rho=0.9
    )
    assert all(torch.equal(x, y) for x, y in zip(fit(targets[:, perm]), fit(manual)))

    with pytest.raises(ValueError, match="block_pairing_seed"):
        _prepare(block_pairing="random")
    with pytest.raises(ValueError, match="block_pairing"):
        _prepare(block_pairing="reversed", block_ridge_input="final")
    with pytest.raises(ValueError, match="block_pairing"):
        _prepare(stage_2_strategy="global_ridge", block_pairing="reversed")


def test_steer_cached_eval_batch_size_matches_full_eval(tmp_path) -> None:
    """cached_eval_batch_size only chunks the cached-space stage-2 check: fit and accuracy are unchanged,
    the correction on the cached test blocks agrees to float64 round-off, and bad values are rejected."""
    import pytest

    torch.manual_seed(7)
    loaders = _tiny_loaders()
    clf_pre = _make_tiny_clf(depth=3)
    clf_ft = _make_tiny_clf(depth=3)
    clf_ft.load_state_dict(clf_pre.state_dict())
    _finetune(clf_ft)
    clf_target = _make_tiny_clf(depth=3)

    def _prepare(**overrides):
        params = dict(
            clf_source=clf_ft, clf_source_pretrained=clf_pre, clf_target=clf_target,
            source_loaders=loaders, target_loaders=loaders, classnames=_CLASSNAMES, task="tiny_cached_eval",
            source_build_cfg_task=_BUILD_CFG, build_cfg_task=_BUILD_CFG, device="cpu",
            feature_regime="linear", stage_2_strategy="block_ridge", block_granularity="residual",
            block_ridge_mode="independent", rho=0.0, ridge_lambda=0.1, block_ridge_lambda_scaling="trace",
            few_shot=2, feature_cache_dir=str(tmp_path / "features"), verbose=False,
        )
        params.update(overrides)
        return get_method("steer").prepare(**params)

    for space in ("feature", "logit"):
        full = _prepare(correction_space=space)
        for chunk in (1, 3, 10_000):
            chunked = _prepare(correction_space=space, cached_eval_batch_size=chunk)
            c_full = full["correction_fn"].__kwdefaults__["_coefficients"]
            c_chunk = chunked["correction_fn"].__kwdefaults__["_coefficients"]
            assert all(torch.equal(x, y) for x, y in zip(c_full, c_chunk))
            assert chunked["diagnostics"]["stage2_test_acc"] == full["diagnostics"]["stage2_test_acc"]
            assert chunked["diagnostics"]["stage0_test_acc"] == full["diagnostics"]["stage0_test_acc"]

    for bad in (0, -4):
        with pytest.raises(ValueError, match="cached_eval_batch_size"):
            _prepare(cached_eval_batch_size=bad)


# --------------------------------------------------------------------------
# stage1_mode="logit_distill": linear probe distillation (opt-in baseline, not steer)
# --------------------------------------------------------------------------


def test_ridge_or_pinv_closed_forms() -> None:
    import pytest

    torch.manual_seed(0)
    x, y = torch.randn(20, 6, dtype=torch.float64), torch.randn(20, 3, dtype=torch.float64)
    # lambda=0, n > d: ordinary least squares.
    coef, lam = steer_mod._ridge_or_pinv(x, y, 0.0)
    assert lam == 0.0
    assert torch.allclose(coef, torch.linalg.lstsq(x, y).solution, atol=1e-10)
    # lambda > 0: (X^T X + lambda I)^-1 X^T Y.
    coef, lam = steer_mod._ridge_or_pinv(x, y, 0.5)
    assert lam == 0.5
    assert torch.allclose(coef, torch.linalg.solve(x.T @ x + 0.5 * torch.eye(6, dtype=x.dtype), x.T @ y), atol=1e-10)
    # trace scaling: lambda * tr(X X^T) / n; a no-op for unit-norm rows.
    _, lam = steer_mod._ridge_or_pinv(x, y, 0.5, "trace")
    assert lam == pytest.approx(0.5 * float(x.square().sum()) / 20)
    unit = torch.nn.functional.normalize(x, dim=-1)
    a, lam_unit = steer_mod._ridge_or_pinv(unit, y, 0.5, "trace")
    assert lam_unit == pytest.approx(0.5)
    assert torch.allclose(a, steer_mod._ridge_or_pinv(unit, y, 0.5)[0], atol=1e-10)
    # lambda=0, n <= d: minimum-norm interpolant (fits the support set exactly).
    xs, ys = x[:4], y[:4]
    coef, _ = steer_mod._ridge_or_pinv(xs, ys, 0.0)
    assert torch.allclose(xs @ coef, ys, atol=1e-10)
    with pytest.raises(ValueError, match="non-negative"):
        steer_mod._ridge_or_pinv(x, y, -1.0)
    with pytest.raises(ValueError, match="global_ridge_lambda_scaling"):
        steer_mod._ridge_or_pinv(x, y, 1.0, "per_block")


def _lpd_cached(tmp_path, task: str, regime: str):
    train = steer_mod._cache_split_dir(str(tmp_path / "features"), "source", "target", task, regime, "train")
    test = steer_mod._cache_split_dir(str(tmp_path / "features"), "source", "target", task, regime, "test")
    load = lambda d, n: torch.load(d / f"{n}.pt").double()  # noqa: E731
    w_a = steer_mod._load_cached_head(train / "head_A.pt").double()
    w_b = steer_mod._load_cached_head(train / "head_B.pt").double()
    return {s: {n: load(d, n) for n in ("features_A", "delta_A", "features_B")} | {"y": torch.load(d / "y_A.pt").long()}
            for s, d in (("train", train), ("test", test))}, w_a, w_b


def test_steer_logit_distill_matches_closed_form_end_to_end(tmp_path) -> None:
    """LPD fits L = ridge(H_B, E) on the support set, E = H_A^ft T_A - H_B T_B; feature and logit spaces give
    the same logit offset H_B L (argmax-identical live predictions); the live accuracy matches the cached one."""
    import pytest

    for regime in ("standard", "linear"):
        task = f"tiny_logit_global_ridge_{regime}"
        results = {}
        for space in ("logit", "feature"):
            prepared, clf_target, loaders = _prepare_tiny_logit(
                tmp_path, strategy="global_ridge", regime=regime, correction_space=space,
                stage1_mode="logit_distill", ridge_lambda=0.05,
            )
            results[space] = (prepared, clf_target, loaders)
        data, w_a, w_b = _lpd_cached(tmp_path, task, regime)
        tr, te = data["train"], data["test"]
        selected = steer_mod._few_shot(tr["y"], 2, 42)
        residual = (tr["features_A"] + tr["delta_A"])[selected] @ w_a.T - tr["features_B"][selected] @ w_b.T
        h_b = tr["features_B"][selected]
        want_l = torch.linalg.solve(h_b.T @ h_b + 0.05 * torch.eye(h_b.shape[1], dtype=h_b.dtype), h_b.T @ residual)

        p_logit, clf_target, loaders = results["logit"]
        p_feat = results["feature"][0]
        coef_logit = p_logit["correction_fn"].__kwdefaults__["_coef"]
        coef_feat = p_feat["correction_fn"].__kwdefaults__["_coef"]
        assert torch.allclose(coef_logit, want_l, atol=1e-8)
        # feature-space correction mapped through the head = the logit-space offset.
        assert torch.allclose(coef_feat @ w_b.T, coef_logit, atol=1e-8)

        for p in (p_logit, p_feat):
            diag = p["diagnostics"]
            assert diag["stage1_mode"] == "logit_distill" and p["stage1_mode"] == "logit_distill"
            assert diag["stage1_rank"] is None and diag["global_ridge_lambda_effective"] == 0.05
            # teacher = A's own cached accuracy (oracle target = full residual).
            teacher = float(((te["features_A"] + te["delta_A"]) @ w_a.T).argmax(1).eq(te["y"]).double().mean())
            assert diag["teacher_test_acc"] == pytest.approx(teacher)
        assert p_logit["diagnostics"]["stage2_test_acc"] == pytest.approx(p_feat["diagnostics"]["stage2_test_acc"])

        # Live forward through the wrapper reproduces the cached-space Stage 2 accuracy.
        for p in (p_logit, p_feat):
            correct = total = 0
            with steer_correction_context(clf_target, p, alpha=1.0):
                for images, labels in loaders.test:
                    correct += int(clf_target(images).argmax(dim=-1).eq(labels).sum())
                    total += int(labels.numel())
            assert correct / total == pytest.approx(p["diagnostics"]["stage2_test_acc"])


def test_logit_distill_target_equals_exact_stage1_target_iff_underdetermined() -> None:
    """With n <= d_A and full-row-rank delta_A on the support, unregularized standard Stage 1 reproduces E on the
    support (delta_A pinv(delta_A) E = E), so steer's logit-space global_ridge then coincides with LPD; with n > d_A
    Stage 1 only keeps E's projection on span(delta_A) and the two differ."""
    torch.manual_seed(0)
    n_total, d_a, d_b, c = 40, 12, 9, 5
    f_a, delta_a = torch.randn(n_total, d_a, dtype=torch.float64), torch.randn(n_total, d_a, dtype=torch.float64)
    f_b = torch.randn(n_total, d_b, dtype=torch.float64)
    w_a, w_b = torch.randn(c, d_a, dtype=torch.float64), torch.randn(c, d_b, dtype=torch.float64)
    for n, equal in ((8, True), (30, False)):
        selected = torch.arange(n)
        stage1_train, _ = steer_mod._stage1_target_corrections(
            f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, delta_a_test=delta_a, selected=selected,
            regularization=0.0, regularization_mode="standard", correction_space="logit",
        )
        residual = steer_mod._logit_distill_residual(
            f_a=f_a[selected], delta_a=delta_a[selected], w_a=w_a, f_b=f_b[selected], w_b=w_b
        )
        assert torch.allclose(stage1_train, residual, atol=1e-8) is equal
        coef_steer = steer_mod._ridge(f_b[selected], stage1_train, 0.1)
        coef_lpd, _ = steer_mod._ridge_or_pinv(f_b[selected], residual, 0.1)
        assert torch.allclose(coef_steer, coef_lpd, atol=1e-8) is equal


def test_steer_logit_distill_is_opt_in_and_validated(tmp_path) -> None:
    import pytest

    default, _, _ = _prepare_tiny_logit(tmp_path, strategy="global_ridge", regime="standard")
    assert "stage1_mode" not in default and "stage1_mode" not in default["diagnostics"]
    for strategy in ("global_mlp", "block_ridge"):
        with pytest.raises(ValueError, match="logit_distill"):
            _prepare_tiny_logit(tmp_path, strategy=strategy, regime="linear", stage1_mode="logit_distill")
    # concat_ridge is the other allowed input (all of B's blocks, concatenated), with a strictly positive lambda
    # and without the global_ridge-only lambda scaling.
    with pytest.raises(ValueError, match="ridge_lambda > 0"):
        _prepare_tiny_logit(tmp_path, strategy="concat_ridge", regime="linear", stage1_mode="logit_distill", ridge_lambda=0.0)
    with pytest.raises(ValueError, match="global_ridge_lambda_scaling"):
        _prepare_tiny_logit(
            tmp_path, strategy="concat_ridge", regime="linear", stage1_mode="logit_distill",
            ridge_lambda=0.3, global_ridge_lambda_scaling="trace",
        )
    plain, _, _ = _prepare_tiny_logit(tmp_path, strategy="concat_ridge", regime="linear")  # default: untouched
    assert "stage1_mode" not in plain and "lpd_input" not in plain["diagnostics"]
    with pytest.raises(ValueError, match="stage1_mode"):
        _prepare_tiny_logit(tmp_path, strategy="global_ridge", regime="standard", stage1_mode="direct")
    with pytest.raises(ValueError, match="global_ridge_lambda_scaling"):
        _prepare_tiny_logit(tmp_path, strategy="global_ridge", regime="standard", global_ridge_lambda_scaling="trace")
    # lambda=0 is allowed in LPD (minimum-norm least squares), unlike steer's ridge.
    zero, _, _ = _prepare_tiny_logit(
        tmp_path, strategy="global_ridge", regime="standard", stage1_mode="logit_distill", ridge_lambda=0.0
    )
    assert zero["diagnostics"]["global_ridge_lambda_effective"] == 0.0


def test_steer_logit_distill_concat_ridge_matches_closed_form_end_to_end(tmp_path) -> None:
    """LPD with all of B's blocks as input: the correction is _fit_concat_ridge on the support-set logit residual
    E = H_A^ft T_A - H_B T_B (Stage 1 skipped); feature and logit spaces give the same logit offset; the live
    accuracy matches the cached-space one."""
    import pytest

    for regime in ("standard", "linear"):
        task = f"tiny_logit_concat_ridge_{regime}"
        res = {}
        for space in ("logit", "feature"):
            res[space] = _prepare_tiny_logit(
                tmp_path, strategy="concat_ridge", regime=regime, correction_space=space,
                stage1_mode="logit_distill", ridge_lambda=0.3,
            )
        data, w_a, w_b = _lpd_cached(tmp_path, task, regime)
        tr, te = data["train"], data["test"]
        base = steer_mod._cache_split_dir(str(tmp_path / "features"), "source", "target", task, regime, "train")
        blocks_tr = {int(b): v.double() for b, v in torch.load(base / "features_B_blocks.pt").items()}
        base_te = steer_mod._cache_split_dir(str(tmp_path / "features"), "source", "target", task, regime, "test")
        blocks_te = {int(b): v.double() for b, v in torch.load(base_te / "features_B_blocks.pt").items()}
        selected = steer_mod._few_shot(tr["y"], 2, 42)
        residual = (tr["features_A"] + tr["delta_A"])[selected] @ w_a.T - tr["features_B"][selected] @ w_b.T
        want, info = steer_mod._fit_concat_ridge({b: v[selected] for b, v in blocks_tr.items()}, residual, regularization=0.3)

        p_logit, clf_target, loaders = res["logit"]
        p_feat = res["feature"][0]
        got = [c.double() for c in p_logit["correction_fn"].__kwdefaults__["_coefficients"]]
        assert len(got) == len(want) and all(torch.allclose(g, w, atol=1e-8) for g, w in zip(got, want))
        activations = {"global": te["features_B"].float(), "blocks": {b: v.float() for b, v in blocks_te.items()}}
        c_logit = p_logit["correction_fn"](activations).double()
        c_feat = p_feat["correction_fn"](activations).double()
        assert torch.allclose(c_logit, sum(blocks_te[b] @ want[b] for b in sorted(blocks_te)), atol=1e-4)
        assert torch.allclose(c_feat @ w_b.T, c_logit, atol=1e-4)  # same logit offset, W_B pinv(W_B) = I

        for p in (p_logit, p_feat):
            diag = p["diagnostics"]
            assert diag["stage1_mode"] == "logit_distill" and diag["lpd_input"] == "concat_blocks"
            assert diag["stage1_rank"] is None and diag["concat_input_dim"] == info["input_dim"]
            assert diag["concat_effective_lambda"] == pytest.approx(0.3)
        assert p_logit["diagnostics"]["stage2_test_acc"] == pytest.approx(p_feat["diagnostics"]["stage2_test_acc"])
        for p in (p_logit, p_feat):
            correct = total = 0
            with steer_correction_context(clf_target, p, alpha=1.0):
                for images, labels in loaders.test:
                    correct += int(clf_target(images).argmax(dim=-1).eq(labels).sum())
                    total += int(labels.numel())
            assert correct / total == pytest.approx(p["diagnostics"]["stage2_test_acc"])


# --------------------------------------------------------------------------
# Integrated regime: per-block residuals as midpoint-rule path integrals of masked JVPs
# --------------------------------------------------------------------------


def _integrated_split(clfs, loaders, *, k: int, compute_blocks: bool = True):
    source_finetuned, source_pretrained, target = clfs
    return steer_mod._collect_integrated_split(
        source_finetuned_visual=source_finetuned.model.visual.eval(),
        source_pretrained_visual=source_pretrained.model.visual.eval(),
        target_visual=target.model.visual.eval(),
        source_loader=loaders.train,
        target_loader=loaders.train,
        device=torch.device("cpu"),
        integration_points=k,
        compute_blocks=compute_blocks,
    )


def _strongly_finetuned_models(scale: float = 0.5):
    """A fine-tuning large enough that the path theta0 -> theta1 is visibly nonlinear.

    The tiny visual's class/positional embeddings are zero-initialised, which makes the class token
    an all-zero vector at theta0: ``ln_pre`` is then (numerically) discontinuous at t=0 and no
    quadrature on a handful of nodes recovers the exact residual. A real CLIP has no such point, so
    the pretrained embeddings are randomised here.
    """
    torch.manual_seed(0)
    source_pretrained = _make_tiny_clf(depth=3)
    with torch.no_grad():
        source_pretrained.model.visual.class_embedding.normal_()
        source_pretrained.model.visual.positional_embedding.normal_()
    source_finetuned = _make_tiny_clf(depth=3)
    source_finetuned.load_state_dict(source_pretrained.state_dict())
    with torch.no_grad():
        for p in source_finetuned.model.visual.parameters():
            p.add_(scale * torch.randn_like(p))
    return source_finetuned, source_pretrained, _make_tiny_clf(depth=3)


def test_integrated_block_residuals_sum_converges_to_exact_residual() -> None:
    clfs = _strongly_finetuned_models()
    loaders = _tiny_loaders()
    errors = {}
    blocks = {}
    for k in (2, 4, 8, 16):
        data = _integrated_split(clfs, loaders, k=k)
        assert data["delta_A_blocks"].shape == (16, 4, 5)  # 3 residual blocks + output block
        total = data["delta_A_blocks"].double().sum(dim=1)
        exact = data["delta_A"].double()
        errors[k] = float((total - exact).norm() / exact.norm())
        blocks[k] = data["delta_A_blocks"].double()
        # The stored per-example diagnostic is the same quantity, row by row.
        row_error = (total - exact).norm(dim=-1) / (exact.norm(dim=-1) + 1e-12)
        assert torch.allclose(data["integration_rel_error"].double(), row_error, atol=1e-5)
    # Midpoint rule: the error of the sum shrinks with K and is small at K=16.
    assert errors[16] < errors[8] < errors[4] < errors[2]
    assert errors[16] < 1e-2
    assert errors[2] > 10 * errors[16]
    # Each block's contribution stabilises when K doubles.
    assert float((blocks[16] - blocks[8]).norm()) < float((blocks[4] - blocks[2]).norm())


def test_integrated_block_residuals_match_reference_quadrature() -> None:
    """The collector equals the note's pseudocode written out directly (torch.func.jvp)."""
    from torch.func import functional_call, jvp

    clfs = _strongly_finetuned_models()
    loaders = _tiny_loaders()
    k_points = 3
    data = _integrated_split(clfs, loaders, k=k_points)
    visual0 = clfs[1].model.visual.eval()
    theta0 = {n: p.detach() for n, p in visual0.named_parameters()}
    theta1 = {n: p.detach() for n, p in clfs[0].model.visual.named_parameters()}
    delta = {n: theta1[n] - theta0[n] for n in theta0}
    block_of, num_blocks = steer_mod._parameter_blocks_for_visual(visual0, "residual")
    block_of = dict(zip(theta0, block_of, strict=True))
    x = torch.cat([batch[0] for batch in loaders.train])

    def f(params):
        return steer_mod._l2_normalize(functional_call(visual0, params, (x,)))

    expected = torch.zeros(x.shape[0], num_blocks, 5, dtype=torch.float64)
    for k in range(k_points):
        t = (k + 0.5) / k_points
        theta_t = {n: theta0[n] + t * delta[n] for n in theta0}
        for ell in range(num_blocks):
            v = {n: delta[n] if block_of[n] == ell else torch.zeros_like(delta[n]) for n in theta0}
            _, g = jvp(f, (theta_t,), (v,))
            expected[:, ell] += g.detach().double() / k_points
    assert torch.allclose(data["delta_A_blocks"].double(), expected, atol=1e-5)


def test_integrated_residual_of_single_block_update_stays_in_that_block() -> None:
    torch.manual_seed(0)
    source_pretrained = _make_tiny_clf(depth=3)
    source_finetuned = _make_tiny_clf(depth=3)
    source_finetuned.load_state_dict(source_pretrained.state_dict())
    with torch.no_grad():
        for name, p in source_finetuned.model.visual.named_parameters():
            if name.startswith("transformer.resblocks.1."):
                p.add_(0.5 * torch.randn_like(p))
    data = _integrated_split((source_finetuned, source_pretrained, _make_tiny_clf(depth=3)), _tiny_loaders(), k=4)
    blocks = data["delta_A_blocks"]
    assert float(blocks[:, 1].abs().max()) > 0
    for other in (0, 2, 3):
        assert float(blocks[:, other].abs().max()) == 0.0


def test_integrated_exact_residual_and_target_features_equal_standard_regime() -> None:
    clfs = _strongly_finetuned_models()
    loaders = _tiny_loaders()
    integrated = _integrated_split(clfs, loaders, k=2)
    standard = steer_mod._collect_standard_split(
        clf_source_finetuned_visual=clfs[0].model.visual.eval(),
        clf_source_pretrained_visual=clfs[1].model.visual.eval(),
        target_visual=clfs[2].model.visual.eval(),
        source_loader=loaders.train,
        target_loader=loaders.train,
        device=torch.device("cpu"),
        block_granularity="residual",
    )
    for name in ("features_A", "delta_A", "features_B", "y_A"):
        assert torch.equal(integrated[name], standard[name]), name
    for block_id, value in standard["features_B_blocks"].items():
        assert torch.equal(integrated["features_B_blocks"][block_id], value)
    # Chunking the JVPs (lower peak memory) gives the same per-example integrals.
    source_finetuned, source_pretrained, target = clfs
    chunked = steer_mod._collect_integrated_split(
        source_finetuned_visual=source_finetuned.model.visual.eval(),
        source_pretrained_visual=source_pretrained.model.visual.eval(),
        target_visual=target.model.visual.eval(),
        source_loader=loaders.train,
        target_loader=loaders.train,
        device=torch.device("cpu"),
        integration_points=2,
        integration_batch_size=3,
    )
    assert torch.allclose(chunked["delta_A_blocks"], integrated["delta_A_blocks"], atol=1e-6)
    assert torch.equal(chunked["delta_A"], integrated["delta_A"])
    # Without the integration only the two forwards run.
    light = _integrated_split(clfs, loaders, k=2, compute_blocks=False)
    assert "delta_A_blocks" not in light and "integration_rel_error" not in light
    assert torch.equal(light["delta_A"], standard["delta_A"])


def test_steer_integrated_regime_end_to_end_and_cache_per_k(tmp_path, monkeypatch) -> None:
    import pytest

    clfs = _strongly_finetuned_models(scale=0.2)
    loaders = _tiny_loaders()
    common = dict(task="TaskI", loaders=loaders, stage_2_strategy="block_ridge", block_ridge_mode="smoothed_residual",
                  rho=0.9, block_ridge_lambda_scaling="trace", ridge_lambda=0.1)
    prepared = _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=4, **common)
    assert prepared["integration_points"] == 4
    assert prepared["feature_regime"] == "integrated_k4"
    diagnostics = prepared["diagnostics"]
    assert diagnostics["integration_points"] == 4
    assert 0 <= diagnostics["integration_rel_error_mean"] <= diagnostics["integration_rel_error_max"]
    with steer_correction_context(clfs[2], prepared, alpha=1.0):
        logits = clfs[2](torch.randn(4, 3, 16, 16))
    assert logits.shape == (4, len(_CLASSNAMES)) and torch.isfinite(logits).all()

    pair_dir = tmp_path / "features" / "source_to_target" / "TaskI"
    train_dir = pair_dir / "integrated_k4" / "train"
    for name in ("features_A", "delta_A", "delta_A_blocks", "features_B", "features_B_blocks", "y_A", "integration_rel_error"):
        assert (train_dir / f"{name}.pt").exists(), name
    # Test-split block integrals are skipped (diagnostics only) ...
    assert not (pair_dir / "integrated_k4" / "test" / "delta_A_blocks.pt").exists()
    assert (pair_dir / "integrated_k4" / "test" / "delta_A.pt").exists()

    # A second prepare only reads the cache, and reports the cached quadrature error.
    def _no_extraction(**kwargs):
        raise AssertionError("features must come from the cache")

    with monkeypatch.context() as patch:
        patch.setattr(steer_mod, "_collect_integrated_split", _no_extraction)
        again = _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=4, **common)
    assert again["diagnostics"]["integration_rel_error_max"] == diagnostics["integration_rel_error_max"]
    x = torch.randn(4, 3, 16, 16)
    with steer_correction_context(clfs[2], prepared, alpha=1.0):
        first = clfs[2](x)
    with steer_correction_context(clfs[2], again, alpha=1.0):
        second = clfs[2](x)
    assert torch.equal(first, second)

    # K is part of the cache key: another K extracts into its own directory.
    other = _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=2, **common)
    assert (pair_dir / "integrated_k2" / "train" / "delta_A_blocks.pt").exists()
    assert other["diagnostics"]["integration_rel_error_mean"] > diagnostics["integration_rel_error_mean"]
    k2 = torch.load(pair_dir / "integrated_k2" / "train" / "delta_A_blocks.pt")
    k4 = torch.load(train_dir / "delta_A_blocks.pt")
    assert not torch.equal(k2, k4)
    assert torch.equal(torch.load(pair_dir / "integrated_k2" / "train" / "delta_A.pt"), torch.load(train_dir / "delta_A.pt"))

    # ... unless the per-block diagnostics ask for them.
    _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=2, block_ridge_diagnostics=True, **common)
    assert (pair_dir / "integrated_k2" / "test" / "delta_A_blocks.pt").exists()

    # Other Stage 2 strategies accept the regime too (they only use the exact residual).
    _prepare_tiny_steer(tmp_path, clfs, task="TaskI", loaders=loaders, feature_regime="integrated", integration_points=2,
                        stage_2_strategy="global_ridge")

    for bad in (None, 0, -1, 2.5, True):
        with pytest.raises(ValueError, match="integration_points"):
            _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=bad, **common)
    with pytest.raises(ValueError, match="integration_points"):
        _prepare_tiny_steer(tmp_path, clfs, feature_regime="linear", integration_points=4, **common)
    with pytest.raises(ValueError, match="integration_batch_size"):
        _prepare_tiny_steer(tmp_path, clfs, feature_regime="linear", integration_batch_size=4, **common)
    with pytest.raises(ValueError, match="integration_batch_size"):
        _prepare_tiny_steer(tmp_path, clfs, feature_regime="integrated", integration_points=2, integration_batch_size=0, **common)
    _prepare_tiny_steer(tmp_path, clfs, task="TaskChunk", loaders=loaders, feature_regime="integrated", integration_points=2,
                        integration_batch_size=2, stage_2_strategy="block_ridge")
    with pytest.raises(ValueError, match="block_ridge requires"):
        _prepare_tiny_steer(tmp_path, clfs, task="TaskI", loaders=loaders, feature_regime="standard", stage_2_strategy="block_ridge")
