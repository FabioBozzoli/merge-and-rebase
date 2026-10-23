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


def test_group_blocks_concat_and_sum_avg() -> None:
    blocks = {b: torch.full((2, 3), float(b)) for b in range(4)}
    concat = steer_mod._group_blocks_concat(blocks, 2)
    assert set(concat) == {0, 1}
    assert concat[0].shape == (2, 6)

    avg = steer_mod._group_blocks_sum_avg(blocks, 2)
    assert set(avg) == {0, 1}
    assert avg[0].shape == (2, 3)
    assert torch.allclose(avg[0], torch.full((2, 3), 0.5))


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
