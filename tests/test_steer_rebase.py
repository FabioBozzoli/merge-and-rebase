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
        need_blocks=False,
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

    for strategy, regime in [("global_ridge", "standard"), ("global_mlp", "standard"), ("block_ridge", "linear")]:
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
            task=f"tiny_{strategy}",
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
