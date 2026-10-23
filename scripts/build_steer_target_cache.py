"""Build a steer linear-feature cache whose target (B) side uses a different preprocessing.

A's side (features_A, delta_A, delta_A_blocks, y_A and the train/ heads) is copied unchanged
from an existing cache; B's side (features_B, features_B_blocks) is recomputed with B's stock
preprocessing, i.e. *without* the custom SVHN pipeline. Use with MR_SVHN_PREPROCESS_SCOPE=source:
A sees resize-96 + pad (as its checkpoint was trained), B sees its native transform.

Only B's forward passes run (no JVPs). Rows are paired with the source cache through the same
deterministic subset/order steer uses (_aligned_loader_pair), and the labels are checked against
the copied y_A.

    python scripts/build_steer_target_cache.py \\
        --src-root /work/.../features/linear_feature --out-root /work/.../features/steer_svhn96_source_only \\
        --target-model ViT-B-16 --target-pretrained laion2b_s34b_b88k
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.rebase.methods.steer import (
    _TRAIN_FEATURES_PER_CLASS,
    _aligned_loader_pair,
    _collect_target_blocks_split,
    _save_split_cache,
)

SOURCE_FILES = ("features_A", "delta_A", "delta_A_blocks", "y_A")
_VAL_TEST_SPLIT_SEED = 0  # vision_rebase.py's fixed val/test carve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, help="Existing feature_cache_dir (A side is copied from it).")
    parser.add_argument("--out-root", required=True, help="New feature_cache_dir to write.")
    parser.add_argument("--pair-dir", default="ViT-B-16_datacomp_xl_s13b_b90k_to_ViT-B-16_laion2b_s34b_b88k")
    parser.add_argument("--target-model", default="ViT-B-16")
    parser.add_argument("--target-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--task", default="SVHN")
    parser.add_argument("--regime-dir", default="linear", help="Cache leaf for regime/granularity/pooling.")
    parser.add_argument("--block-granularity", default="residual")
    parser.add_argument("--token-pooling", default="mean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    src = Path(args.src_root) / args.pair_dir / args.task / args.regime_dir
    out = Path(args.out_root) / args.pair_dir / args.task / args.regime_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clf_target = OpenClipClassifier.build(
        OpenClipBuildConfig(model_name=args.target_model, pretrained=args.target_pretrained, device=str(device))
    )
    visual = clf_target.model.visual.to(device).eval()
    hf_path, hf_config, split_map = SUITES["vision8"].resolver(args.task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=clf_target.preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        val_fraction=0.1,
        seed=_VAL_TEST_SPLIT_SEED,
        svhn_custom_preprocess=False,
    )

    per_class = _TRAIN_FEATURES_PER_CLASS.get(args.task)
    train_split = "train" if per_class is None else f"train_{per_class}perclass"
    (out / "train").mkdir(parents=True, exist_ok=True)
    for head in ("head_A.pt", "head_B.pt"):
        shutil.copy2(src / "train" / head, out / "train" / head)

    for split, cache_split in (("train", train_split), ("test", "test")):
        loader = getattr(loaders, split)
        _, target_loader = _aligned_loader_pair(loader, loader, per_class=per_class if split == "train" else None)
        data = _collect_target_blocks_split(
            target_visual=visual,
            target_loader=target_loader,
            device=device,
            block_granularity=args.block_granularity,
            token_pooling=args.token_pooling,
        )
        y_src = torch.load(src / cache_split / "y_A.pt", map_location="cpu", weights_only=True)
        if not torch.equal(data["y"].long(), y_src.long()):
            raise RuntimeError(f"{cache_split}: labels differ from the source cache -- rows are not paired.")
        (out / cache_split).mkdir(parents=True, exist_ok=True)
        for name in SOURCE_FILES:
            shutil.copy2(src / cache_split / f"{name}.pt", out / cache_split / f"{name}.pt")
        _save_split_cache(out / cache_split, {"features_B": data["features_B"], "features_B_blocks": data["features_B_blocks"]})
        head_b = torch.load(out / "train" / "head_B.pt", map_location="cpu", weights_only=True)
        head_b = head_b["weight"] if isinstance(head_b, dict) else head_b
        acc = (data["features_B"].double() @ head_b.double().T).argmax(1).eq(data["y"]).double().mean()
        print(f"{cache_split}: {len(data['y'])} rows, B zero-shot (stock preprocessing) = {float(acc):.4f} -> {out / cache_split}")


if __name__ == "__main__":
    main()
