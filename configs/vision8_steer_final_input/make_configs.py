"""Generate the ViT-B/16 -> ViT-B/16 steer runs: block_ridge whose every ridge reads B's final feature.

Ablation of ``configs/vision8_steer_trace``: same recipe and sweep, but ``block_ridge_input="final"``
makes every block ridge take the target encoder's final (L2-normalized) feature as input, while
still predicting the per-block targets (A's per-block delta mapped into B's space). With
``mode="independent"`` this reduces exactly to global_ridge; the runs use ``smoothed_residual``,
and prepare() logs the independent-equivalent accuracy next to the stage 2 one.
Since the input rows are unit-norm, ``trace`` scaling leaves lambda unchanged here.

Recipe (per run): ``block_ridge_input="final"``, ``block_ridge_lambda_scaling="trace"``,
``block_ridge_mode="smoothed_residual"`` with rho=0.9, ``ridge_lambda``=0.1, fixed alpha=1.0, the 8
vision tasks. Sweep: few_shot in {1, 2, 5, 10, 20} x seed in {33, 54, 89} -> 15 configs.

    python configs/vision8_steer_final_input/make_configs.py [--ckpt-root DIR] [--feature-cache-dir DIR]

    # ViT-B/16 -> ViT-L/14 (no block grouping needed: only the source defines the blocks)
    python configs/vision8_steer_final_input/make_configs.py --base configs/vision8_steer_linear_vitb_vitl.json \\
        --out-dir configs/vision8_steer_final_input_vitl --prefix vision8_steer_final_input_vitl \\
        --pair "ViT-B/16 -> ViT-L/14"

The feature cache is keyed by source/target/task/regime/split (not few_shot or seed), so one shared
``--feature-cache-dir`` means the features are computed once per task and every other run only refits.
The default points at the cache the linear-regime steer runs already filled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "vision8_steer_linear.json"
FEW_SHOTS = (1, 2, 5, 10, 20)
SEEDS = (33, 54, 89)
BASE_CKPT_ROOT = "/work/intesasanpaolo_phd/merge-and-rebase/checkpoints"


def build(base: dict, few_shot: int, seed: int, ckpt_root: str, cache_dir: str, pair: str) -> dict:
    cfg = json.loads(json.dumps(base))
    for key in ("alpha_min", "alpha_max", "alpha_step", "alpha_patience"):
        cfg.pop(key, None)
    cfg.update({"alpha_search": False, "alpha": 1.0, "alpha_selection": "shared", "seed": seed})
    cfg["tuned_ckpts"] = {t: p.replace(BASE_CKPT_ROOT, ckpt_root, 1) for t, p in base["tuned_ckpts"].items()}
    cfg["method_params"] = {
        "feature_regime": "linear",
        "stage_2_strategy": "block_ridge",
        "block_granularity": "residual",
        "block_ridge_input": "final",
        "block_ridge_mode": "smoothed_residual",
        "rho": 0.9,
        "block_ridge_lambda_scaling": "trace",
        "ridge_lambda": 0.1,
        "stage1_lambda": 1.0,
        "few_shot": few_shot,
        "seed": seed,
        "feature_cache_dir": cache_dir,
        "force_recompute_features": False,
    }
    cfg["_description"] = f"steer block_ridge final-feature input, trace-lambda, {pair}, few_shot={few_shot}, seed={seed}."
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="vision8_steer_final_input", help="Config file name prefix.")
    parser.add_argument("--pair", default="ViT-B/16 -> ViT-B/16", help="Label used in the config description.")
    parser.add_argument("--ckpt-root", default=BASE_CKPT_ROOT, help="Replaces the checkpoint root in tuned_ckpts.")
    parser.add_argument("--feature-cache-dir", default="/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for few_shot in FEW_SHOTS:
        for seed in SEEDS:
            cfg = build(base, few_shot, seed, args.ckpt_root, args.feature_cache_dir, args.pair)
            path = args.out_dir / f"{args.prefix}_fs{few_shot}_seed{seed}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Wrote {len(FEW_SHOTS) * len(SEEDS)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
