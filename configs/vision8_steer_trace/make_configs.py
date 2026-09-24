"""Generate the ViT-B/16 -> ViT-B/16 steer runs: block_ridge with trace-scaled lambda.

Recipe (per run): end-of-block (residual) features, ``block_ridge_lambda_scaling="trace"``,
``block_ridge_mode="smoothed_residual"`` with rho=0.9, ``ridge_lambda``=0.1, fixed alpha=1.0, the 8
vision tasks. Sweep: few_shot in {1, 2, 5, 10, 20} x seed in {33, 54, 89} -> 15 configs.

    python configs/vision8_steer_trace/make_configs.py [--ckpt-root DIR] [--feature-cache-dir DIR]

    # ViT-B/16 -> ViT-L/14 (12 -> 24 blocks: the target's blocks are grouped down with `concat`)
    python configs/vision8_steer_trace/make_configs.py --base configs/vision8_steer_linear_vitb_vitl.json \\
        --out-dir configs/vision8_steer_trace_vitl --prefix vision8_steer_trace_vitl --pair "ViT-B/16 -> ViT-L/14"

The feature cache is keyed by source/target/task/regime/split (not few_shot or seed), so one shared
``--feature-cache-dir`` means the features are computed once per task and every other run only refits.
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
    cfg["_description"] = f"steer block_ridge trace-lambda, {pair}, few_shot={few_shot}, seed={seed}."
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="vision8_steer_trace", help="Config file name prefix.")
    parser.add_argument("--pair", default="ViT-B/16 -> ViT-B/16", help="Label used in the config description.")
    parser.add_argument("--ckpt-root", default=BASE_CKPT_ROOT, help="Replaces the checkpoint root in tuned_ckpts.")
    parser.add_argument("--feature-cache-dir", default="/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature_trace")
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
