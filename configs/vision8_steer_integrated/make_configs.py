"""Generate the ViT-B/16 -> ViT-B/16 steer runs with integrated per-block residuals (nonlinear source).

``feature_regime="integrated"`` (opt-in, docs/rebasin.md): the source A is a *nonlinearly* fine-tuned
checkpoint (checkpoints_nonlinear/), Stage 1 is fit on the exact two-forward residual and block l's
Stage 2 target is the path integral D_l = int_0^1 J f(theta0 + t dtheta) (M_l dtheta) dt, estimated by
the midpoint rule on ``integration_points`` = K nodes (K * 13 masked JVPs per batch for ViT-B/16).

Recipe: the reference steer recipe of configs/vision8_steer_trace (block_ridge, residual granularity,
smoothed_residual rho=0.9, trace-scaled ridge_lambda=0.1, stage1_lambda=1, alpha=1 fixed), plus
``train_features_per_class=50`` on every task (the integration is the expensive part and the support
set never exceeds 20 shots per class). Sweep: few_shot {1,2,5,10,20} x seed {33,54,89} per K.

    python configs/vision8_steer_integrated/make_configs.py            # K=4
    python configs/vision8_steer_integrated/make_configs.py --k 4 8 16 # convergence study

The feature cache is keyed by pair/task/``integrated_k<K>``/split (not few_shot or seed): one extraction
per (task, K), every other run only refits. SVHN: resize-96 + pad on A and B, i.e. the runs must export
MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none MR_SVHN_PREPROCESS_SCOPE=both
(scripts/slurm/steer_integrated_vision.sh does).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "vision8_steer_nonlinear.json"
FEW_SHOTS = (1, 2, 5, 10, 20)
SEEDS = (33, 54, 89)
DEFAULT_CACHE = "/work/fbozzoli_phd/features/integrated_feature"


def build(base: dict, k: int, few_shot: int, seed: int, cache_dir: str, per_class: int | None, chunk: int | None = None) -> dict:
    cfg = json.loads(json.dumps(base))
    for key in ("alpha_min", "alpha_max", "alpha_step", "alpha_patience"):
        cfg.pop(key, None)
    cfg.update({"alpha_search": False, "alpha": 1.0, "alpha_selection": "shared", "seed": seed})
    cfg["method_params"] = {
        "feature_regime": "integrated",
        "integration_points": k,
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
    if per_class is not None:
        cfg["method_params"]["train_features_per_class"] = per_class
    if chunk is not None:
        # JVP chunk size: lower peak GPU memory during the extraction, same integrals up to floating point.
        cfg["method_params"]["integration_batch_size"] = chunk
    cfg["_description"] = (
        f"steer block_ridge trace-lambda with integrated per-block residuals (K={k} midpoint nodes), nonlinear "
        f"source checkpoints, ViT-B/16 datacomp -> ViT-B/16 laion2b, few_shot={few_shot}, seed={seed}, alpha=1 fixed."
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="vision8_steer_integrated")
    parser.add_argument("--k", type=int, nargs="+", default=[4], help="Numbers of midpoint quadrature nodes.")
    parser.add_argument("--feature-cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--train-features-per-class", type=int, default=50, help="0 = full train split.")
    parser.add_argument("--integration-batch-size", type=int, default=0, help="JVP chunk size (0 = whole loader batch).")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_class = args.train_features_per_class or None
    count = 0
    for k in args.k:
        for few_shot in FEW_SHOTS:
            for seed in SEEDS:
                cfg = build(base, k, few_shot, seed, args.feature_cache_dir, per_class, args.integration_batch_size or None)
                path = args.out_dir / f"{args.prefix}_k{k}_fs{few_shot}_seed{seed}.json"
                path.write_text(json.dumps(cfg, indent=2) + "\n")
                count += 1
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
