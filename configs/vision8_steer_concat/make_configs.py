"""Generate the steer concat_ridge grid (one ridge on the concatenation of all of B's blocks).

Same grid, bases, checkpoints and feature caches as configs/vision8_steer_global (global_ridge), so the two
are directly comparable: concat_ridge's input [h_1/s_1, ..., h_L/s_L, f_B/s_{L+1}] contains global_ridge's
input f_B, with the same Stage 1 target.
  linear    = NTK checkpoints, feature_regime="linear",   cache features/linear_feature (already has B's blocks)
  nonlinear = full-FT checkpoints, feature_regime="standard", cache features/nonlinear_feature for A and f_B;
              B's block activations are read from features/linear_feature (target_blocks_cache_dir): B is the
              same pretrained model in both regimes, and the run checks labels and f_B match before using them.
  Nothing is extracted or written: every run only reads the existing caches.
  block_granularity=residual, token_pooling=mean, per-block trace input scaling (always on),
  ridge_lambda fixed (default 1.0; --ridge-lambda for a sweep), stage1_lambda=1.0, alpha=1.0 fixed.

    python configs/vision8_steer_concat/make_configs.py [--features linear nonlinear] [--few-shots 1 2 5 10 20]
        [--ridge-lambda 1.0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("global_configs", HERE.parent / "vision8_steer_global" / "make_configs.py")
_global = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_global)
BASES, FEATURES, KEEP, SEEDS = _global.BASES, _global.FEATURES, _global.KEEP, _global.SEEDS
FEW_SHOTS = (1, 2, 5, 10, 20)


def build(base: dict, features: str, few_shot: int, seed: int, ridge_lambda: float) -> dict:
    regime, cache_dir = FEATURES[features]
    cfg = {k: base[k] for k in KEEP}
    cfg.update({"tasks": "all", "method": "steer", "alpha_search": False, "alpha_selection": "shared",
                "alpha": 1.0, "seed": seed})
    cfg["method_params"] = {
        "feature_regime": regime,
        "stage_2_strategy": "concat_ridge",
        "block_granularity": "residual",
        "token_pooling": "mean",
        "ridge_lambda": ridge_lambda,
        "stage1_lambda": 1.0,
        "few_shot": few_shot,
        "seed": seed,
        "feature_cache_dir": cache_dir,
        "force_recompute_features": False,
    }
    if regime == "standard":
        cfg["method_params"]["target_blocks_cache_dir"] = FEATURES["linear"][1]
        cfg["method_params"]["target_blocks_regime"] = "linear"
    target = f"{base['target_clip_model']} {base['target_clip_pretrained']}"
    cfg["_description"] = (f"steer concat_ridge (all residual blocks + final feature, per-block trace scaling, "
                           f"ridge_lambda={ridge_lambda}), {features} features ({regime} regime), "
                           f"ViT-B-16 datacomp -> {target}, few_shot={few_shot}, seed={seed}, alpha=1 fixed.")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", default=list(FEATURES), choices=list(FEATURES))
    p.add_argument("--few-shots", nargs="+", type=int, default=list(FEW_SHOTS))
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    args = p.parse_args()
    suffix = "" if args.ridge_lambda == 1.0 else f"_lam{args.ridge_lambda:g}"
    n = 0
    for features in args.features:
        for pair in ("vitb", "vitl"):
            base = json.loads(BASES[(pair, features)].read_text())
            for few_shot in args.few_shots:
                for seed in SEEDS:
                    path = HERE / f"concatridge_{features}_{pair}_fs{few_shot}_seed{seed}{suffix}.json"
                    path.write_text(json.dumps(build(base, features, few_shot, seed, args.ridge_lambda), indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} configs to {HERE}")


if __name__ == "__main__":
    main()
