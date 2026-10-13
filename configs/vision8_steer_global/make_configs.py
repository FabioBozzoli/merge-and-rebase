"""Generate the steer global-Stage-2 variants (global_mlp / global_ridge x linear / non-linear features).

Same grid as configs/vision8_steer_globalmlp_nonlinear (global_mlp on non-linear features, already run):
few_shot in {1, 10} x seed in {33, 54, 89} x pair {vitb: ViT-B/16 -> ViT-B/16 laion2b, vitl: -> ViT-L/14 datacomp},
stage1_lambda=1.0, fixed alpha=1.0.
  linear    = NTK checkpoints (checkpoints/...), feature_regime="linear", cache features/linear_feature
  nonlinear = full fine-tuning checkpoints (checkpoints_nonlinear/...), feature_regime="standard",
              cache features/nonlinear_feature
  global_mlp:   hidden 1024, 100 epochs
  global_ridge: ridge_lambda=1.0 fixed (trace scaling exists only for block_ridge)

    python configs/vision8_steer_global/make_configs.py [--variants globalridge_nonlinear] [--few-shots 1 2 5 10 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE.parent
# Only the model pair and tuned_ckpts are taken from these bases.
BASES = {
    ("vitb", "linear"): CFG / "vision8_steer_trace" / "vision8_steer_trace_fs10_seed33.json",
    ("vitl", "linear"): CFG / "vision8_steer_trace_vitl" / "vision8_steer_trace_vitl_fs10_seed33.json",
    ("vitb", "nonlinear"): CFG / "vision8_steer_svhn96_noaug_nonlinear.json",
    ("vitl", "nonlinear"): CFG / "vision8_steer_svhn96_noaug_vitl_nonlinear.json",
}
FEATURES = {
    "linear": ("linear", "/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature"),
    "nonlinear": ("standard", "/work/intesasanpaolo_phd/merge-and-rebase/features/nonlinear_feature"),
}
STRATEGIES = {
    "globalmlp": {"stage_2_strategy": "global_mlp", "mlp_hidden_dim": 1024, "mlp_epochs": 100},
    "globalridge": {"stage_2_strategy": "global_ridge", "ridge_lambda": 1.0},
}
# global_mlp on non-linear features already lives in configs/vision8_steer_globalmlp_nonlinear.
VARIANTS = (("globalmlp", "linear"), ("globalridge", "nonlinear"), ("globalridge", "linear"))
FEW_SHOTS = (1, 10)
SEEDS = (33, 54, 89)
KEEP = ("suite", "source_clip_model", "source_clip_pretrained", "target_clip_model", "target_clip_pretrained",
        "batch_size", "num_workers", "val_fraction", "no_humanize", "weights", "tuned_ckpts")


def build(base: dict, strategy: str, features: str, few_shot: int, seed: int) -> dict:
    regime, cache_dir = FEATURES[features]
    cfg = {k: base[k] for k in KEEP}
    cfg.update({"tasks": "all", "method": "steer", "alpha_search": False, "alpha_selection": "shared",
                "alpha": 1.0, "seed": seed})
    cfg["method_params"] = {
        "feature_regime": regime,
        **STRATEGIES[strategy],
        "stage1_lambda": 1.0,
        "few_shot": few_shot,
        "seed": seed,
        "feature_cache_dir": cache_dir,
        "force_recompute_features": False,
    }
    target = f"{base['target_clip_model']} {base['target_clip_pretrained']}"
    cfg["_description"] = (f"steer {STRATEGIES[strategy]['stage_2_strategy']}, {features} features ({regime} regime), "
                           f"ViT-B-16 datacomp -> {target}, few_shot={few_shot}, seed={seed}, alpha=1 fixed.")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=[f"{s}_{f}" for s, f in VARIANTS],
                   choices=[f"{s}_{f}" for s in STRATEGIES for f in FEATURES])
    p.add_argument("--few-shots", nargs="+", type=int, default=list(FEW_SHOTS))
    args = p.parse_args()
    n = 0
    for strategy, features in (v.split("_") for v in args.variants):
        for pair in ("vitb", "vitl"):
            base = json.loads(BASES[(pair, features)].read_text())
            for few_shot in args.few_shots:
                for seed in SEEDS:
                    path = HERE / f"{strategy}_{features}_{pair}_fs{few_shot}_seed{seed}.json"
                    path.write_text(json.dumps(build(base, strategy, features, few_shot, seed), indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} configs to {HERE}")


if __name__ == "__main__":
    main()
