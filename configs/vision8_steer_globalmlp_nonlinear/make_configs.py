"""Generate the steer global_mlp runs on non-linear (full fine-tuning) source checkpoints.

Recipe (per run): feature_regime="standard" (true delta f(theta_ft) - f(theta_0), checkpoints_nonlinear/),
stage_2_strategy="global_mlp" (hidden 1024, 100 epochs) on B's final feature, stage1_lambda=1.0,
fixed alpha=1.0. Cache: features/nonlinear_feature. Sweep: few_shot in {1, 10} x seed in {33, 54, 89}
for both ViT-B/16 -> ViT-B/16 laion2b (vitb) and ViT-B/16 -> ViT-L/14 datacomp (vitl).

    python configs/vision8_steer_globalmlp_nonlinear/make_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Only the model pair and tuned_ckpts (checkpoints_nonlinear/) are taken from these bases.
BASES = {
    "vitb": HERE.parent / "vision8_steer_svhn96_noaug_nonlinear.json",
    "vitl": HERE.parent / "vision8_steer_svhn96_noaug_vitl_nonlinear.json",
}
FEW_SHOTS = (1, 10)
SEEDS = (33, 54, 89)
CACHE_DIR = "/work/intesasanpaolo_phd/merge-and-rebase/features/nonlinear_feature"


def build(base: dict, pair: str, few_shot: int, seed: int) -> dict:
    keep = ("suite", "source_clip_model", "source_clip_pretrained", "target_clip_model", "target_clip_pretrained",
            "batch_size", "num_workers", "val_fraction", "no_humanize", "weights", "tuned_ckpts")
    cfg = {k: base[k] for k in keep}
    cfg.update({"tasks": "all", "method": "steer", "alpha_search": False, "alpha_selection": "shared",
                "alpha": 1.0, "seed": seed})
    cfg["method_params"] = {
        "feature_regime": "standard",
        "stage_2_strategy": "global_mlp",
        "mlp_hidden_dim": 1024,
        "mlp_epochs": 100,
        "stage1_lambda": 1.0,
        "few_shot": few_shot,
        "seed": seed,
        "feature_cache_dir": CACHE_DIR,
        "force_recompute_features": False,
    }
    target = f"{base['target_clip_model']} {base['target_clip_pretrained']}"
    cfg["_description"] = (f"steer global_mlp, non-linear ckpts (standard regime), ViT-B-16 datacomp -> {target}, "
                           f"few_shot={few_shot}, seed={seed}, alpha=1 fixed.")
    return cfg


def main() -> None:
    n = 0
    for pair, base_path in BASES.items():
        base = json.loads(base_path.read_text())
        for few_shot in FEW_SHOTS:
            for seed in SEEDS:
                path = HERE / f"globalmlp_nonlinear_{pair}_fs{few_shot}_seed{seed}.json"
                path.write_text(json.dumps(build(base, pair, few_shot, seed), indent=2) + "\n")
                n += 1
    print(f"Wrote {n} configs to {HERE}")


if __name__ == "__main__":
    main()
