"""SVHN-only variant of configs/vision8_steer_concat for ViT-B/16 datacomp -> ViT-L/14 datacomp, with
MR_SVHN_PREPROCESS_SCOPE=source: A sees resize-96 + pad (as its checkpoint was trained), B (ViT-L) its stock
preprocessing. Each config is the vitl concat_ridge config with only the caches swapped:
  linear    feature_cache_dir = features/steer_svhn96_source_only (A side copied, B side recomputed stock)
  nonlinear feature_cache_dir = features/steer_svhn96_source_only_nonlinear (symlinks: A side from
            nonlinear_feature, features_B from steer_svhn96_source_only); B's blocks from
            steer_svhn96_source_only (target_blocks_cache_dir, regime linear).
Run with MR_SVHN_PREPROCESS_SCOPE=source (scripts/slurm/submit_vision_concatridge.sh, env CONFIG_DIR).
--nonlinear-only <cache> <out dir>: nonlinear configs only, with feature_cache_dir=<cache> (e.g. the standard
cache written by svhn96_scope_vitl.sh globalridge_nonlinear_source under steer_svhn96_source_only); B's blocks
still come from steer_svhn96_source_only/.../linear.

    python configs/vision8_steer_concat_svhn_source/make_configs.py [--nonlinear-only <cache> <out dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "vision8_steer_concat"
FEATURES = "/work/intesasanpaolo_phd/merge-and-rebase/features"
CACHES = {"linear": f"{FEATURES}/steer_svhn96_source_only", "nonlinear": f"{FEATURES}/steer_svhn96_source_only_nonlinear"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nonlinear-only", nargs=2, metavar=("CACHE", "OUT_DIR"))
    args = p.parse_args()
    caches, out = CACHES, HERE
    if args.nonlinear_only:
        caches, out = {"nonlinear": args.nonlinear_only[0]}, Path(args.nonlinear_only[1])
        out.mkdir(parents=True, exist_ok=True)
    n = 0
    for features, cache in caches.items():
        for path in sorted(SRC.glob(f"concatridge_{features}_vitl_fs*_seed*.json")):
            if "_lam" in path.stem:
                continue
            cfg = json.loads(path.read_text())
            params = cfg["method_params"]
            params["feature_cache_dir"] = cache
            if features == "nonlinear":
                params["target_blocks_cache_dir"] = CACHES["linear"]
            cfg["tasks"] = "SVHN"
            cfg["_description"] += " SVHN only, MR_SVHN_PREPROCESS_SCOPE=source (ViT-L with stock preprocessing)."
            (out / path.name).write_text(json.dumps(cfg, indent=2) + "\n")
            n += 1
    print(f"Wrote {n} configs to {out}")


if __name__ == "__main__":
    main()
