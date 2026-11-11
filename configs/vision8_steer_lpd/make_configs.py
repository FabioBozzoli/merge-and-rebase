"""Generate the vision8 linear probe distillation (LPD) runs: ``stage1_mode="logit_distill"``.

LPD is an opt-in BASELINE, not steer: Stage 1 is skipped and one ridge on B's final feature fits the
support-set logit residual, min_L ||H_A^ft T_A - (H_B T_B + H_B L)||^2 + lambda ||L||^2 (docs/rebasin.md).
The reference steer recipe stays configs/vision8_steer_trace (block_ridge, smoothed_residual, rho=0.9).

Same pairs, checkpoints, feature caches, few_shot {1,2,5,10,20} x seed {33,54,89} and alpha=1 as the trace
family; sweep over ``ridge_lambda`` (0 = minimum-norm least squares). ``correction_space="logit"`` (the
feature-space variant gives the same logit offset; see tests).

    python configs/vision8_steer_lpd/make_configs.py                        # ViT-B/16 -> ViT-B/16
    python configs/vision8_steer_lpd/make_configs.py --base configs/vision8_steer_linear_vitb_vitl.json \\
        --out-dir configs/vision8_steer_lpd_vitl --prefix vision8_steer_lpd_vitl --pair "ViT-B/16 -> ViT-L/14"
    # SVHN on ViT-L: --feature-cache-dir .../features/steer_svhn96_source_only (MR_SVHN_PREPROCESS_SCOPE=source)

LPD with all of B's blocks concatenated as input (instead of the final feature): add ``--stage2 concat_ridge`` and a distinct
``--out-dir/--prefix`` (e.g. configs/vision8_steer_lpdconcat, vision8_steer_lpdconcat_vitl, vision8_steer_lpdconcat_vitl_svhnsrc).

Note: no validation split selects lambda here; report every lambda (picking the best on test is optimistic).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_trace_cfg", HERE.parent / "vision8_steer_trace" / "make_configs.py")
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)

LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


def _lambda_tag(value: float) -> str:
    return "0" if value == 0 else f"{value:g}".replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=_trace.BASE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="vision8_steer_lpd")
    parser.add_argument("--pair", default="ViT-B/16 -> ViT-B/16")
    parser.add_argument("--ckpt-root", default=_trace.BASE_CKPT_ROOT)
    parser.add_argument("--feature-cache-dir", default="/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature")
    parser.add_argument("--correction-space", default="logit", choices=["logit", "feature"])
    parser.add_argument("--stage2", default="global_ridge", choices=["global_ridge", "concat_ridge"],
                        help="LPD input: global_ridge = B's final feature (default); concat_ridge = all of B's blocks concatenated.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lam in args.lambdas:
        for few_shot in _trace.FEW_SHOTS:
            for seed in _trace.SEEDS:
                cfg = _trace.build(base, few_shot, seed, args.ckpt_root, args.feature_cache_dir, args.pair)
                cfg["method_params"] = {
                    "feature_regime": "linear",
                    "stage1_mode": "logit_distill",
                    "stage_2_strategy": "global_ridge",
                    "correction_space": args.correction_space,
                    "ridge_lambda": lam,
                    "global_ridge_lambda_scaling": "none",
                    "few_shot": few_shot,
                    "seed": seed,
                    "feature_cache_dir": args.feature_cache_dir,
                    "force_recompute_features": False,
                }
                if args.stage2 == "concat_ridge":
                    # all of B's residual blocks + final feature, each scaled to unit mean squared norm (always on);
                    # no global_ridge_lambda_scaling (it is global_ridge-only).
                    cfg["method_params"].pop("global_ridge_lambda_scaling")
                    cfg["method_params"].update({"stage_2_strategy": "concat_ridge", "block_granularity": "residual", "token_pooling": "mean"})
                _inp = "concat_ridge over all of B's blocks" if args.stage2 == "concat_ridge" else "global_ridge on B's final feature"
                cfg["_description"] = (
                    f"[BASELINE, not steer] linear probe distillation (stage1_mode=logit_distill, {_inp}, on the logit "
                    f"residual, {args.correction_space} space, ridge_lambda={lam}), {args.pair}, few_shot={few_shot}, seed={seed}, alpha=1."
                )
                path = args.out_dir / f"{args.prefix}_lam{_lambda_tag(lam)}_fs{few_shot}_seed{seed}.json"
                path.write_text(json.dumps(cfg, indent=2) + "\n")
                count += 1
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
