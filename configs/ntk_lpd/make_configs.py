"""Linear probe distillation (LPD, ``stage1_mode="logit_distill"``) for steer_text: t5-base -> {t5-large, roberta-base}.

LPD is an opt-in BASELINE, not steer: Stage 1 is skipped and one ridge on B's pooled feature fits the support-set
logit residual (docs/text-rebasin.md). Same setup as ``configs/ntk_rebasin_study`` cell ``lin_global_ridge``
(ntk checkpoints, linear regime, same heads and feature cache, few_shot {200,300,500} x seed {33,54,89}, alpha=1),
with trace-scaled (dimensionless) ``ridge_lambda`` swept over LAMBDAS (0 = minimum-norm least squares).
The reference steer cell stays ``lin_block_ridge_attn_seg`` (smoothed_residual, rho=0.9).

    python configs/ntk_lpd/make_configs.py [--input-pooling global|segments]

Note: no validation split selects lambda; report every lambda (picking the best on test is optimistic).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_ntk_study", HERE.parent / "ntk_rebasin_study" / "make_configs.py")
_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_study)

LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


def _lambda_tag(value: float) -> str:
    return "0" if value == 0 else f"{value:g}".replace("-", "m")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--input-pooling", default="global", choices=["global", "segments"])
    p.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    p.add_argument("--stage2", default="global_ridge", choices=["global_ridge", "joint_ridge"],
                   help="LPD input: global_ridge = B's pooled feature (default); joint_ridge = all of B's blocks concatenated "
                        "(attention outputs, segment-pooled [premise; hypothesis; global], as the reference steer cell).")
    p.add_argument("--targets", nargs="+", default=None, help="Restrict the targets (default: all).")
    args = p.parse_args()
    base_cell = "lin_global_ridge" if args.input_pooling == "global" else "lin_global_ridge_seg"
    stem = "lpd" if args.input_pooling == "global" else "lpd_seg"
    if args.stage2 == "joint_ridge":
        if args.input_pooling != "global":
            raise SystemExit("--stage2 joint_ridge reads B's blocks; --input-pooling only applies to global_ridge")
        stem = "lpd_joint_seg"
    n = 0
    for target in (args.targets or _study.TARGETS):
        for task in _study.TASKS:
            for few_shot in _study.FEW_SHOTS:
                for seed in _study.SEEDS:
                    for lam in args.lambdas:
                        cfg = _study.build(target, base_cell, task, few_shot, seed)
                        mp = cfg["method_params"]
                        mp.pop("stage1_lambda", None)  # ignored by LPD
                        mp.update({
                            "stage1_mode": "logit_distill",
                            "stage_2_strategy": "global_ridge",
                            "ridge_lambda": lam,
                            "global_ridge_lambda_scaling": "trace",
                        })
                        cfg["_description"] = (
                            f"[BASELINE, not steer] linear probe distillation (stage1_mode=logit_distill, global_ridge, "
                            f"input={args.input_pooling}, trace-scaled ridge_lambda={lam}): t5-base -> {target}, {task}, "
                            f"few_shot={few_shot}, seed={seed}."
                        )
                        if args.stage2 == "joint_ridge":
                            # same block inputs as the reference steer cell lin_block_ridge_attn_seg; one ridge on their
                            # concatenation (each block scaled to unit mean squared norm), no global_ridge-only lambda scaling.
                            mp.pop("global_ridge_lambda_scaling")
                            mp.update({
                                "stage_2_strategy": "joint_ridge",
                                "block_source": "attention",
                                "target_block_pooling": "segments",
                                "block_pooling": "mean",
                                "block_group_strategy": "concat",
                            })
                            cfg["_description"] = (
                                f"[BASELINE, not steer] linear probe distillation (stage1_mode=logit_distill, joint_ridge over all of "
                                f"B's attention blocks, segment-pooled, + pooled output; ridge_lambda={lam}): t5-base -> {target}, "
                                f"{task}, few_shot={few_shot}, seed={seed}."
                            )
                        path = args.out_dir / target / task / f"{stem}_lam{_lambda_tag(lam)}_fs{few_shot}_seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(cfg, indent=2) + "\n")
                        n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
