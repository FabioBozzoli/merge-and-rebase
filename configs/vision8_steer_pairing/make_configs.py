"""Generate the block-pairing shuffle ablation of the vision8 steer block_ridge runs.

Same recipe as ``configs/vision8_steer_trace`` (residual blocks, smoothed_residual, rho=0.9,
trace-scaled ridge_lambda=0.1, alpha=1.0), but each B input block regresses a re-paired A
target: ``block_pairing="reversed"`` (block b <- target N-1-b) or ``"random"`` (uniform
permutation drawn with ``block_pairing_seed``, independent of the support-set ``seed``).
The fit itself (carry order, residual state, rho) is unchanged.

Sweep: (reversed + random x PERM_SEEDS) x few_shot x seed.

    python configs/vision8_steer_pairing/make_configs.py [--ckpt-root DIR] [--feature-cache-dir DIR]

Features are read from the existing linear cache (the pairing acts after loading), so no recompute.
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

PERM_SEEDS = (0, 1, 2, 3, 4)
PAIRINGS = [("identity", None), ("reversed", None)] + [("random", s) for s in PERM_SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=_trace.BASE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="vision8_steer_pairing")
    parser.add_argument("--pair", default="ViT-B/16 -> ViT-B/16")
    parser.add_argument("--ckpt-root", default=_trace.BASE_CKPT_ROOT)
    parser.add_argument("--feature-cache-dir", default="/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature")
    parser.add_argument("--mode", default="smoothed_residual", choices=["smoothed_residual", "independent"],
                        help="block_ridge_mode; independent = no carry (the rho=0 limit).")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--skip-identity", action="store_true", help="Do not write the identity-pairing configs.")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pairing, perm_seed in PAIRINGS:
        if args.skip_identity and pairing == "identity":
            continue
        tag = pairing if perm_seed is None else f"{pairing}{perm_seed}"
        for few_shot in _trace.FEW_SHOTS:
            for seed in _trace.SEEDS:
                cfg = _trace.build(base, few_shot, seed, args.ckpt_root, args.feature_cache_dir, args.pair)
                cfg["method_params"]["block_ridge_mode"] = args.mode
                cfg["method_params"]["rho"] = args.rho
                cfg["method_params"]["block_pairing"] = pairing
                if perm_seed is not None:
                    cfg["method_params"]["block_pairing_seed"] = perm_seed
                cfg["_description"] = (
                    f"steer block_ridge trace-lambda, mode={args.mode} rho={args.rho}, block_pairing={tag}, {args.pair}, few_shot={few_shot}, seed={seed}."
                )
                path = args.out_dir / f"{args.prefix}_{tag}_fs{few_shot}_seed{seed}.json"
                path.write_text(json.dumps(cfg, indent=2) + "\n")
                count += 1
    print(f"Wrote {count} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
