"""Block-pairing shuffle ablation for steer_text: t5-base -> {t5-large, roberta-base}.

Same cell as ``configs/ntk_rebasin_study`` ``lin_block_ridge_attn_seg`` (ntk checkpoints, linear regime,
stage1_lambda=1, block_ridge on B's attention outputs pooled per segment, smoothed_residual rho=0.9,
trace-scaled ridge_lambda=0.1, concat grouping, alpha=1, same heads, same feature cache), plus
``block_pairing``: ``reversed`` or ``random`` (permutation drawn with ``block_pairing_seed``, independent of
the support ``seed``). Input block b regresses target perm[b]; the fit is otherwise unchanged.

    python configs/ntk_pairing/make_configs.py
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

CELL = "lin_block_ridge_attn_seg"
PERM_SEEDS = (0, 1, 2, 3, 4)
PAIRINGS = [("identity", None), ("reversed", None)] + [("random", s) for s in PERM_SEEDS]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--mode", default="smoothed_residual", choices=["smoothed_residual", "independent"],
                   help="block_ridge_mode; independent = no carry (the rho=0 limit).")
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--skip-identity", action="store_true", help="Do not write the identity-pairing configs.")
    args = p.parse_args()
    n = 0
    for target in _study.TARGETS:
        for task in _study.TASKS:
            for few_shot in _study.FEW_SHOTS:
                for seed in _study.SEEDS:
                    for pairing, perm_seed in PAIRINGS:
                        if args.skip_identity and pairing == "identity":
                            continue
                        tag = pairing if perm_seed is None else f"{pairing}{perm_seed}"
                        cfg = _study.build(target, CELL, task, few_shot, seed)
                        cfg["method_params"]["block_ridge_mode"] = args.mode
                        cfg["method_params"]["rho"] = args.rho
                        cfg["method_params"]["block_pairing"] = pairing
                        if perm_seed is not None:
                            cfg["method_params"]["block_pairing_seed"] = perm_seed
                        cfg["_description"] = cfg["_description"].replace(CELL, f"{CELL} mode={args.mode} rho={args.rho} block_pairing={tag}")
                        path = args.out_dir / target / task / f"{CELL}_{tag}_fs{few_shot}_seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(cfg, indent=2) + "\n")
                        n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
