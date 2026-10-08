"""Generate the t5-base (NTK) -> FacebookAI/roberta-base steer_text configs.

Every config transports the same six NLI tasks from the same NTK t5-base checkpoints as the t5-large runs,
with RoBERTa as the target (``model_kind="encoder_classification"``: masked mean of the last layer + one
linear head, see ``RobertaEncoderForSequenceClassification``). The source side of the feature cache (A's
jvp pass, target-independent) is reused from an existing target's cache through
``method_params.reuse_source_features_from`` -- only RoBERTa's own side is computed.

Variants (``--variants``):

- ``attn_segments``  block_ridge on the attention outputs, blocks pooled as [premise; hypothesis; global]
- ``attn_global``    block_ridge on the attention outputs, blocks pooled with a single global mean
- ``resid_segments`` / ``resid_global``   the same on the end-of-block (residual) features
- ``global_ridge`` / ``global_mlp``       Stage 2 on the last layer's pooled output only

block_ridge variants use trace-scaled lambda_2, ``smoothed_residual`` with rho=0.9 and ``ridge_lambda=0.1``; all
use a fixed alpha=1, ``feature_regime="linear"`` (the NTK checkpoints' delta must be linearized, also for the
global modes). RoBERTa and T5-base both have 12 blocks, so there is no block grouping.

    python configs/block_ridge_ntk_roberta/make_configs.py [--out-dir DIR] [--variants attn_segments attn_global]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASKS = ("mnli", "qnli", "rte", "scitail", "snli", "sick")
SICK_MAX_FEW_SHOT = 500  # only 606 class-2 rows in the 4000-row train pool: more shots per class is infeasible

_BLOCK = {
    "stage_2_strategy": "block_ridge",
    "block_ridge_lambda_scaling": "trace",
    "block_ridge_mode": "smoothed_residual",
    "rho": 0.9,
    "ridge_lambda": 0.1,
    "block_group_strategy": "concat",
}
VARIANTS: dict[str, dict] = {
    "attn_segments": {**_BLOCK, "block_source": "attention", "target_block_pooling": "segments"},
    "attn_global": {**_BLOCK, "block_source": "attention", "target_block_pooling": "global"},
    "resid_segments": {**_BLOCK, "block_source": "residual", "target_block_pooling": "segments"},
    "resid_global": {**_BLOCK, "block_source": "residual", "target_block_pooling": "global"},
    "global_ridge": {"stage_2_strategy": "global_ridge", "ridge_lambda": 1.0},
    "global_mlp": {"stage_2_strategy": "global_mlp", "mlp_epochs": 100, "mlp_hidden_dim": 1024},
}


def build(task: str, variant: str, few_shot: int, seed: int, args: argparse.Namespace) -> dict:
    heads = Path(args.heads_dir) / f"roberta-base_{task}_nearest_mean_seed33_fewshot300.pt"
    return {
        "suite": "nli6",
        "source_model_name_or_path": "google-t5/t5-base",
        "source_model_arch": "t5",
        "target_model_name_or_path": "FacebookAI/roberta-base",
        "target_model_arch": "auto",
        "model_kind": "encoder_classification",
        "num_labels": 3,
        "device": "cuda",
        "dtype": "fp32",
        "eval_mode": "head_logits",
        "head_key_pattern": "classification_head",
        "split": "test",
        "val_fraction": 0.1,
        "max_samples_per_task": 2000,
        "max_train_samples": 4000,
        "batch_size": 16,
        "num_workers": 0,
        "max_length": 256,
        "method": "steer_text",
        "alpha_search": False,
        "alpha": 1.0,
        "alpha_selection": "shared",
        "weights": None,
        "strict_load": False,
        "eval_source_finetuned": False,
        "save_steer_artifacts_dir": None,
        "_description": f"steer_text {variant}, t5-base (NTK) -> roberta-base, {task}, few_shot={few_shot}, seed={seed}.",
        "tasks": task,
        "seed": seed,
        "tuned_ckpts": {task: f"{args.ckpt_root}/{task}"},
        "target_task_heads": str(heads),
        "feature_cache_dir": str(args.cache_dir),
        "method_params": {
            "feature_regime": "linear",
            "stage1_lambda": 1.0,
            "force_recompute_features": False,
            "few_shot": few_shot,
            "reuse_source_features_from": args.reuse_from,
            **VARIANTS[variant],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--variants", nargs="+", default=["attn_segments", "attn_global"], choices=sorted(VARIANTS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--few-shots", nargs="+", type=int, default=[200, 300, 500, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 54, 89])
    p.add_argument("--ckpt-root", default="/home/fbozzoli/checkpoints_t5_converted/t5base_6text_noreg_ntk")
    p.add_argument("--heads-dir", default=str(REPO / "results/ntk_segment_pooling/heads_roberta"))
    p.add_argument("--cache-dir", default=str(REPO / "results/ntk_segment_pooling/feature_cache"))
    p.add_argument("--reuse-from", default="google-t5__t5-large__encoder_classification",
                   help="Target tag of an existing cache pair (same source) whose A side is reused; '' disables.")
    args = p.parse_args()
    args.reuse_from = args.reuse_from or None

    n = 0
    for variant in args.variants:
        for task in args.tasks:
            for few_shot in args.few_shots:
                if task == "sick" and few_shot > SICK_MAX_FEW_SHOT:
                    continue
                for seed in args.seeds:
                    cfg = build(task, variant, few_shot, seed, args)
                    if cfg["method_params"]["reuse_source_features_from"] is None:
                        del cfg["method_params"]["reuse_source_features_from"]
                    path = args.out_dir / task / f"{task}_roberta_{variant}_fs{few_shot}_seed{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(cfg, indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
