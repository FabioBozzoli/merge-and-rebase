"""Generate the NTK-vs-nonlinear steer_text study configs: t5-base -> {t5-large, roberta-base}.

Question: does the NTK regime (tangent finetuning + linearized delta) make the source delta easier to
transport than the ordinary nonlinear regime? Two arms of the same t5-base source:

- ``lin``: ``t5base_6text_noreg_ntk`` checkpoints, ``feature_regime="linear"`` (delta_A = J(theta0) dtheta);
- ``nl``:  ``t5base_6text_noreg_nonlinear_v2`` checkpoints, ``feature_regime="standard"``.

Cells (``CELLS``): global ridge and global MLP in both arms; block ridge on B's attention outputs (input of
each block's W_O), pooled per segment [premise; hypothesis; global], only in ``lin``, because block_ridge needs the per-block linearized deltas.
``*_global_{ridge,mlp}_seg``: the global cells with the regressor's input segment-pooled from B's last hidden
state (``global_input_pooling="segments"``); head and Stage 1 unchanged. Their ``features_B_output_segpool.pt``
is computed from B alone and cached beside the existing split files, which are reused.

Every config shares stage1_lambda=1, alpha=1 (no alpha search), and the target's fs300/seed33 nearest-mean
head regardless of few_shot, so the few_shot axis varies only the steer support.

Feature cache: ONE root for both arms. The arms are kept apart only by the regime sub-directory
(``<pair>/<task>/linear`` vs ``<pair>/<task>/standard``). This is safe *only* because regime and checkpoint
are 1:1 here: the cache key does not include the source checkpoint, so no other checkpoint may ever write
into this root under either regime.

    python configs/ntk_rebasin_study/make_configs.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/work/intesasanpaolo_phd/merge-and-rebase")
TEXT_FEATURE = WORK / "features/text_feature/results/ntk_segment_pooling"
CACHE_DIR = TEXT_FEATURE / "feature_cache"

TASKS = ("mnli", "qnli", "rte", "scitail", "sick", "snli")
FEW_SHOTS = (200, 300, 500)
SEEDS = (33, 54, 89)

TARGETS = {
    "t5-large": {
        "target_model_name_or_path": "google-t5/t5-large",
        "target_model_arch": "t5",
        "head": lambda task: TEXT_FEATURE / f"heads_fs300/t5-large_{task}_nearest_mean_seed33_fewshot300.pt",
    },
    "roberta-base": {
        "target_model_name_or_path": "FacebookAI/roberta-base",
        "target_model_arch": "auto",
        "head": lambda task: TEXT_FEATURE / f"heads_roberta/roberta-base_{task}_nearest_mean_seed33_fewshot300.pt",
    },
}

ARMS = {
    "lin": {"feature_regime": "linear", "ckpt_root": WORK / "checkpoints_t5_converted/t5base_6text_noreg_ntk"},
    "nl": {"feature_regime": "standard", "ckpt_root": WORK / "checkpoints_t5_converted/t5base_6text_noreg_nonlinear_v2"},
}

CELLS: dict[str, dict] = {
    "lin_global_ridge": {"arm": "lin", "stage_2_strategy": "global_ridge", "ridge_lambda": 1.0},
    "lin_global_mlp": {"arm": "lin", "stage_2_strategy": "global_mlp", "mlp_epochs": 100, "mlp_hidden_dim": 1024},
    "lin_block_ridge_attn_seg": {
        "arm": "lin",
        "stage_2_strategy": "block_ridge",
        "block_source": "attention",
        "target_block_pooling": "segments",  # B blocks pooled as [premise; hypothesis; global]
        "block_pooling": "mean",
        "block_ridge_mode": "smoothed_residual",
        "rho": 0.9,
        "block_ridge_lambda_scaling": "trace",
        "ridge_lambda": 0.1,  # beta: block b is penalized with beta * ||X_b||_F^2 / n
        "block_group_strategy": "concat",
    },
    "nl_global_ridge": {"arm": "nl", "stage_2_strategy": "global_ridge", "ridge_lambda": 1.0},
    "nl_global_mlp": {"arm": "nl", "stage_2_strategy": "global_mlp", "mlp_epochs": 100, "mlp_hidden_dim": 1024},
}
# Segment-pooled twins of the global cells: the regressor reads B's last hidden state as
# [premise; hypothesis; global] (3D) and still predicts the D-dim correction to the pooled feature.
# Everything else is the twin's, so <cell>_seg - <cell> isolates the regressor's input pooling.
for _name in ("lin_global_ridge", "lin_global_mlp", "nl_global_ridge", "nl_global_mlp"):
    CELLS[f"{_name}_seg"] = {**CELLS[_name], "global_input_pooling": "segments"}


def build(target: str, cell: str, task: str, few_shot: int, seed: int) -> dict:
    spec = {k: v for k, v in CELLS[cell].items() if k != "arm"}
    arm = ARMS[CELLS[cell]["arm"]]
    tgt = TARGETS[target]
    return {
        "suite": "nli6",
        "source_model_name_or_path": "google-t5/t5-base",
        "source_model_arch": "t5",
        "target_model_name_or_path": tgt["target_model_name_or_path"],
        "target_model_arch": tgt["target_model_arch"],
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
        # A's ntk checkpoint through a nonlinear forward scores near chance, so the control is only
        # meaningful for the nonlinear arm.
        "eval_source_finetuned": arm["feature_regime"] == "standard",
        "save_steer_artifacts_dir": None,
        "_description": f"ntk_rebasin_study {cell}: t5-base -> {target}, {task}, few_shot={few_shot}, seed={seed}.",
        "tasks": task,
        "seed": seed,
        "tuned_ckpts": {task: str(arm["ckpt_root"] / task)},
        "target_task_heads": str(tgt["head"](task)),
        "method_params": {
            "feature_regime": arm["feature_regime"],
            "stage1_lambda": 1.0,
            "force_recompute_features": False,
            "few_shot": few_shot,
            "feature_cache_dir": str(CACHE_DIR),
            **spec,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()

    n = 0
    for target in TARGETS:
        for cell in CELLS:
            for task in TASKS:
                for few_shot in FEW_SHOTS:
                    for seed in SEEDS:
                        path = args.out_dir / target / task / f"{cell}_fs{few_shot}_seed{seed}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(build(target, cell, task, few_shot, seed), indent=2) + "\n")
                        n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
