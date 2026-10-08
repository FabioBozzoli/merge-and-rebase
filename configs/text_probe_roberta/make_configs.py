"""Generate the FacebookAI/roberta-base few-shot linear-probe configs (the baseline for steer_text).

One config per (few_shot, seed), all six NLI tasks in it. The split settings are the ones of the
steering configs (``configs/block_ridge_ntk_roberta``), so val and test are the same rows, and
``linear_probe_support="steer"`` draws the support with steer_text's own ``_few_shot`` -- at the same
few_shot/seed the probe sees exactly the examples the steering saw. ``linear_probe_lr`` is a list: the
lr is chosen on val, per task (see ``eval/text_linear_probe.py``).

    python configs/text_probe_roberta/make_configs.py [--out-dir DIR]
    python -m merge_and_rebase.eval.text_linear_probe --config configs/text_probe_roberta/fs200_seed33.json \\
        --run-name probe_roberta_fs200_seed33 --local-log-dir results/text_probe_roberta/logs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("mnli", "qnli", "rte", "scitail", "snli", "sick")
SICK_MAX_FEW_SHOT = 500  # only 606 class-2 rows in the 4000-row train pool: more shots per class is infeasible


def build(few_shot: int, seed: int, lrs: list[float], epochs: int) -> dict:
    return {
        "_description": f"roberta-base linear probe, nearest-mean init, {few_shot}/class, seed {seed}, lr chosen on val.",
        "suite": "nli6",
        "tasks": [t for t in TASKS if not (t == "sick" and few_shot > SICK_MAX_FEW_SHOT)],
        "target_model_name_or_path": "FacebookAI/roberta-base",
        "target_model_arch": "auto",
        "model_kind": "encoder_classification",
        "num_labels": 3,
        "device": "cuda",
        "dtype": "fp32",
        "split": "test",
        "val_fraction": 0.1,
        "max_samples_per_task": 2000,
        "max_train_samples": 4000,
        "batch_size": 16,
        "num_workers": 0,
        "max_length": 256,
        "seed": seed,
        "linear_probe_shots_per_class": few_shot,
        "linear_probe_support": "steer",
        "linear_probe_init": "nearest_mean",
        "linear_probe_epochs": epochs,
        "linear_probe_lr": lrs,
        "linear_probe_dropout": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--few-shots", nargs="+", type=int, default=[200, 300, 500, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 54, 89])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for few_shot in args.few_shots:
        for seed in args.seeds:
            path = args.out_dir / f"fs{few_shot}_seed{seed}.json"
            path.write_text(json.dumps(build(few_shot, seed, args.lrs, args.epochs), indent=2) + "\n")
            n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
