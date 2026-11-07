"""Generate the segment-pooled few-shot linear-probe configs for t5-large and roberta-base, with their global twins.

``feature_pooling="segments"`` makes the head read ``[premise mean; hypothesis mean; global mean]`` of the
last layer (3x wider head); ``"global"`` is the plain masked mean, i.e. the existing probe. Both variants
are emitted from the same template so they differ in that one field only, and the paired difference
(segments - global) at the same target/few_shot/seed isolates the effect of the pooling.

Split, support draw (``linear_probe_support="steer"``) and probe budget follow
``configs/text_probe_roberta/make_configs.py``, so at the same few_shot/seed the probe sees the examples
steer_text's support saw. No ``premise_hypothesis_template``: segment pooling needs the pair encoding
(``segment_masks_from_eos`` raises on a single-string input).

    python configs/text_probe_segments/make_configs.py [--out-dir DIR]
    python -m merge_and_rebase.eval.text_linear_probe \\
        --config configs/text_probe_segments/roberta-base/segments_fs300_seed33.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("mnli", "qnli", "rte", "scitail", "snli", "sick")
FEW_SHOTS = (200, 300, 500)
SEEDS = (33, 54, 89)
POOLINGS = ("global", "segments")

TARGETS = {
    "t5-large": {
        "target_model_name_or_path": "google-t5/t5-large",
        "target_model_arch": "t5",
        "lrs": [1e-3, 3e-3, 1e-2],  # the team's T5 probes use 1e-2
    },
    "roberta-base": {
        "target_model_name_or_path": "FacebookAI/roberta-base",
        "target_model_arch": "auto",
        "lrs": [1e-4, 3e-4, 1e-3],  # as configs/text_probe_roberta
    },
}


def build(target: str, pooling: str, few_shot: int, seed: int, epochs: int) -> dict:
    tgt = TARGETS[target]
    return {
        "_description": (
            f"{target} linear probe, feature_pooling={pooling}, nearest-mean init, {few_shot}/class, "
            f"seed {seed}, lr chosen on val."
        ),
        "suite": "nli6",
        "tasks": list(TASKS),
        "target_model_name_or_path": tgt["target_model_name_or_path"],
        "target_model_arch": tgt["target_model_arch"],
        "model_kind": "encoder_classification",
        "feature_pooling": pooling,
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
        "linear_probe_lr": tgt["lrs"],
        "linear_probe_dropout": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    n = 0
    for target in TARGETS:
        for pooling in POOLINGS:
            for few_shot in FEW_SHOTS:
                for seed in SEEDS:
                    path = args.out_dir / target / f"{pooling}_fs{few_shot}_seed{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(build(target, pooling, few_shot, seed, args.epochs), indent=2) + "\n")
                    n += 1
    print(f"Wrote {n} configs under {args.out_dir}")


if __name__ == "__main__":
    main()
