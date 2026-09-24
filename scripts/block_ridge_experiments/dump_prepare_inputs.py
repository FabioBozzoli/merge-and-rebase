"""Run text_rebase up to steer_text.prepare(), save what an offline refit needs, and exit.

Everything block_ridge fits on is in the feature cache except what comes off the live
models and loaders: the two heads, mask_class and the train split's local labels (which
_few_shot draws the support from). This saves exactly those, plus the cache keys, to
<local-log-dir>/<task>_prepare_inputs.pt, so a refit script can sweep strategies, seeds
and support sizes without loading a model.

Launched as a text_rebase entrypoint, with the same arguments:
    ENTRYPOINT=scripts.block_ridge_experiments.dump_prepare_inputs \
      scripts/slurm/submit_text_rebase.sh <exp-name> <config> [key=value ...]
"""
import os
import runpy
import sys

import torch

import merge_and_rebase.rebase.text.steer_text as st


def _dump(self, **kwargs):
    w_a, _ = st._head_tensors(kwargs["llm_source"].model)
    w_b, b_b = st._head_tensors(kwargs["llm_target"].model)
    task = kwargs["task"]
    out = os.path.join(log_dir, f"{task}_prepare_inputs.pt")
    torch.save(
        {
            "task": task,
            "w_a": w_a,
            "w_b": w_b,
            "b_b": b_b,
            "mask_class": None if kwargs.get("mask_class") is None else [int(c) for c in kwargs["mask_class"]],
            "local_labels_train": torch.as_tensor(
                [int(y) for y in kwargs["source_loaders"].local_labels["train"]], dtype=torch.long
            ),
            "cache": {
                "feature_cache_dir": kwargs["feature_cache_dir"],
                "source_tag": kwargs["source_tag"],
                "target_tag": kwargs["target_tag"],
                "feature_regime": kwargs["feature_regime"],
            },
            "method_params": {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool, type(None)))},
        },
        out,
    )
    print(f"[dump_prepare_inputs] wrote {out}", flush=True)
    raise SystemExit(0)


log_dir = sys.argv[sys.argv.index("--local-log-dir") + 1]
st.SteerTextRebase.prepare = _dump
sys.argv = ["text_rebase", *sys.argv[1:]]
runpy.run_module("merge_and_rebase.eval.text_rebase", run_name="__main__")
