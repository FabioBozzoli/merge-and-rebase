# block_ridge on B's attention outputs (ntk arm) — reproducible runs

Transport t5-base ntk finetunes (A) to t5-large (B) with `steer_text`, using **block_ridge with
trace-scaled λ and smoothed-residual carry, fit on B's per-block attention outputs** (the input
of each block's `SelfAttention.o`). Why this setting, and the offline evidence behind it:
`docs/block-ridge-label-free-and-pooling.md` (§13).

This directory holds, per task, one fully resolved config per (cell, seed). A run is one plain
`python` command; nothing is resolved at launch time and no two runs share any file.

## Cells (task: snli, few_shot = 1000 per class, seeds 33 / 54 / 89)

| cell | Stage 2 | B block features | `ridge_lambda` |
|---|---|---|---|
| `attention_block_ridge_trace_carry` | **the method**: `block_ridge`, `block_ridge_lambda_scaling=trace`, `block_ridge_mode=smoothed_residual`, `rho=1.0` | `block_source=attention` (input of `SelfAttention.o`), mean-pooled | 0.1 |
| `residual_block_ridge_trace_carry` | reference: same fit | `block_source=residual` (block outputs), mean-pooled | 1e-4 |
| `global_ridge` | reference: one ridge on B's pooled output feature f_B | — | 0.625 |

All three share: Stage-1 λ = 1, concat grouping of B's 24 blocks into A's 12, `reuse_logitmap`
targets, identity weights, α = 1 (no α search), `feature_regime=linear`.

- **few_shot = 1000** is the largest support of the original grid (groups `*-fs1000`): 1000
  examples per class, 3000 rows, drawn by the seeded class-balanced sampler (`_few_shot`).
  Labels are used only to balance the draw; the fit itself is label-free.
- **`ridge_lambda`** is the value the offline refit picked on held-out rows without labels
  (`scripts/block_ridge_experiments/preprocess_curve.py`, logit-space R²): 0.1 for the attention
  cell (the same at 1500 and 3400 rows), 1e-4 for the residual cell (the 3400-row pick), and for
  global 0.1 × mean‖f_B‖² = 0.625.
- **`eval_source_finetuned=false`**: that control evaluates A's ntk checkpoint through a
  nonlinear forward, where it scores near chance; it does not affect the result.

## Requirements

- **This repository** at the commit that added this directory, with its environment:
  `uv sync` (pinned by `uv.lock`). Recorded runs used Python 3.12.13, torch 2.11.0+cu128,
  transformers 5.17.0, datasets 5.0.1.
- **One GPU** supported by that torch build (compute capability ≥ 7.5; the cluster's P100s are
  not). Peak memory is small (t5-large forward, batch 16); ask for 32 GB of host RAM.
- **Read access** to these files (paths are absolute in the configs):
  - `/work/intesasanpaolo_phd/merge-and-rebase/checkpoints_t5_converted/t5base_6text_noreg_ntk/snli` (A)
  - `/work/intesasanpaolo_phd/merge-and-rebase/text_rebase_heads/t5-large_snli_nearest_mean_enc_seed33_fewshot300.pt` (B's head)
- **Hugging Face Hub** (downloaded on first use into your `HF_HOME`). Recorded runs used
  `google-t5/t5-base@a9723ea7`, `google-t5/t5-large@150ebc2c`, `stanfordnlp/snli@cdb5c3d5`.
  A different revision changes the numbers; `check_results.py` will show it.
- **Disk**: ~1.4 GB per run for its feature cache, under the directory you run from.

## Run

From any directory you want the results in (paths inside the configs are relative to it):

```bash
REPO=/path/to/merge-and-rebase          # this repository
source $REPO/.venv/bin/activate
export PYTHONUNBUFFERED=1

# one run
cfg=$REPO/configs/block_ridge_ntk_attention/snli/snli_attention_block_ridge_trace_carry_fs1000_seed33.json
python -m merge_and_rebase.eval.text_rebase --config $cfg \
  --local-log-dir results/block_ridge_ntk_attention/snli/$(basename $cfg .json) --run-name summary

# all nine, one after the other (they are independent: any order, or in parallel on several GPUs)
for cfg in $REPO/configs/block_ridge_ntk_attention/snli/*_seed*.json; do
  python -m merge_and_rebase.eval.text_rebase --config $cfg \
    --local-log-dir results/block_ridge_ntk_attention/snli/$(basename $cfg .json) --run-name summary
done
```

Each run writes `results/block_ridge_ntk_attention/snli/<config>/`:
`summary.json` (results), `summary.events.jsonl`, and `feature_cache/` (its own features).
A cold run takes 6–8 minutes on one GPU (A's per-block jvps and B's blocks are computed once,
then the fit and the evaluation); rerunning the same config reuses its `feature_cache/`.
Delete that directory, or set `method_params.force_recompute_features: true`, to recompute.

On a Slurm cluster, wrap the same command, e.g.
`sbatch --gres=gpu:1 --mem=32G --time=01:00:00 --wrap "bash -c 'cd <run-dir> && source $REPO/.venv/bin/activate && python -m merge_and_rebase.eval.text_rebase --config <cfg> --local-log-dir results/block_ridge_ntk_attention/snli/<stem> --run-name summary'"`.

## Check

```bash
python $REPO/configs/block_ridge_ntk_attention/check_results.py results/block_ridge_ntk_attention/snli
```

It prints every run's stage 0 (B alone), stage 1 (oracle), stage 2 (fit, in cached-feature space)
and the live "rebased" accuracy, and checks that live equals cached stage 2 and that all
numbers equal `snli/expected_results.json`. Exit status 0 means reproduced.

Expected (test accuracy, 1800 examples; the checker compares all digits):

| cell | seed 33 | seed 54 | seed 89 | mean ± sd |
|---|---|---|---|---|
| attention block_ridge trace + carry | 0.804444 | 0.797778 | 0.807222 | **0.8031 ± 0.0049** |
| global_ridge | 0.777222 | 0.781111 | 0.773889 | 0.7774 ± 0.0036 |
| residual block_ridge trace + carry | 0.766667 | 0.762778 | 0.749444 | 0.7596 ± 0.0090 |
| B alone (stage 0) | 0.591111 | 0.591111 | 0.591111 | |
| oracle (stage 1) | 0.873889 | 0.872222 | 0.871111 | |

Verified bit-identical across: a first run on a shared cache, a warm rerun of all nine, a
fresh cache filled from scratch, and all nine run concurrently cold with per-run caches.
These ran on the same cluster's GPUs; on different hardware, GPU arithmetic can differ in the
last bits and occasionally flip a test example. Use `--tol 0.0006` (one example of 1800) to
tell such a difference from a real one.

## Regenerate the configs

`python configs/block_ridge_ntk_attention/make_configs.py` writes them;
`--check` exits non-zero if any file differs from what it would write. Edit `make_configs.py`,
not the JSONs.
