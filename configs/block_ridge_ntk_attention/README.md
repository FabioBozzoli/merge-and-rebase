# block_ridge on B's attention outputs (ntk arm) — reproducible runs

Transport t5-base ntk finetunes (A) to t5-large (B) with `steer_text`, using **block_ridge with
trace-scaled λ and smoothed-residual carry, fit on B's per-block attention outputs** (the input
of each block's `SelfAttention.o`). Why this setting, and the offline evidence behind it:
`docs/block-ridge-label-free-and-pooling.md` (§13).

This directory holds, per task, one fully resolved config per (cell, seed). A run is one plain
`python` command; nothing is resolved at launch time and no two runs share any file.

## Cells (tasks mnli, qnli, rte, scitail, sick, snli; seeds 33 / 54 / 89)

| cell | Stage 2 | B block features | `ridge_lambda` |
|---|---|---|---|
| `attention_block_ridge_trace_carry` | **the method**: `block_ridge`, `block_ridge_lambda_scaling=trace`, `block_ridge_mode=smoothed_residual`, `rho=1.0` | `block_source=attention` (input of `SelfAttention.o`), mean-pooled | β = 0.1 |
| `residual_block_ridge_trace_carry` | reference: same fit | `block_source=residual` (block outputs), mean-pooled | β = 1e-4 |
| `global_ridge` | reference: one ridge on B's pooled output feature f_B | — | λ = 0.625 (absolute) |

All three share: Stage-1 λ = 1, concat grouping of B's 24 blocks into A's 12, `reuse_logitmap`
targets, identity weights, α = 1 (no α search), `feature_regime=linear`.

- **Support** (`few_shot`, per class, drawn by the seeded class-balanced sampler `_few_shot`):
  the largest of the original grid, 1000, except **sick, 500** — its smallest class has 606 rows
  in the 4000-row pool, so the grid stopped at 500 there. Labels are used only to balance the
  draw; the fit itself is label-free.
- **`ridge_lambda`**. For block_ridge with `block_ridge_lambda_scaling=trace` it is β: each
  block is penalized with β · tr(X_b X_bᵀ)/n, i.e. relative to its own mean eigenvalue. For
  global_ridge it is an absolute λ. The values are the ones the offline refit picked **on
  snli**, on held-out rows without labels (`scripts/block_ridge_experiments/preprocess_curve.py`,
  logit-space R²): β = 0.1 for the attention cell, 1e-4 for the residual cell, and for global
  β = 0.1 × mean‖f_B‖² (6.25 on snli) = 0.625. **They are not tuned per task**: the same values
  are used for every task. For global this is β ≈ 0.08–0.12 across tasks.
- **`eval_source_finetuned=false`**: that control evaluates A's ntk checkpoint through a
  nonlinear forward, where it scores near chance; it does not affect the result.

## Requirements

- **This repository** at the commit that added these configs, with its environment:
  `uv sync` (pinned by `uv.lock`). Recorded runs used Python 3.12.13, torch 2.11.0+cu128,
  transformers 5.17.0, datasets 5.0.1.
- **One GPU** supported by that torch build (compute capability ≥ 7.5; the cluster's P100s are
  not). Peak memory is small (t5-large forward, batch 16); ask for 32 GB of host RAM.
- **Read access** to these files (paths are absolute in the configs):
  - `/work/intesasanpaolo_phd/merge-and-rebase/checkpoints_t5_converted/t5base_6text_noreg_ntk/<task>` (A)
  - `/work/intesasanpaolo_phd/merge-and-rebase/text_rebase_heads/t5-large_<task>_nearest_mean_enc_seed33_fewshot300.pt` (B's head)
- **Hugging Face Hub** (downloaded on first use into your `HF_HOME`). Recorded runs used:

  | repository | revision |
  |---|---|
  | `google-t5/t5-base` | `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` |
  | `google-t5/t5-large` | `150ebc2c4b72291e770f58e6057481c8d2ed331a` |
  | `stanfordnlp/snli` | `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` |
  | `nyu-mll/glue` (mnli, qnli, rte) | `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` |
  | `allenai/scitail` | `0cc4353235b289165dfde1c7c5d1be983f99ce44` |
  | `yangwang825/sick` | `4c90d63ab7702fcb591bb3a4d5d9bd1ccff50127` |

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

# every run of every task, one after the other (they are independent: any order,
# or in parallel on several GPUs)
for task in mnli qnli rte scitail sick snli; do
  for cfg in $REPO/configs/block_ridge_ntk_attention/$task/*_seed*.json; do
    python -m merge_and_rebase.eval.text_rebase --config $cfg \
      --local-log-dir results/block_ridge_ntk_attention/$task/$(basename $cfg .json) --run-name summary
  done
done
```

Each run writes `results/block_ridge_ntk_attention/<task>/<config>/`:
`summary.json` (results), `summary.events.jsonl`, and `feature_cache/` (its own features).
A cold run takes 6–8 minutes on one GPU (A's per-block jvps and B's blocks are computed once,
then the fit and the evaluation); rerunning the same config reuses its `feature_cache/`.
Delete that directory, or set `method_params.force_recompute_features: true`, to recompute.

On a Slurm cluster, wrap the same command, one job per config, e.g.
`sbatch --gres=gpu:1 --mem=32G --time=01:00:00 --wrap "bash -c 'cd <run-dir> && source $REPO/.venv/bin/activate && python -m merge_and_rebase.eval.text_rebase --config <cfg> --local-log-dir results/block_ridge_ntk_attention/<task>/<stem> --run-name summary'"`.

## Check

```bash
for task in mnli qnli rte scitail sick snli; do
  python $REPO/configs/block_ridge_ntk_attention/check_results.py results/block_ridge_ntk_attention/$task --task $task
done
```

For each task it prints every run's stage 0 (B alone), stage 1 (oracle), stage 2 (fit, in
cached-feature space) and the live "rebased" accuracy, and checks that live equals cached stage 2
and that all numbers equal `<task>/expected_results.json`. Exit status 0 means reproduced.

Expected live test accuracy (the checker compares all digits; per-seed values are in
`<task>/expected_results.json`):

| task | support | B alone | oracle | **attention block_ridge** (seeds 33 / 54 / 89) | global_ridge | residual block_ridge |
|---|---|---|---|---|---|---|
| mnli | 1000/class | 0.5072 | 0.8154 | **0.7350 ± 0.0058** (0.7283 / 0.7383 / 0.7383) | 0.7169 ± 0.0089 | 0.6857 ± 0.0029 |
| qnli | 1000/class | 0.6839 | 0.8576 | **0.8311 ± 0.0058** (0.8283 / 0.8378 / 0.8272) | 0.7969 ± 0.0042 | 0.7902 ± 0.0058 |
| rte | 1000/class | 0.6265 | 0.6600 | **0.7015 ± 0.0167** (0.6827 / 0.7149 / 0.7068) | 0.6693 ± 0.0123 | 0.5984 ± 0.0145 |
| scitail | 1000/class | 0.7739 | 0.9011 | **0.8863 ± 0.0008** (0.8861 / 0.8872 / 0.8856) | 0.8652 ± 0.0016 | 0.8191 ± 0.0148 |
| sick | 500/class | 0.7622 | 0.8485 | **0.8257 ± 0.0041** (0.8211 / 0.8289 / 0.8272) | 0.8130 ± 0.0047 | 0.7898 ± 0.0050 |
| snli | 1000/class | 0.5911 | 0.8724 | **0.8031 ± 0.0049** (0.8044 / 0.7978 / 0.8072) | 0.7774 ± 0.0036 | 0.7596 ± 0.0090 |

Test sets have 1800 examples, except rte (249: one example is 0.4 points).

Verified bit-identical on snli across: a first run on a shared cache, a warm rerun of all nine,
a fresh cache filled from scratch, and all nine run concurrently cold with per-run caches; the
other tasks' references come from one concurrent cold run of all 45 configs. All ran on the same
cluster's GPUs; on different hardware, GPU arithmetic can differ in the last bits and
occasionally flip a test example. Use `--tol 0.0006` (one example of 1800; 0.0041 for rte) to
tell such a difference from a real one.

## Regenerate the configs

`python configs/block_ridge_ntk_attention/make_configs.py` writes them;
`--check` exits non-zero if any file differs from what it would write. Edit `make_configs.py`,
not the JSONs.
