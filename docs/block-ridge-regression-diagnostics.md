# Why `block_ridge` underperforms: regression diagnostics

`steer_text` reports three accuracies per run: `stage0` (B alone), `stage1`
(the *oracle*: the correction built from A's true test-time delta) and `stage2`
(the fitted predictor). On the t5-base→t5-large ntk arm, `block_ridge` recovers
only a small part of the stage0→stage1 gap and trails `global_ridge`. Those
three numbers alone do not show why. This document describes the tooling that
treats Stage 2 as the regression it is, and records what it found.

The mechanics of a run live in [`t5-encoder-steer-flow.md`](t5-encoder-steer-flow.md),
the grid in [`t5-encoder-steer-grid.md`](t5-encoder-steer-grid.md).

---

## Running it

**1. Rerun the cells with their artifacts kept.** The grid deliberately drops
`block_ridge`'s `<task>_steer_artifacts.pt` (~218 MB each), and `grid.py submit`
only submits *missing* cells, so it cannot redo a completed one. Instead:

```
python scripts/slurm/submit_block_ridge_diag.py --group ntk-br-fs200 [--task mnli] [--seed 33] [--dry-run]
```

This reuses `grid.py`'s own `overrides_for` / `resources_for`, so each resolved
`config.json` is the grid cell's config byte-for-byte except
`save_steer_artifacts_dir`. It writes under a separate root,
`/work/.../t5enc_block_ridge_diag/`, and refuses the grid's root, so it cannot
overwrite a finished run. The feature cache is warm, so each job only refits and
re-evaluates (~15 min slot). It refuses to run against a cold cache.
`--dry-run` still creates the experiment directory (`submit_text_rebase.sh`
resolves the config before it checks `DRY_RUN`), so point it at a scratch
`--results-root` first.

**2. Diagnose one run** (CPU only, reads the cache plus the artifacts):

```
source .venv/bin/activate
python scripts/diagnose_block_ridge.py \
  --exp-dir /work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag/mnli_linear_noreg-ntk_block_ridge_fs200_seed33 \
  --output  /work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag/_reports/mnli_..._seed33.json
```

## What the script checks

Nothing about `block_ridge` is refit. The Stage-1 map, `pinv(w_B)`, the support
indices `selected` and the 13 ridge coefficients all come from the artifact file.
Grouping and prediction use `steer.py`'s `_BLOCK_GROUP_STRATEGIES` and
`_predict_block_ridge`, and accuracy uses `steer_text._accuracy`.

**Validation gate.** Before reporting anything, the script recomputes
stage0/1/2 test accuracy. It compares them with the run's own recorded
diagnostics, with the original grid cell of the same name, and it compares its
`global_ridge` refit with the `global_ridge` twin cell. If any comparison is off
by more than one example, it exits non-zero without writing a report.

**Regression fit** is measured on three row sets:

| rows | question |
|---|---|
| `train_support` | the `selected` rows the ridge was fit on: does it fit? |
| `train_unselected` | the rest of the train split: does it generalize within the split? |
| `test` | the eval split: does it generalize? |

Each row set gets these metrics:

- **R²**, computed with sklearn's conventions (implemented in torch, since sklearn is not in `.venv`).
  - `variance_weighted`: the fraction of total target variance explained.
  - `uniform_average`: the mean of per-dim R².
- **Cosine**, the median **norm ratio** `|pred|/|true|` (the shrinkage lens), and the median relative error, all per example.

These are computed in two spaces:

- **Logit space** (`correction @ w_B[mask_class].T`) is the headline. The Stage-1 target
  `δ_A · logit_mapᵀ · pinv(w_B)ᵀ` lies in the row space of `w_B`, which has at most C
  dimensions. The head sees the correction only through those directions, so
  logit-space fit is what decides accuracy.
- **Feature space** is reported alongside it.

Metrics are reported for the summed correction, for each block against its own
per-block target (with that block's share of target energy), and for a
`global_ridge` refit on the same support (the production `_ridge`, same
`selected`, same Stage-1 map).

The script also asserts that the per-block deltas sum to the full delta. They
do, with relative error ~1e-6, so `block_ridge` targets the same quantity the
oracle scores.

**Hard vs easy** (test only):

- the 8-way stage0/stage1/stage2 correctness cross-tab. The key bucket is
  `stage0 wrong, stage1 right, stage2 wrong`: fixable, but missed.
- a block-vs-global correctness 2×2.
- accuracy and fit per quartile of (a) the logit-space norm of the oracle
  correction and (b) B's stage0 margin (true logit minus the best other logit).

## Reading the numbers

| pattern | meaning |
|---|---|
| train R² low | underfitting: too little capacity, or λ too strong for the feature scale |
| train R² ≈ 1, test R² ≪ train | overfitting the support; the ridge is interpolating |
| test R² high, stage2 acc still low | the regression is fine; the problem is Stage 1 / head geometry |
| norm ratio ≪ 1 on large-correction quartiles | L2 shrinkage specifically eats large corrections |

## Findings (ntk arm, fs200, `independent`, λ = 1)

**mnli, seed 33:**

| rows | model | logit R² (vw) | cosine | \|p\|/\|y\| |
|---|---|---|---|---|
| train_support | block_ridge | **1.000** | 1.000 | 1.00 |
| train_support | global_ridge | 0.867 | 0.963 | 0.91 |
| test | block_ridge | **0.057** | 0.730 | 1.00 |
| test | global_ridge | 0.379 | 0.830 | 0.89 |

Test accuracy was 0.507 for stage0, 0.817 for the stage1 oracle, 0.562 for
block_ridge and 0.666 for global_ridge. Every block regressor from 0 to 11 fits
its support exactly (R² = 1.000) and scores **negative** R² on test (−0.18 to
−0.95). The summed correction reaches only 0.057.

This is overfitting, and the cause is **feature scale versus a single λ**. On the
600 support rows:

| input | mean ‖x‖ | Gram eigenvalues | eigenvalues < λ = 1 |
|---|---|---|---|
| blocks 0–11 (B's residual stream, 2 blocks concatenated, 2048-d) | 7e2 → 5e4 | min 2.6e2 → 7e4 | **0 of 600** |
| block 12 = `global_ridge`'s input (post final LayerNorm) | 2.4 | min 0.03, median 0.87 | 321 of 600 |

T5's un-normalized residual stream grows by orders of magnitude with depth. With
n = 600 < d = 2048, `_ridge` takes the dual branch, and `(XXᵀ + λI)` is
effectively `XXᵀ` for every intermediate block. Each one is therefore a
minimum-norm interpolator. The same λ = 1 regularizes `global_ridge`, and
block 12 of `block_ridge`, meaningfully. The correction is carried by the deep
blocks (blocks 9–11 hold ~60% of the target energy), which are exactly the
largest-scale, least-regularized ones. The errors of 12 interpolators then add
up in the sum.

This is not the "no λ sweep" caveat in another form. Any single λ is wrong
across blocks, because their scales span five orders of magnitude.

The hard-vs-easy view agrees:

- 323 test examples (18%) are "fixable but missed", against 56 that
  block_ridge fixes and the oracle does not.
- block_ridge loses to global_ridge in every quartile of both axes.
- The norm ratio is ≈ 1 on average, so this is not shrinkage. It is above 1
  on small corrections and falls to 0.73 on the largest, while R² goes from
  −2.5 to 0.22. The predictions are noisy in magnitude and direction rather
  than biased toward zero.

Cross-task results are in the table below (all 6 tasks × 3 seeds). Reports
are under `t5enc_block_ridge_diag/_reports/`.

All 18 runs pass the validation gate. The table shows the mean over seeds 33, 54 and 89. Accuracy is on test. R² is logit-space and variance-weighted. "Worst block" is the worst test R² among blocks 0–11.

| task | stage0 | oracle | block_ridge | global_ridge | block train R² | block test R² | global train R² | global test R² | worst block | fixable-but-missed |
|---|---|---|---|---|---|---|---|---|---|---|
| mnli | 0.507 | 0.815 | 0.559 | 0.655 | 1.000 | 0.040 | 0.854 | 0.310 | -1.28 | 0.172 |
| sick | 0.762 | 0.848 | 0.747 | 0.798 | 1.000 | 0.451 | 0.911 | 0.663 | -1.75 | 0.056 |
| rte | 0.627 | 0.664 | 0.577 | 0.616 | 1.000 | -1.503 | 0.747 | 0.043 | -3.98 | 0.076 |
| scitail | 0.774 | 0.902 | 0.780 | 0.826 | 1.000 | 0.310 | 0.890 | 0.624 | -7.09 | 0.052 |
| qnli | 0.684 | 0.858 | 0.712 | 0.761 | 1.000 | 0.244 | 0.866 | 0.473 | -2.00 | 0.094 |
| snli | 0.591 | 0.871 | 0.656 | 0.734 | 1.000 | 0.315 | 0.853 | 0.607 | -1.98 | 0.133 |

The failure mode is the same everywhere:

- block_ridge's train R² is exactly 1.000 on every task.
- global_ridge's train R² is 0.75–0.91.
- block_ridge's test R² is 0.2–0.3 lower than global_ridge's on five tasks, and 1.5 lower on rte.
- On sick and rte, block_ridge ends up *below* B's uncorrected baseline.

**Directions this points to** (not implemented here):

- Standardize each block's inputs before its ridge, for example to unit mean
  row norm, or apply a LayerNorm to match the global features.
- Alternatively, scale λ per block, e.g. `λ_b = λ · tr(X_bX_bᵀ)/n`.
