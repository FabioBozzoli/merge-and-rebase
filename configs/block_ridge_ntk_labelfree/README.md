# Label-free block_ridge / joint_ridge / global_ridge on the ntk arm

Configs for the setting found in `docs/block-ridge-label-free-and-pooling.md` (t5-base ntk →
t5-large encoder, `feature_regime=linear`): the support is **3400 unlabelled train rows**
drawn at random (`total_support_examples`, i.e. `_random_sample`; rte: 1890) instead of
`few_shot` per class. steer_text never uses labels to fit, so this only needs inputs run
through A and B.

Every JSON here is fully resolved from `configs/text_rebase_t5enc_steer.json` by
`scripts/slurm/resolve_config.py`, with the grid's own per-task overrides (`grid.overrides_for`
for the `noreg-ntk:linear` arm) plus the settings below. Launch them, 3 seeds each:

```bash
configs/block_ridge_ntk_labelfree/launch.sh                          # every config, all tasks
configs/block_ridge_ntk_labelfree/launch.sh snli qnli                # some tasks
CELLS="global_ridge joint_ridge_unitnorm" configs/block_ridge_ntk_labelfree/launch.sh snli
DRY_RUN=1 configs/block_ridge_ntk_labelfree/launch.sh                # resolve only
```

Runs go to `$RESULTS_ROOT` (default `/work/intesasanpaolo_phd/merge-and-rebase/t5enc_steer_text_labelfree`),
never the grid's root, named `<task>_labelfree_<cell>_seed<seed>`. `launch.sh` needs
`scripts/slurm/submit_text_rebase.sh`, `resolve_config.py` and `text_rebase_job.sbatch`.

## Cells

| cell (file `<task>_<cell>.json`) | Stage 2 | B block pooling | preprocessing | penalty |
|---|---|---|---|---|
| `global_ridge` | `global_ridge` on f_B | — | — | `ridge_lambda = β · mean‖f_B‖²` (absolute) |
| `block_ridge_trace_carry` | `block_ridge`, `block_ridge_lambda_scaling=trace`, `block_ridge_mode=smoothed_residual`, `rho=1.0`, concat grouping, `reuse_logitmap` targets, identity weights | `mean` | `none` | `ridge_lambda = β` (per block: β · tr(K_b)/n) |
| `block_ridge_trace_carry_<pooling>[_zscore]` | as above | `unitnorm` / `rmsnorm` | `none` / `zscore` | `ridge_lambda = β` |
| `joint_ridge_<pooling>[_zscore]` | `joint_ridge` (one ridge on all trace-normalized grouped blocks, total target) | `unitnorm` / `rmsnorm` | `none` / `zscore` | `ridge_lambda = β` (trace-relative per block) |

β is the value picked most often across the 3 seeds in the offline refit, where it was chosen
on 600 held-out train rows by logit-space R² against A's Stage-1 target (no labels). A
production run uses one fixed β. The offline global fit used λ = β · (mean squared norm of
f_B on the support); that scale is stable across supports (≤ 0.5% spread), so it is converted
to an absolute `ridge_lambda`.

| task | support | global β → `ridge_lambda` | block carry β | extra cells (β) |
|---|---|---|---|---|
| mnli | 3400 | 0.1 → 0.58915 | 0.1 | `joint_ridge_rmsnorm` (1.0), `joint_ridge_unitnorm` (1.0) |
| qnli | 3400 | 1.0 → 5.2672 | 0.0001 | `block_ridge_trace_carry_unitnorm_zscore` (1.0) |
| snli | 3400 | 0.1 → 0.625 | 0.0001 | `joint_ridge_unitnorm_zscore` (1.0), `joint_ridge_unitnorm` (0.1) |
| scitail | 3400 | 0.1 → 0.58389 | 0.1 | — |
| sick | 3400 | 10.0 → 74.606 | 10.0 | — |
| rte | 1890 | 1.0 → 5.0474 | 1.0 | `block_ridge_trace_carry_rmsnorm` (1.0) |

## Results the configs correspond to

Offline, cached-feature space, mean over seeds 33/54/89, n = 3400 (rte 1890). "α=1" is what
these configs run; "α on held-out" picks α on held-out rows by accuracy (uses labels).

**BEST** marks the highest α=1 accuracy in the task over every cell tried (5 poolings × up
to 5 preprocessings × 4 fits); picked on test, so its margin over the runner-up is optimistic.

| task | oracle | cell | α=1 | α on held-out |
|---|---|---|---|---|
| mnli | 0.813 | `global_ridge` | 0.7230 | 0.7219 |
| | | `block_ridge_trace_carry` | 0.7143 | 0.7159 |
| | | **BEST** `joint_ridge_rmsnorm` | **0.7298** | 0.7293 |
| | | `joint_ridge_unitnorm` | 0.7294 | 0.7331 |
| qnli | 0.857 | `global_ridge` | 0.8072 | 0.8085 |
| | | `block_ridge_trace_carry` | 0.8181 | 0.8181 |
| | | **BEST** `block_ridge_trace_carry_unitnorm_zscore` | **0.8224** | 0.8206 |
| snli | 0.873 | `global_ridge` | 0.7748 | 0.7719 |
| | | `block_ridge_trace_carry` | 0.7702 | 0.7720 |
| | | **BEST (launchable)** `joint_ridge_unitnorm_zscore` | **0.7941** | 0.7933 |
| | | `joint_ridge_unitnorm` | 0.7935 | 0.7913 |
| scitail | 0.901 | **BEST** `global_ridge` | **0.8880** | 0.8883 |
| | | `block_ridge_trace_carry` | 0.8787 | 0.8828 |
| sick | 0.848 | **BEST (launchable)** `global_ridge` | **0.7857** | 0.8154 |
| | | `block_ridge_trace_carry` | 0.7665 | 0.8052 |
| rte | 0.666 | `global_ridge` | 0.6479 | 0.6493 |
| | | `block_ridge_trace_carry` | 0.6399 | 0.6546 |
| | | **BEST** `block_ridge_trace_carry_rmsnorm` | **0.6667** | 0.6747 |

Two task bests are not launchable, because steer_text does not implement them:
- snli: joint ridge with **centered** unit-norm pooling + z-score, 0.7954 (+0.13 points over
  the launchable cell, ~2 test examples);
- sick: global_ridge on f_B **centered with its top principal direction removed**, 0.7935
  (sick at α = 1 is unstable across seeds; read its α-on-held-out column).

Caveats:
- A production run is close to, not identical with, these numbers: it uses one fixed β, and
  `_random_sample` draws `randperm(N)[:n]` while the offline curves drew `randperm(N)[600:600+n]`.
  Joint ridge is the most sensitive to the draw (mnli 0.714–0.743 across 3400-row draws;
  §9 of the doc) — it wins on average, not on every draw.
- rte has 249 test examples (1 example = 0.4 points); its ranking is noise.

## Validated

Three of these configs ran end to end at seed 33 (snli `joint_ridge_unitnorm_zscore`, qnli
`block_ridge_trace_carry_unitnorm_zscore`, mnli `joint_ridge_rmsnorm`): live test accuracy
equals the cached Stage-2 accuracy exactly (0.7839, 0.8244, 0.7144), and an offline refit on
the same support reproduces each to the digit.
