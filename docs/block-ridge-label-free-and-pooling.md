# block_ridge on the ntk arm: target strategies, the α puzzle, label-free support and block pooling

This continues `docs/block-ridge-regression-diagnostics.md`. That report ended with block_ridge
as an interpolator: every intermediate block's ridge fits its support exactly, and the fix
tried there (trace-scaled λ, joint fit) only tied `global_ridge`. This report follows the
investigation from a review of new Stage-2 target strategies, through why the correction
scale α matters so much, to the two changes that finally let a block-based Stage 2 beat
`global_ridge`: a **label-free support larger than the block dimension**, and **normalizing
tokens before pooling** B's blocks. It ends with the production options that implement them.

Setting everywhere unless stated: t5-base ntk checkpoints (A) → t5-large encoder (B),
`model_kind=encoder_classification`, `feature_regime=linear`, the grid's nearest-mean B heads,
Stage-1 λ = 1, tasks mnli/qnli/snli/scitail/sick/rte, seeds 33/54/89. "Oracle" is Stage 1
evaluated with A's true test delta. Accuracies are test accuracies in cached-feature space
unless marked "live"; every live check in this report matched cached space exactly.

Scripts live in `scripts/block_ridge_experiments/`; their outputs in `$BR_LABELFREE_ROOT`
(default `/work/tesi_pmoriello/claude_debug_block_targets`). §11 lists how to rerun each step.

---

## 1. Review of the new block_ridge target and weighting strategies

**Idea.** `steer_text.prepare` gained two options for block_ridge's Stage 2:
`block_ridge_target_strategy` (`reuse_logitmap`: the Stage-1 map applied to each block's
delta, the original; `blockwise_logitmap`: a separate Stage-1 map fit per block;
`last_only`: every block regresses the full Stage-1 target) and
`block_residuals_weighting_strategy` (`identity`; `mean`: each block's coefficients × 1/L).

**Hypothesis.** The implementation matches that description, and the default reproduces the
grid's block_ridge.

**Experiment.** `strategy_harness.py` runs the real `text_rebase` entrypoint on rte, fs200,
seed 33, wraps `prepare()` and fits all 6 combinations, checking each one's targets and
weighted coefficients against an independent implementation of the description, and the
default against the grid cell's recorded Stage-2 accuracy.

**Result.**

| target \ weighting | identity | mean |
|---|---|---|
| reuse_logitmap | 0.5863 (= grid run) | 0.6145 |
| blockwise_logitmap | 0.4980 | 0.4940 |
| last_only | 0.5863 | 0.5904 |

Targets and weights matched the reference exactly for every combination. Four defects were
found and fixed:

1. `blockwise_logitmap` assigned each block's Stage-1 map to `logit_map`, so the artifacts'
   `stage1_logit_map` ended up holding the output block's map (61× off the true one).
2. The default was `blockwise_logitmap`, which would have silently changed every existing
   block_ridge config and grid cell; the default is now `reuse_logitmap`.
3. A misspelled strategy crashed with `UnboundLocalError`; both options are now validated.
4. `stage2_state` did not record the strategies or weights; it now does.

Two semantic findings, left as they are:

- **`blockwise_logitmap` counts the base gap once per block.** Its residual for block b is
  `(f_A + δ_b)W_Aᵀ − f_B W_Bᵀ = G + δ_b W_Aᵀ`, with `G = f_A W_Aᵀ − f_B W_Bᵀ`. Each
  support's `δ_b[S]` has full row rank (400 of 400 here), so the pseudo-inverse reproduces
  that residual exactly: summed over 13 blocks the targets are `13·G + δW_Aᵀ` (132% off the
  Stage-1 target), and with `mean` weighting `G + δW_Aᵀ/13` (the finetuning signal 13×
  too small). A consistent version would fit block b against `δ_b W_Aᵀ + G/L`.
- **`mean` weighting is α = 1/L.** The weight is uniform and applied after the fit, so it
  only rescales the output; it is redundant with an α search.

## 2. Why do `reuse_logitmap` and `last_only` tie on rte?

**Hypothesis.** Either a bug makes the two fits identical, or a structural reason makes
them nearly so.

**Experiment.** `dump_prepare_inputs.py` (a `text_rebase` entrypoint that stops at
`prepare()`) saves heads, `mask_class` and labels per task; `target_strategies.py` refits both
strategies from the cache on 6 tasks × 3 seeds and compares their test predictions.

**Result.** `reuse` beats `last_only` in 15 of 18 cells; the rte seed-33 tie is a 9–9
split of the 18 test examples where the two disagree (argmax agreement 0.928 on 249 examples,
two classes). Structurally, `last_only`'s correction is ≈ 12–13× `reuse`'s and nearly
parallel to it (cosine 0.93–0.97): with interpolating blocks, `last_only` gives every block
the whole target and `reuse` gives each a slice summing to it. Because the correction
dominates B's decision (§3), the 13× scale rarely changes the argmax.

**Conclusion.** Coincidence, not a bug.

## 3. Why does α matter so much?

**Idea.** Accuracy curves over α plateau from α ≈ 1/16 and are best at α ≈ 1/32 for
block_ridge — as if the correction swamped B's own signal.

**Hypothesis.** Either the correction is too large, or B's decision signal is too small.

**Experiment.** `alpha_mechanism.py`: B's logits split into class-shared and class-specific
parts; decision margins; oracle vs Stage-2 corrections along α; where the flips happen.

**Result.**

| task | median B margin | oracle margin shift / B margin | Stage-2 margin shift / B margin | Stage-2 vs oracle R² |
|---|---|---|---|---|
| mnli | 0.019 | 55× | 50× | 0.06 |
| qnli | 0.015 | 91× | 84× | 0.24 |
| rte | 0.010 | 106× | 159× | −1.45 |
| scitail | 0.028 | 136× | 120× | −0.06 |
| sick | 0.050 | 55× | 52× | 0.53 |
| snli | 0.016 | 94× | 80× | 0.37 |

- 99.9–100% of the energy in B's logits is common to all classes: the plain nearest-mean
  head's rows are nearly parallel (cosine 0.94–0.996 between class centroids).
- Stage 1 is built to override B: through `pinv(W_B)` its correction is
  `(A_logits − B_logits)/(1+λ)`, so at α = 1 the corrected logits are the midpoint of A's
  and B's. The oracle confirms it helps: mnli 0.507 → 0.714 already at α = 1/64, flat at
  ≈ 0.817 from α = 1/4.
- Stage 2's prediction has the right size but is a weak guess at the oracle, so once it
  overrides B, accuracy falls to what the guess alone achieves. Small α only flips
  examples B was unsure of (net fixes by B-margin quartile, qnli at α = 1/16: +74, +12,
  +10, −4).

**Conclusion.** α trades trust in Stage 2 against B. Its useful range sits two orders of
magnitude below 1 because B's head has near-zero margins, not because the correction is
too large.

## 4. A centered nearest-mean head for B

**Hypothesis.** Removing the shared direction from B's head gives B margins the correction
has to compete with, and moves α to a sensible scale.

**Experiment.** `build_nearest_mean_head.py --center` builds
`c_k = normalize(μ_k − μ̄)` with bias `b_k = −c_k·μ̄` (exactly cosine nearest-mean on
centered features, still an affine head). `centered_head.py` refits Stage 1/2 offline with
each head; one live `text_rebase` run checks it.

**Result** (mean over 6 tasks × 3 seeds, few_shot 200; "tuned α" picked on held-out rows):

| B head | B alone | median margin | oracle | global, tuned α | block, α=1 | block, tuned α |
|---|---|---|---|---|---|---|
| plain | 0.6575 | 0.023 | 0.8262 | 0.7417 | 0.6717 | 0.7073 |
| centered | 0.6543 | 0.265 | 0.8233 | 0.7374 | 0.6823 | 0.7069 |
| centered + bias fix | 0.6543 | 0.265 | 0.8268 | 0.7417 | 0.6824 | 0.7051 |

- Centering removes the shared logit component (99.9% → 0–3%) and raises margins ~11×;
  the tuned α for block_ridge moves from 1/32–1/512 to ≈ 1/4.
- **Stage 1 leaves B's bias out of its residual.** It fits `(f_A+δ)W_Aᵀ − f_B W_Bᵀ`, so with
  a biased B head the corrected logits end `b_B/2` off the intended midpoint. The plain head
  has no bias; the centered head's bias gaps (0.34–1.14) match its margins. The consistent
  correction adds `α·p_B·(b_A − b_B)/(1+λ)` as a separate term (a linear map of δ cannot
  produce a constant). A's own bias is negligible (gaps ≤ 0.06): A's trained head keeps its
  class offset in its weights instead (`W_A·μ_A` spreads 0.17–3.8, rows nearly orthogonal
  to A's mean feature). The fix recovers the oracle (0.8233 → 0.8268) and snli
  (oracle 0.851 → 0.866, global 0.704 → 0.735).
- The bias is required: dropping it collapses B (qnli 0.516, scitail 0.408, snli 0.356).

**Conclusion.** Centering rescales how much B counts; α was already doing that job, so the
best accuracy after tuning α does not change. Not adopted; the builder option stays
available, and the bias fix is not in `steer_text`.

## 5. A label-free support larger than the blocks' dimension

**Idea.** Neither Stage 1 nor Stage 2 uses labels: Stage 1 fits A's logits minus B's, Stage 2
fits Stage 1's targets. Labels only balance classes when drawing `few_shot`. So the support
can be any unlabelled inputs — and the ntk cache has 4000 train rows (rte 2490), against the
2048 dimensions of a concat-grouped B block that no fs ≤ 500 run ever exceeded.

**Hypothesis.** Past n = 2048 the intermediate blocks stop interpolating, and block_ridge
improves.

**Experiment.** `support_curve.py`: per seed, shuffle the train rows (no labels), hold out
600 as V, and fit on nested supports n ∈ {600, …, 3400}. Penalties are picked on V by
logit-space R² against A's Stage-1 target (label-free). Ridge is solved in kernel form from
one eigendecomposition per block; a guard checks it against production `_fit_block_ridge`
(8e-11). Variants: block_ridge as run (λ = 1), trace λ, trace λ + smoothed-residual carry
(ρ = 1), `sum_avg` grouping, joint ridge, global_ridge.

**Result** (mean over 5 tasks × 3 seeds, rte excluded for its 1890-row cap; oracle 0.858):

| n | as run | trace λ | trace λ + carry | sum_avg + trace | joint | global |
|---|---|---|---|---|---|---|
| 600 | 0.674 | 0.707 | 0.738 | 0.704 | 0.737 | 0.745 |
| 1200 | 0.679 | 0.740 | 0.764 | 0.735 | 0.765 | 0.770 |
| **2000** | **0.592** | 0.756 | 0.780 | 0.751 | 0.782 | 0.787 |
| 2600 | 0.690 | 0.760 | 0.777 | 0.754 | 0.781 | 0.785 |
| 3400 | 0.738 | 0.773 | 0.790 | 0.767 | 0.789 | 0.796 |

- **Double descent.** block_ridge as run *drops* to 0.592 at n = 2000, just under 2048
  (test R² −1.4 to −2.7), then recovers once n > d — the interpolation mechanism seen directly.
- **The carry finally works.** With trace λ large enough (β = 0.1–1) to stop exact fits,
  each block leaves a leftover and the carry passes it to f_B. That is also why
  `smoothed_residual` was identical to `independent` in the whole grid: an exact fit leaves
  nothing to carry.
- `sum_avg` grouping is consistently ~0.5 points below concat.
- Every method keeps improving with n; no block variant beats global_ridge.

## 6. Is the geometry of the pooled features the problem?

**Idea.** Mean-pooled intermediate blocks look alike for every input.

**Experiment.** `pooling_collapse.py` measures, per block (mnli, snli, 4000 train rows):

| | blocks 0–12 | blocks 14–23 | f_B |
|---|---|---|---|
| share of each example's energy that is the dataset mean | 0.91–0.96 | 0.61–0.87 | 0.48–0.66 |
| share of remaining variance on one principal direction | 0.09–0.69 | 0.49–0.98 | 0.06–0.08 |
| energy in the 5 largest of 1024 channels | 0.44–0.53 | 0.48–0.72 | 0.12–0.15 |
| share of remaining variance explained by class | 0.2–0.6% | 0.02–0.4% | 0.6–1.1% |
| cosine with the previous block, same example | 0.98–0.999 | 0.89–0.998 | — |

**Hypothesis.** Reconditioning the pooled features (centering, per-channel z-score,
dropping the top principal direction, per-example LayerNorm) lets the ridge use them.

**Experiment.** `preprocess_curve.py`: the same label-free protocol with each
preprocessing (support statistics only, unpenalized intercept when centered).

**Result.** No preprocessing moves results by more than ~1 point; the best block cell at
n = 3400 is z-score + joint, 0.793, against global 0.796. Per-example LayerNorm of the
pooled vector hurts (up to −3.5).

**Conclusion.** Trace-scaled ridge already copes with scale; what limits the blocks is what
mean pooling keeps, not how it is shaped afterwards.

## 7. Normalize each token before pooling

**Hypothesis.** A few tokens with very large norms dominate each block's mean; normalizing
every token first gives each token the same weight and recovers information pooling lost.

**Experiment.** `collect_b_pooled.py` (a `text_rebase` entrypoint, B forward only) re-pools
every block four ways: `unitnorm` (token / ‖token‖), `rmsnorm` (B's own final RMSNorm, the
way f_B is pooled), and both after subtracting the block's mean token over the train split.
Checks: plain mean pooling matches the cache (≤ 2e-5) and RMSNorm pooling of block 23
equals f_B exactly. `preprocess_curve.py --b-pooled … --pooling …` refits.

**Result** (mean over 5 tasks × 3 seeds, α = 1, no preprocessing):

| n | fit | mean | unitnorm | rmsnorm | center + unitnorm | center + rmsnorm |
|---|---|---|---|---|---|---|
| 1500 | joint | 0.771 | **0.779** | 0.766 | 0.751 | 0.759 |
| 1500 | global | 0.772 | 0.772 | 0.772 | 0.772 | 0.772 |
| 3400 | trace λ | 0.773 | 0.781 | 0.780 | 0.766 | 0.775 |
| 3400 | trace λ + carry | 0.790 | 0.791 | 0.787 | 0.785 | 0.784 |
| 3400 | joint | 0.789 | **0.800** | 0.794 | 0.781 | 0.783 |
| 3400 | global | 0.796 | 0.796 | 0.796 | 0.796 | 0.796 |

- Joint ridge on unit-normalized blocks is the first block-based fit to beat global_ridge:
  paired over 15 task-seeds at n = 3400, +0.44 ± 0.29 points at α = 1 (9/15 wins) and
  +0.64 ± 0.26 with α picked on V (12/15). It was picked from ~20 cells, so part of the
  margin is selection.
- Per task (n = 3400, α = 1, joint unitnorm vs global): snli +1.9 (every seed), qnli +1.1,
  mnli +0.6 (every seed); scitail −1.1 (every seed), sick −0.4 (a tie with α tuned; one seed
  collapses at α = 1 for every method), rte within noise (249 test examples).
- Normalization removes the dominant direction (upper blocks: top-direction share
  0.5–0.98 → 0.09–0.17) and raises the class share (snli block 17: 0.0% → 1.1%). The
  shared mean (~90%) and the large channels remain.
- RMSNorm's learned gain does not help; under z-score, unitnorm and rmsnorm are identical
  (z-score divides out any per-channel gain).
- Centering before normalizing hurts: the pooled vectors re-collapse (95–99% shared mean in
  upper blocks), plausibly because the large components sit on few tokens, so subtracting
  the mean token leaves most tokens with the same deviation.
- On scitail the block fits have *higher* test R² but lower accuracy than global: the extra
  variance they explain is not along the decision.

## 8. Production implementation

`steer_text.prepare` gained, all opt-in with defaults that reproduce old runs exactly:

- **`stage_2_strategy="joint_ridge"`**: one ridge on all grouped blocks, each divided by the
  root of its support mean squared norm and concatenated, fit to the total Stage-1 target;
  `ridge_lambda` is a per-block trace-relative penalty (same units as trace λ). Requires
  `feature_regime="linear"` (B's blocks are collected there). Coefficients are rescaled per
  block so prediction reuses block_ridge's correction path.
- **`block_pooling`** (`mean` | `unitnorm` | `rmsnorm`): implemented in `_TextBlockCapture`,
  used identically at fit time and in the live correction hook. `rmsnorm` uses each block's
  own stack's final norm (`block_final_norms`). Non-mean poolings are recollected from B
  alone (no jvps) and cached as `features_B_blocks_pool-<name>.pt` beside the split's other
  files, written atomically so concurrent seeds cannot read a partial file.
- **`block_feature_preprocessing`** (`none` | `zscore`): support statistics, an
  unpenalized intercept, folded into the coefficients and one bias vector.
- Any other strategy rejects the block options instead of ignoring them.

Tests (`tests/test_text_rebase.py`): the defaults reproduce the plain per-block fit
exactly; joint ridge equals an independent concatenated ridge with and without z-score;
z-score folding is exact in both block modes; pooled features are cached apart and pooled
identically live; the new rejections.

Validation: three real runs (snli joint unitnorm+zscore, qnli carry unitnorm+zscore, mnli
joint rmsnorm; seed 33) — live test accuracy equals cached-space Stage-2 accuracy exactly
(0.7839, 0.8244, 0.7144), and an offline refit with production's exact support reproduces
them to the digit.

## 9. How much does the support draw matter?

**Idea.** The validation runs were 1–1.5 points below the offline means on mnli and snli,
while the offline seed spread was 0.2–0.4.

**Experiment.** `support_sensitivity.py`: joint ridge and global on four supports per seed —
the first 3400 rows of the seeded permutation (production's `_random_sample`), the last 3400
(the offline curves), a middle 3400, and all 4000.

**Result.**

| | mnli joint | mnli global | snli joint | snli global |
|---|---|---|---|---|
| 9 supports of 3400 rows (mean) | 0.730 | 0.720 | 0.790 | 0.776 |
| range over those supports | 0.714–0.743 | 0.711–0.729 | 0.784–0.796 | 0.772–0.780 |
| all 4000 rows | 0.735 | 0.719 | 0.792 | 0.779 |

**Conclusion.** Joint ridge depends more on which rows it sees than global does, but wins
on average and in all 18 draws (one mnli draw by only 0.06 points). Production's seed-33 draw was the worst
mnli one. Production runs fix β, so they need no held-out rows and can use every row.

## 10. Configs

`configs/block_ridge_ntk_labelfree/` holds fully resolved configs (β from the offline picks,
label-free support of 3400 rows, α = 1) and `launch.sh`, which submits every config × 3
seeds; its `README.md` marks the best cell per task. Best per task at α = 1 (picked on test
among ~100 cells, so optimistic): mnli joint + rmsnorm 0.730; qnli carry + unitnorm + zscore
0.822; snli joint + unitnorm (+ zscore) 0.794; scitail global 0.888; sick global 0.786
(0.794 with offline-only preprocessing of f_B); rte carry + rmsnorm 0.667 (noise).
`launch.sh` depends on `scripts/slurm/submit_text_rebase.sh`, `resolve_config.py` and
`text_rebase_job.sbatch`.

## 11. Reproducing

```bash
export BR_LABELFREE_ROOT=/work/tesi_pmoriello/claude_debug_block_targets   # outputs
# heads/labels bundle per task (GPU, via the grid's ntk overrides, eval_source_finetuned=false)
ENTRYPOINT=scripts.block_ridge_experiments.dump_prepare_inputs RESULTS_ROOT=$BR_LABELFREE_ROOT/dump \
  scripts/slurm/submit_text_rebase.sh dump_<task> configs/text_rebase_t5enc_steer.json <grid overrides> eval_source_finetuned=false
# §1  python scripts/block_ridge_experiments/strategy_harness.py <config.json> <log-dir>
# §2  python scripts/block_ridge_experiments/target_strategies.py $BR_LABELFREE_ROOT/dump --out f.json
# §3  python scripts/block_ridge_experiments/alpha_mechanism.py $BR_LABELFREE_ROOT/dump
# §4  python -m scripts.build_nearest_mean_head ... --center ; python scripts/block_ridge_experiments/centered_head.py <dump> <heads-dir>
# §5  python scripts/block_ridge_experiments/support_curve.py $BR_LABELFREE_ROOT/dump --task <task> --out support_curve/<task>.json
#     python scripts/block_ridge_experiments/support_curve_table.py
# §6  python scripts/block_ridge_experiments/pooling_collapse.py <task>
#     python scripts/block_ridge_experiments/preprocess_curve.py $BR_LABELFREE_ROOT/dump --task <task> --out preprocess_curve/<task>.json
#     python scripts/block_ridge_experiments/preprocess_table.py
# §7  ENTRYPOINT=scripts.block_ridge_experiments.collect_b_pooled RESULTS_ROOT=$BR_LABELFREE_ROOT/b_pooled \
#       scripts/slurm/submit_text_rebase.sh bpool_<task> configs/text_rebase_t5enc_steer.json <grid overrides> eval_source_finetuned=false
#     python scripts/block_ridge_experiments/preprocess_curve.py $BR_LABELFREE_ROOT/dump --task <task> --preps none zscore \
#       --b-pooled $BR_LABELFREE_ROOT/b_pooled/bpool_<task>/<task>_b_pooled.pt --pooling <pooling> --out pool_curve/<task>_<pooling>.json
#     python scripts/block_ridge_experiments/pool_table.py ; python scripts/block_ridge_experiments/pool_geometry.py <task>
# §9  python scripts/block_ridge_experiments/support_sensitivity.py mnli rmsnorm 0 1.0 0.1
```

The curve scripts are CPU-only (8 cores, 48 GB for n = 3400; ~15 min per task); the
entrypoints need one GPU for a few minutes per task.

## 12. Open questions

- **Other pooling sources inside the block.** Any per-token linear map (Q/K/V, W_O, the FFN
  output projection) commutes with mean pooling and adds no information for a linear
  Stage 2. Points after token mixing or a nonlinearity can: the attention output before
  W_O (`SelfAttention.o`'s input) pools values weighted by how much attention each token
  receives; the FFN hidden activations are nonlinear (4096-d per layer). Next experiment.
- Per-block targets (`reuse_logitmap`) are what separate block_ridge from joint ridge;
  joint wins where it wins by not using them.
- The gap to the oracle is still ≈ 6 points on average and keeps closing with n.
