# Why `block_ridge` underperforms: a diagnostic investigation

On the t5-base → t5-large encoder grid ([`t5-encoder-steer-grid.md`](t5-encoder-steer-grid.md)),
`steer_text`'s `block_ridge` stage 2 consistently trails `global_ridge` and
`global_mlp`. This document records how we found out why. It follows the
investigation in the order it happened. Each step is written up as

- **idea**: what prompted the step,
- **hypothesis**: what it could tell us,
- **experiment**: how it was run,
- **expected outcomes**: what each possible result would mean,
- **outcome**: what we actually saw.

The mechanics of a single run are in
[`t5-encoder-steer-flow.md`](t5-encoder-steer-flow.md). Every number here can be
regenerated with the commands in [Reproducing](#reproducing).

**Short version.** `block_ridge` loses for three stacked reasons, each isolated
by its own experiment:

1. **A single $\lambda$ is effectively zero for the deep blocks.** The
   un-normalized T5 residual stream has activation scales that span about five
   orders of magnitude. With $\lambda=1$ every intermediate block fits its
   support exactly and does worse than the mean on test.
2. **Fitting each block separately compounds the errors.** Thirteen independent
   ridges each chase their own block's slice of the target, and their errors add
   up in the sum.
3. **B's intermediate blocks carry no extra usable signal.** With both problems
   fixed, a joint fit on all blocks only ties `global_ridge`, from 200 up to 500
   support examples per class.

The correction scale $\alpha$ changes none of this. Above $\alpha\approx0.1$ the
correction swamps B's own logits, and every predictor's accuracy stops moving.

---

## 0. Setup and notation

**Models.**
- The source $A$ is `t5-base`. Its task delta was fine-tuned in *tangent* mode:
  the NTK checkpoints, `checkpoints_t5_converted/t5base_6text_noreg_ntk/`.
- The target $B$ is `t5-large`, never fine-tuned. Its head is nearest-class-mean
  centroids (see [`t5-encoder-nearest-mean-head.md`](t5-encoder-nearest-mean-head.md)).
- Pooled features are masked token means, $f_A\in\mathbb{R}^{768}$ and
  $f_B\in\mathbb{R}^{1024}$.
- $B$'s head is $W_B\in\mathbb{R}^{C\times 1024}$ with bias $b_B$. The task only
  scores the head rows $\mathcal{K}$ (`mask_class`).

**$A$'s delta in the linear regime.** Because $A$ was trained on its linearization
around the pretrained weights $\theta_0$, its feature delta on input $x$ is the
Jacobian–vector product

$$
\delta_A(x) \;=\; J_{\theta}f_A(x;\theta_0)\,(\theta-\theta_0),
$$

computed with forward-mode `jvp` (`LinearizedModule`,
`steer_text._collect_linear_split`). The parameters are partitioned into 13
blocks, $b=0,\dots,11$ for the encoder blocks and $b=12$ for the trailing
output parameters. Masking all but one block gives

$$
\delta_A^{(b)}(x) = J_{\theta_b} f_A(x;\theta_0)\,(\theta_b-\theta_{0,b}),
\qquad
\sum_{b=0}^{12}\delta_A^{(b)}(x) = \delta_A(x),
$$

exactly, because the linearization is linear in the parameter delta. On the
cached mnli features the relative error of this identity is $7\times10^{-7}$.

**Stage 1: the head-aware logit map.** Take a class-balanced support set
$S$ with $n_S = |\mathcal{K}|\cdot\texttt{few\_shot}$ rows (`_few_shot`). Build
the logit residual between fine-tuned $A$ and $B$,

$$
R = (F_A + \Delta_A)\,W_A^\top \;-\; F_B\,W_B^\top \;\in\mathbb{R}^{n_S\times C},
$$

and regress it on $A$'s delta (`_stage1_projection`):

$$
L = \frac{1}{1+\lambda_1}\,\big(\Delta_A^{+}R\big)^\top \in \mathbb{R}^{C\times 768},
\qquad \lambda_1 = \texttt{stage1\_lambda} = 1 .
$$

The logit correction $L\,\delta_A(x)$ is pulled back into $B$'s feature space
with $P_B = W_B^{+}\in\mathbb{R}^{1024\times C}$:

$$
t(x) = P_B\,L\,\delta_A(x), \qquad t^{(b)}(x) = P_B\,L\,\delta_A^{(b)}(x),\qquad \sum_b t^{(b)} = t .
$$

$W_B$ has full row rank ($C\le 1024$), so $W_B P_B = I_C$ and

$$
W_B\, t(x) = L\,\delta_A(x).
$$

**The head sees the correction only through $C$ numbers.** The target lives in
the $\le C$-dimensional row space of $W_B$. This fact drives the choice of
metric in §1.

**Prediction, and the three recorded accuracies.** With a correction $c(x)$ and a
scale $\alpha$ (`apply_correction`), the prediction is

$$
\hat y(x) = \arg\max_{k\in\mathcal{K}} \big[\,W_B\,(f_B(x) + \alpha\,c(x)) + b_B\,\big]_k .
$$

The run records three test accuracies, all with $\alpha=1$:

- `stage0_test_acc` uses $c=0$ ($B$ alone).
- `stage1_test_acc` uses $c=t$. This is an **oracle**: it uses $A$'s true delta
  on the test example, which is never available at deployment.
- `stage2_test_acc` uses $c=\hat t$, the Stage-2 prediction from $B$'s features
  alone.

**Stage 2: `global_ridge` vs `block_ridge`.** `global_ridge` fits one ridge on the
final features:
$\hat t(x) = f_B(x)^\top C$ with
$C=\arg\min\|F_{B,S}C - T_S\|^2 + \lambda\|C\|^2$.

`block_ridge` uses $B$'s per-block pooled activations $h_\ell(x)$,
$\ell = 0,\dots,23$, with $h_{24}=f_B$. The 24 residual blocks are grouped by
concatenation onto $A$'s 12 (`_group_blocks_concat`):

$$
X_g = [\,h_{2g}\;\;h_{2g+1}\,]\in\mathbb{R}^{2048}\;(g=0..11),\qquad X_{12} = f_B\in\mathbb{R}^{1024}.
$$

It then fits one ridge per block against that block's own target slice
(`_fit_block_ridge`, mode `independent`), and sums the predictions:

$$
C_b = \arg\min_{C}\;\|X_{b,S}\,C - T^{(b)}_S\|_F^2 + \lambda\|C\|_F^2,
\qquad
\hat t(x) = \sum_{b=0}^{12} x_b^\top C_b .
$$

Since $n_S\le 1500 < 2048$, `_ridge` takes the dual branch. With the block's Gram
(kernel) matrix $K_b = X_{b,S}X_{b,S}^\top$:

$$
C_b = X_{b,S}^\top\,(K_b + \lambda I)^{-1}\,T^{(b)}_S .
$$

The grid ran $\lambda_1=\lambda=1$ and $\alpha=1$ without a sweep, deliberately.
Tuning was not the question.

---

## 1. Regression diagnostics: does the ridge fit, and does it generalize?

**Idea.** The three accuracies say *that* `block_ridge` fails, but not *how*. On
mnli, fs200, seed 33 they are 0.507 → 0.817 (oracle) → 0.562 (block_ridge). The
correction is clearly achievable, yet the fitted regressor recovers less than a
fifth of the oracle's gain. Stage 2 is a regression problem, so we can measure
it like one. There was no existing regression metric: the pipeline only ever
reported accuracies, and nothing measured the fit on the train side.

**Hypothesis.** The failure is visible as poor regression quality. Where it
shows up tells us which failure it is:

- **Train fit:** is the ridge able to fit its own support?
- **Held-out fit:** does it generalize to the unselected train rows, and to test?
- **Hard vs easy:** which examples does it fail on? We first considered testing
  on a different downstream dataset. We dropped that because no ground-truth
  target exists off-task. Instead we stratify the same task's test set.

**Experiment.**

- **Real artifacts, not a reimplementation.** The grid drops `block_ridge`'s
  fitted objects to save disk (~218 MB a run), and `grid.py submit` only submits
  *missing* cells. So we reran the grid cells with artifacts kept, using
  `scripts/slurm/submit_block_ridge_diag.py`. It imports `grid.py`'s own
  `overrides_for` and `resources_for`, so the resolved config matches the grid's
  exactly apart from the artifact directory. It writes to a separate root, so no
  grid result can be overwritten, and the warm feature cache makes each rerun a
  refit only.
- **Validation gate.** `scripts/diagnose_block_ridge.py` rebuilds stage0/1/2
  test accuracy from the artifacts and the cache. It refuses to report unless
  all three match the run's recorded values, and the original grid cell, to
  within one example. It also refits `global_ridge` on the same support and
  checks it against the `global_ridge` twin cell. All 18 fs200 runs pass.
- **Metric: $R^2$ in logit space.** For targets $Y$ and predictions $P$ over $n$
  rows and $d$ output dimensions:

  $$
  R^2_{\text{vw}} = 1-\frac{\sum_{j}\sum_i (Y_{ij}-P_{ij})^2}{\sum_j\sum_i (Y_{ij}-\bar Y_j)^2},
  \qquad
  R^2_{\text{ua}} = \frac{1}{d}\sum_j \Big(1-\frac{\sum_i (Y_{ij}-P_{ij})^2}{\sum_i (Y_{ij}-\bar Y_j)^2}\Big).
  $$

  These are sklearn's `variance_weighted` and `uniform_average`. The first is the
  fraction of total target variance explained. The second stops a few
  high-variance dimensions from hiding many badly fit ones.

  By §0 the head only sees $W_{B,\mathcal{K}}\,c$, so the headline $R^2$ is
  computed on $Y W_{B,\mathcal{K}}^\top$ vs $P W_{B,\mathcal{K}}^\top$.
  Feature-space $R^2$ is reported alongside it. We also report per-example
  cosine and the norm ratio $\|p\|/\|y\|$. A ratio well below 1 would indicate
  ridge's shrinkage toward zero.
- **Three row sets:**
  - the support $S$ (train fit),
  - the unselected train rows $U=\text{train}\setminus S$ (same split, never fit),
  - test.

  Each set is scored per block ($x_b^\top C_b$ vs $t^{(b)}$) and summed
  ($\hat t$ vs $t$).
- **Hard vs easy (test only):**
  - the $2^3$ cross-tab of per-example correctness for stage0, stage1 and stage2;
  - accuracy and $R^2$ per quartile of the oracle's logit-correction norm
    $\|L\,\delta_A(x)\|$ (how big a correction the example needs);
  - the same per quartile of $B$'s margin
    $m(x)=s_y(x)-\max_{k\ne y}s_k(x)$, where $s=W_Bf_B+b_B$.

**Expected outcomes.**

| pattern | meaning |
|---|---|
| train $R^2$ low | underfitting: too little capacity, or $\lambda$ too strong for the feature scale |
| train $R^2\approx1$, test $R^2\ll$ train | overfitting: the ridge interpolates its support |
| test $R^2$ high, stage2 accuracy still low | the regression is fine; the problem is Stage 1 or the head geometry |
| norm ratio $\ll 1$ on large-correction quartiles | L2 shrinkage specifically eats large corrections |

**Outcome.** mnli, fs200, seed 33, summed correction:

| rows | model | feature $R^2_{\text{vw}}$ | logit $R^2_{\text{vw}}$ | logit $R^2_{\text{ua}}$ | cosine | $\|p\|/\|y\|$ |
|---|---|---|---|---|---|---|
| support | block_ridge | 1.000 | **1.000** | 1.000 | 1.000 | 1.00 |
| support | global_ridge | 0.859 | 0.867 | 0.859 | 0.963 | 0.91 |
| unselected train | block_ridge | 0.122 | 0.091 | 0.054 | 0.734 | 0.98 |
| unselected train | global_ridge | 0.451 | 0.449 | 0.409 | 0.832 | 0.87 |
| test | block_ridge | 0.072 | **0.057** | 0.021 | 0.730 | 1.00 |
| test | global_ridge | 0.380 | 0.379 | 0.353 | 0.830 | 0.89 |

Per block, blocks 0–11 each fit their support exactly ($R^2=1.000$), and every
one scores **negative** test $R^2$, from −0.18 to −0.95. Only block 12, which is
$f_B$ itself, generalizes (0.935 on the support, 0.729 on test). Its target
share is negligible, though. Blocks 9–11 carry about 60% of the target energy,
$\sum_i\|t^{(b)}_i\|^2$.

The result is the same across tasks. Mean over 3 seeds, fs200, logit $R^2_{\text{vw}}$:

| task | stage0 | oracle | block_ridge | global_ridge | block train $R^2$ | block test $R^2$ | global train $R^2$ | global test $R^2$ | worst block test $R^2$ | fixable-but-missed |
|---|---|---|---|---|---|---|---|---|---|---|
| mnli | 0.507 | 0.815 | 0.559 | 0.655 | 1.000 | 0.040 | 0.854 | 0.310 | −1.28 | 0.172 |
| sick | 0.762 | 0.848 | 0.747 | 0.798 | 1.000 | 0.451 | 0.911 | 0.663 | −1.75 | 0.056 |
| rte | 0.627 | 0.664 | 0.577 | 0.616 | 1.000 | −1.503 | 0.747 | 0.043 | −3.98 | 0.076 |
| scitail | 0.774 | 0.902 | 0.780 | 0.826 | 1.000 | 0.310 | 0.890 | 0.624 | −7.09 | 0.052 |
| qnli | 0.684 | 0.858 | 0.712 | 0.761 | 1.000 | 0.244 | 0.866 | 0.473 | −2.00 | 0.094 |
| snli | 0.591 | 0.871 | 0.656 | 0.734 | 1.000 | 0.315 | 0.853 | 0.607 | −1.98 | 0.133 |

"Worst block" is the worst among blocks 0–11. "Fixable-but-missed" is the test
fraction where stage0 is wrong, stage1 right and stage2 wrong. On sick and rte,
block_ridge ends up below $B$'s uncorrected baseline.

**The hard-vs-easy view says it is not shrinkage.** On mnli seed 33:

- 323 test examples (17.9%) are fixable but missed, against 56 that block_ridge
  fixes and the oracle does not.
- Per correction-norm quartile, block_ridge's logit $R^2$ runs −2.48, −0.92,
  −0.01, 0.22, and its median norm ratio runs 1.37, 1.04, 0.91, 0.73.
- block_ridge trails `global_ridge` in every quartile of both axes.

The predictions are about the right size on average. They are noisy in
direction and magnitude, not biased toward zero.

**Diagnosis: row 2 of the expected-outcomes table, overfitting by interpolation.**
It is also not a train/test shift, because the unselected *train* rows fail just
as badly as test.

## 2. Why does it interpolate? Feature scale against a fixed $\lambda$

**Idea.** The same $\lambda=1$ regularizes `global_ridge` noticeably (train
$R^2\approx0.86$), yet it has no visible effect on the intermediate blocks.
Ridge's effect depends on $\lambda$ *relative to the data's spectrum*.

**Hypothesis.** With $K_b = U\Sigma U^\top$, the dual ridge's in-sample fit is

$$
X_{b,S}\,C_b = K_b\,(K_b+\lambda I)^{-1}T^{(b)}_S
= U\,\mathrm{diag}\!\Big(\frac{\sigma_i}{\sigma_i+\lambda}\Big)\,U^\top\,T^{(b)}_S .
$$

If every eigenvalue satisfies $\sigma_i\gg\lambda$, every factor is about 1. The
fit then reproduces $T^{(b)}_S$ exactly, and the solution becomes the
minimum-norm interpolant, whose predictions on new inputs have high variance.
We expect this for $B$'s intermediate blocks, since T5's un-normalized residual
stream grows with depth. We do not expect it for $f_B$, which passes through the
final LayerNorm.

**Experiment.** Mean row norm and the spectrum of $K_b$ on mnli's 600 support rows.

**Expected outcomes.**
- If $\min_i\sigma_i \gg 1$ for blocks 0–11 but not for block 12, the hypothesis
  is confirmed.
- If the spectra were comparable, interpolation would need another explanation,
  for example target noise.

**Outcome.**

| input | mean $\|x\|$ | $\sigma_{\min}$ | median $\sigma$ | $\sigma_{\max}$ | eigenvalues $<\lambda=1$ |
|---|---|---|---|---|---|
| block 0 | 7.0e2 | 2.6e2 | 6.8e3 | 2.8e8 | 0 / 600 |
| block 5 | 4.3e3 | 3.6e3 | 7.9e4 | 1.1e10 | 0 / 600 |
| block 11 | 5.2e4 | 7.3e4 | 1.3e6 | 1.8e12 | 0 / 600 |
| block 12 $=f_B$ | 2.4 | 0.033 | 0.87 | 1.7e3 | 321 / 600 |

**Confirmed.**
- For block 0 the *weakest* shrinkage factor is $262/263\approx0.996$. Deeper
  blocks are closer still to 1.
- For $f_B$ the median factor is $0.87/1.87\approx0.47$: real regularization.
- Squared scales differ by about $5{,}000\times$ from block 0 to block 11, and by
  about $5\times10^8$ from block 11 to $f_B$.
- $f_B$ is identical to block 12 (`allclose`). `global_ridge` is therefore
  exactly the one well-regularized piece of `block_ridge`, but that piece
  carries almost none of the target.

## 3. Trace-scaled $\lambda$ per block

**Idea.** Make $\lambda$ scale-free per block. Set it proportional to the mean
eigenvalue of the block's Gram matrix:

$$
\lambda_b = \beta\cdot\frac{\operatorname{tr}(K_b)}{n_S} = \beta\cdot\frac{1}{n_S}\sum_{i\in S}\|x_{b,i}\|^2 = \beta\,s_b .
$$

This is equivalent to standardizing each block. Let $\tilde X_b = X_b/\sqrt{s_b}$
and fit with penalty $\beta$. Then

$$
\tilde x^\top \tilde X_S^\top(\tilde X_S\tilde X_S^\top+\beta I)^{-1}T
= \frac{x^\top X_S^\top}{s_b}\Big(\frac{K_b+\beta s_b I}{s_b}\Big)^{-1}T
= x^\top X_S^\top (K_b+\lambda_b I)^{-1}T,
$$

which is the unscaled ridge with $\lambda_b=\beta s_b$. So one value of $\beta$
means the same thing for every block. This is added to `src/` as the opt-in
`_fit_block_ridge(..., regularization_scaling="trace")`, exposed as
`method_params.block_ridge_lambda_scaling`. The default is `"none"`, so existing
fits are unchanged.

**Hypothesis.** Scale-aware regularization stops the interpolation. It should
raise test $R^2$ and accuracy toward `global_ridge`.

**Experiment.** `scripts/block_ridge_experiments/trace_lambda.py`.
- For each run it refits with the production `_fit_block_ridge` on the run's own
  support, Stage-1 map and cache. It first asserts that the unscaled refit
  reproduces the saved coefficients ($<10^{-6}$ relative) and the recorded
  `stage2_test_acc`.
- $\beta$ is chosen per run by accuracy on $U$, and test is only reported. The
  first grid was $\beta\in\{1,0.1,0.01,0.001\}$.

**Expected outcomes.**
- If accuracy rises and approaches `global_ridge`, scale was the whole problem.
- If it rises but a gap remains, scale is one cause among several.
- If it doesn't rise, the hypothesis is wrong.

**Outcome (first grid).**
- $\beta=1$ over-regularizes badly, e.g. mnli 0.396 against 0.559 for the fit as
  run. $\lambda_b$ then equals the mean eigenvalue, while the median eigenvalue
  is about 70× smaller for block 0 and about 2,000× smaller for block 11, so
  most directions are shrunk away.
- Validation chose the smallest $\beta$ in the grid in 13 of 18 runs, so the grid
  was too narrow.

**A natural objection: why not just increase $\lambda$?** Trace scaling already
*is* a large increase on the intermediate blocks. $s_b$ is about $5\times10^5$
for block 0 and $3\times10^9$ for block 11, so even $\beta=10^{-3}$ means
$\lambda_0\approx5\times10^2$ and $\lambda_{11}\approx3\times10^6$, against the
original 1. The only block where it lowers $\lambda$ is $f_B$ ($s\approx6$).

A single larger uniform $\lambda$ cannot fix the problem. Any $\lambda$ that
suits block 11 is heavy shrinkage for block 0, and any $\lambda$ that suits
block 0 is still negligible for block 11. Their scales differ by 5,000×.

**Outcome (grid widened to $\beta\in\{1,\dots,10^{-6}\}$).** fs200, mean test
accuracy over 3 seeds:

| task | as run | $\beta=1$ | 0.1 | 0.01 | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | val-picked | global_ridge |
|---|---|---|---|---|---|---|---|---|---|---|
| mnli | 0.559 | 0.396 | 0.487 | 0.550 | **0.579** | 0.570 | 0.561 | 0.560 | 0.579 | 0.655 |
| sick | 0.747 | 0.620 | 0.724 | 0.761 | **0.772** | 0.768 | 0.755 | 0.747 | 0.771 | 0.798 |
| rte | 0.577 | 0.548 | 0.577 | 0.576 | **0.592** | 0.584 | 0.578 | 0.577 | 0.576 | 0.616 |
| scitail | 0.780 | 0.599 | 0.756 | 0.805 | **0.810** | 0.792 | 0.781 | 0.780 | 0.810 | 0.826 |
| qnli | 0.712 | 0.662 | 0.728 | **0.746** | 0.733 | 0.717 | 0.713 | 0.712 | 0.742 | 0.761 |
| snli | 0.656 | 0.392 | 0.469 | 0.596 | 0.675 | **0.680** | 0.668 | 0.657 | 0.680 | 0.734 |

- The optimum is interior, $\beta\in[10^{-4},10^{-2}]$.
- As $\beta\to10^{-6}$ every task returns *exactly* to the fit as run. $\lambda_b$
  is negligible again, which is a direct check of §2.
- At the validation-picked $\beta$:
  - test logit $R^2$ rises on every task (mnli 0.04 → 0.19, rte −1.50 → 0.19);
  - accuracy gains 2–3 points on five tasks, and rte is flat;
  - a fifth to two-thirds of the gap to `global_ridge` closes on those five tasks
    (mnli 9.6 → 7.6 points, scitail 4.6 → 1.6).

**Middle row of the expected outcomes.** Scale is a real cause, but not the
only one.

## 4. One joint ridge on all blocks

**Idea.** Even when well regularized, `block_ridge` fits 13 regressors
independently, each to its own slice $t^{(b)}$, and then sums them. The error of
the sum is $\sum_b e_b$. Nothing in the fit rewards the errors cancelling, and
each block must predict a component that may be poorly determined by that
block's activations alone.

A joint fit instead minimizes the error of the *total* directly. With the
trace-standardized blocks concatenated,
$Z=[\tilde X_0,\dots,\tilde X_{12}]\in\mathbb{R}^{n_S\times 25600}$:

$$
\min_{C}\;\Big\|\,Z_S\,C - T_S\,\Big\|_F^2 + \beta\,\|C\|_F^2
\;\;\Longleftrightarrow\;\;
\min_{\{C_b\}}\;\Big\|\sum_b X_{b,S}C_b - T_S\Big\|_F^2 + \beta\sum_b s_b\,\|C_b\|_F^2 .
$$

This is the same per-block penalty as §3 but with a single shared target. Its
kernel is $K_Z=\sum_b K_b/s_b$, a sum of equally weighted normalized block
kernels.

The hypothesis class *contains* `global_ridge`: set $C_b=0$ for $b<12$. So the
joint fit can match `global_ridge`, and it can only beat it if the intermediate
blocks carry information $f_B$ lacks.

**Hypothesis.**
- **H4a:** separate fitting is the remaining cause, so the joint fit closes most
  of the gap.
- **H4b:** $B$'s intermediate blocks carry correction signal beyond $f_B$, so the
  joint fit *beats* `global_ridge`.

**Experiment.** `scripts/block_ridge_experiments/joint_ridge.py`.
- The joint ridge uses production `_ridge` on $Z$, with each $s_b$ computed on
  the support only and applied unchanged to $U$ and test.
- **Fairness:** `global_ridge` gets the identical treatment. $f_B$ is
  standardized, the same $\beta$ grid $\{10,1,\dots,10^{-6}\}$ is searched, and
  $\beta$ is chosen on $U$.
- As a sanity check, the grid's $\lambda=1$ on $f_B$ corresponds to
  $\beta\approx1/6$. The tuned `global_ridge` at $\beta=1$ and $\beta=0.1$
  brackets the grid's recorded accuracy.

**Expected outcomes.**
- joint ≈ `global_ridge` $\gg$ per-block: H4a holds, H4b does not.
- joint $>$ `global_ridge`: H4b holds, the per-block design has value, and it was
  only badly fit.
- joint ≈ per-block: the joint target does not help, so the fault lies elsewhere.

**Outcome (fs200).**

| task | as run | per-block, trace | **joint** | global, tuned | global, grid $\lambda=1$ |
|---|---|---|---|---|---|
| mnli | 0.559 | 0.579 | **0.648** | 0.651 | 0.655 |
| sick | 0.747 | 0.771 | **0.794** | 0.800 | 0.798 |
| rte | 0.577 | 0.576 | **0.598** | 0.625 | 0.616 |
| scitail | 0.780 | 0.810 | **0.814** | 0.826 | 0.826 |
| qnli | 0.712 | 0.742 | **0.754** | 0.757 | 0.761 |
| snli | 0.656 | 0.680 | **0.727** | 0.735 | 0.734 |

- **H4a holds.** The joint fit gains 0.4–7 points over per-block trace, most on
  mnli and snli, the tasks with the largest gaps.
- **H4b does not.** The joint fit trails tuned `global_ridge` by 0.3–1.2 points on
  five tasks and by 2.7 on rte. Extra blocks the model is free to ignore end up
  adding variance, not signal.
- **The selection is stable.** Validation picked $\beta=0.1$ in 15 of 18 runs,
  well inside the grid.

## 5. Does more support change the verdict?

**Idea.** With $n_S$ as small as 400–600 rows against a 25,600-dimensional
concatenation, the extra blocks might simply lack the data to help.

**Hypothesis.** If H4b is false only at small $n_S$, the joint-minus-global gap
should shrink, or turn positive, at fs300 and fs500.

**Experiment.** The grid groups `ntk-br-fs300` and `ntk-br-fs500` were rerun with
artifacts kept, and §3 and §4 repeated on them. Each refit job was chained to its
rerun with `--dependency=afterok`. That made 108 jobs, all completed.

**Expected outcomes.**
- A gap that trends upward with $n_S$ means the blocks help given enough data.
- A flat gap means they don't, at least up to 500 examples per class.

**Outcome.** Test accuracy at fs 200 / 300 / 500:

| task | as run | per-block, trace | joint | global, tuned | joint − global (pts) |
|---|---|---|---|---|---|
| mnli | .559 / .560 / .571 | .579 / .606 / .638 | .648 / .658 / .685 | .651 / .661 / .690 | −0.3 / −0.3 / −0.5 |
| sick | .747 / .752 / .763 | .771 / .788 / .802 | .794 / .804 / .813 | .800 / .809 / .815 | −0.6 / −0.5 / −0.2 |
| rte | .577 / .560 / .580 | .576 / .573 / .582 | .598 / .621 / .636 | .625 / .610 / .639 | −2.7 / +1.1 / −0.3 |
| scitail | .780 / .773 / .759 | .810 / .823 / .841 | .814 / .836 / .849 | .826 / .846 / .855 | −1.2 / −1.0 / −0.6 |
| qnli | .712 / .731 / .719 | .742 / .758 / .768 | .754 / .761 / .772 | .757 / .771 / .782 | −0.3 / −1.0 / −1.0 |
| snli | .656 / .655 / .659 | .680 / .700 / .728 | .727 / .740 / .751 | .735 / .747 / .760 | −0.8 / −0.7 / −0.9 |

- **`block_ridge` as run does not improve with support,** and on scitail it gets
  worse. $n_S\le1500<2048$ keeps every intermediate block in the interpolating
  regime of §2.
- **The per-block fit gains the most from support.** Its gap to the joint fit
  shrinks (mnli 6.9 → 4.7 points, snli 4.7 → 2.3), because more rows make the
  separate fits less noisy.
- **The joint fit stays below tuned `global_ridge` in 17 of 18 (task, fs) pairs.**
  The one exception is rte fs300 (+1.1), and rte is the noisiest task: its test
  $R^2$ is near or below zero for every method. The gap does not trend upward.

**Flat gap.** Up to 500 examples per class, $B$'s intermediate blocks add
nothing that $f_B$ lacks.

## 6. Aside: is the linear regime the right one?

`block_ridge` requires `feature_regime="linear"`, so we checked that this matches
the checkpoint.

- All 285 `block_ridge` configs (grid runs plus reruns) use source `t5-base` with
  the task checkpoint from `t5base_6text_noreg_ntk/`.
- Those checkpoints were trained on the tangent model: per
  [`t5-encoder-steer-grid.md`](t5-encoder-steer-grid.md), the training args
  differ from `nonlinear` only by `tangent: true`. The mnli checkpoint scores
  0.383 through an ordinary forward pass and 0.830 through the linearized one.
  The Stage-1 oracle's 0.815–0.817 is consistent with the latter.
- The *pretrained* weights $\theta_0$ are the ordinary `t5-base`; only the
  fine-tuning is linearized.
- $B$ is not linearized. Its per-block activations, the regression inputs, come
  from a plain forward pass.

The regime therefore matches the checkpoint, and it is not a cause. All three
causes above sit on $B$'s side.

## 7. The correction scale $\alpha$

**Idea.** Every number so far uses $\alpha=1$. A noisy correction might do better
when damped.

There is also a free equivalence. For every predictor here the target is linear
in $L$, and so is every ridge prediction at a fixed $\beta$, since ridge is
linear in its targets. And $L\propto 1/(1+\lambda_1)$. So

$$
\alpha\;\text{at}\;\lambda_1 \;\equiv\; \frac{\alpha}{1+\lambda_1}\;\text{at}\;\lambda_1=0 .
$$

The $\alpha$ sweep therefore also covers `stage1_lambda`, and the grid's
$\lambda_1=1,\ \alpha=1$ is an unregularized map at $\alpha=\tfrac12$.

**Hypothesis.** Accuracy has an interior optimum in $\alpha$, and a noisier
predictor wants a smaller one.

**Experiment.** `scripts/block_ridge_experiments/alpha_sweep.py`.
- It uses cached-feature space, i.e. `stage2_test_acc` with $\alpha\ne1$.
- It covers all six predictors: oracle, as run, per-block trace, joint, and
  `global_ridge` tuned and at the grid's $\lambda$. Tuned predictors reuse their
  validation-picked $\beta$.
- It asserts that $\alpha=0$ reproduces stage0 exactly and that $\alpha=1$ of the
  saved fit reproduces the recorded `stage2_test_acc`.
- The first grid was $\alpha\in\{0,0.2,\dots,1\}$ over all 54 runs.

**Expected outcomes.**
- A peak strictly between 0 and 1 means damping helps.
- If the peak differs between predictors, $\alpha$ could change the ranking.

**Outcome (coarse grid).** fs200, mean over tasks × seeds:

| method | $\alpha=0$ | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|
| oracle | 0.657 | 0.829 | 0.826 | 0.827 | 0.826 | 0.826 |
| block_ridge as run | 0.657 | 0.677 | 0.674 | 0.673 | 0.672 | 0.672 |
| per-block trace | 0.657 | 0.697 | 0.694 | 0.693 | 0.693 | 0.693 |
| joint | 0.657 | 0.724 | 0.722 | 0.722 | 0.722 | 0.722 |
| global, tuned | 0.657 | 0.733 | 0.733 | 0.732 | 0.732 | 0.732 |
| global, grid | 0.657 | 0.734 | 0.732 | 0.732 | 0.732 | 0.732 |

The curves are flat from 0.2 to 1 for every predictor and every support size,
including the oracle.

**Explanation: the argmax geometry.** Let $s=W_Bf_B+b_B$ and $g=W_B\,c$. The
corrected prediction moves from $B$'s choice $k_0$ to class $k$ when

$$
\alpha\,(g_k - g_{k_0}) \;>\; s_{k_0}-s_k \;\ge\; 0 .
$$

As $\alpha\to\infty$ the prediction becomes $\arg\max_k g_k$, independent of
$B$. The switch happens around $\alpha^\star\approx |m|/\|g\|$. $B$'s
nearest-mean head has tiny margins, while the oracle's logit corrections are
large:

| task | $B$'s margin range | oracle $\|L\delta_A\|$ range | lowest-quartile boundary |
|---|---|---|---|
| mnli | [−0.097, 0.113] | [0.79, 31.5] | 1.76 |
| qnli | [−0.066, 0.091] | [0.65, 5.1] | 1.29 |
| rte | [−0.031, 0.059] | [0.89, 3.6] | 1.25 |
| scitail | [−0.078, 0.098] | [0.82, 13.7] | 1.89 |
| sick | [−0.226, 0.325] | [1.10, 26.0] | 3.16 |
| snli | [−0.064, 0.067] | [1.07, 14.2] | 2.24 |

This gives $\alpha^\star\sim0.1/2=0.05$, so by $\alpha=0.2$ the correction
decides almost every prediction. A useful sweep has to look below 0.2.

**Outcome (fine grid $\alpha\in\{0,0.005,\dots,0.2\}$ plus 1).** Mean over tasks × seeds:

| method | fs | $\alpha=0$ | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.075 | 0.1 | 0.15 | 0.2 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 200 | 0.657 | 0.730 | 0.764 | 0.799 | 0.813 | 0.822 | 0.826 | 0.827 | 0.829 | 0.829 | 0.826 |
| as run | 200 | 0.657 | 0.693 | **0.703** | **0.703** | 0.700 | 0.694 | 0.689 | 0.685 | 0.679 | 0.677 | 0.672 |
| per-block trace | 200 | 0.657 | 0.688 | 0.701 | **0.710** | 0.709 | 0.706 | 0.703 | 0.701 | 0.698 | 0.697 | 0.693 |
| joint | 200 | 0.657 | 0.695 | 0.711 | 0.723 | **0.727** | **0.727** | 0.726 | 0.726 | 0.725 | 0.724 | 0.722 |
| global, tuned | 200 | 0.657 | 0.693 | 0.709 | 0.723 | 0.729 | 0.734 | **0.735** | 0.734 | 0.733 | 0.733 | 0.732 |
| global, grid | 200 | 0.657 | 0.693 | 0.711 | 0.723 | 0.729 | 0.734 | **0.737** | 0.736 | 0.735 | 0.734 | 0.732 |
| as run | 300 | 0.657 | 0.698 | **0.706** | 0.703 | 0.697 | 0.690 | 0.686 | 0.683 | 0.679 | 0.678 | 0.672 |
| per-block trace | 300 | 0.657 | 0.693 | 0.708 | 0.717 | **0.720** | **0.720** | 0.717 | 0.716 | 0.714 | 0.712 | 0.708 |
| joint | 300 | 0.657 | 0.696 | 0.713 | 0.729 | 0.735 | 0.738 | **0.740** | **0.740** | 0.739 | 0.739 | 0.737 |
| global, grid | 300 | 0.657 | 0.697 | 0.717 | 0.731 | 0.739 | 0.744 | 0.744 | 0.747 | **0.748** | **0.748** | 0.746 |
| as run | 500 | 0.657 | 0.701 | **0.710** | 0.709 | 0.703 | 0.695 | 0.690 | 0.687 | 0.683 | 0.681 | 0.675 |
| per-block trace | 500 | 0.657 | 0.696 | 0.715 | 0.729 | 0.733 | **0.736** | 0.735 | 0.734 | 0.731 | 0.730 | 0.727 |
| joint | 500 | 0.657 | 0.702 | 0.722 | 0.739 | 0.747 | 0.751 | 0.753 | **0.754** | 0.753 | 0.753 | 0.751 |
| global, grid | 500 | 0.657 | 0.700 | 0.722 | 0.742 | 0.750 | 0.757 | 0.759 | **0.760** | **0.760** | 0.760 | 0.758 |

- **The oracle rises monotonically and saturates.** It reaches 96% of its gain by
  $\alpha=0.05$, as the margin argument predicts, and is flat after that.
- **Damping helps the noisier predictors, and the noisier the predictor the
  smaller its best $\alpha$.** `block_ridge` as run peaks at
  $\alpha\approx0.01$–$0.02$ (0.703 vs 0.672 at $\alpha=1$, fs200). Per-block
  trace peaks at 0.02–0.05, joint at 0.03–0.1, and `global_ridge` at 0.075–0.2.

  A plausible reading, not separately tested: at small $\alpha$ only examples
  whose predicted logit gap exceeds $B$'s margin flip. The correction then acts
  as a magnitude-gated override, and it keeps $B$'s decision wherever the
  predicted correction is small, which is where §1 showed block_ridge's
  predictions are least reliable (logit $R^2=-2.5$ in the smallest-correction
  quartile).
- **The ranking is unchanged wherever the corrections matter.** At each
  predictor's best $\alpha$, `global_ridge` ≥ joint > per-block trace > as run,
  at every fs. At $\alpha\le0.005$ all predictors sit within about 0.01 of each
  other, because barely any predictions change.
- **The best-$\alpha$ numbers are optimistic.** $\alpha$ was read off test here,
  with no validation selection. The conclusions rest on the ranking, which holds
  across the whole curve, not on the peak values.

---

## Conclusions

1. `block_ridge` as run is an interpolator. With $n_S<2048$ and $\lambda=1$
   against Gram eigenvalues of $10^2$–$10^{12}$, each intermediate block fits its
   support exactly, and its test $R^2$ is negative (§1–2).
2. Trace-scaled $\lambda_b=\beta\,\operatorname{tr}(K_b)/n_S$ fixes the scale
   problem. The best $\beta$ is $10^{-4}$–$10^{-2}$. It narrows the gap to
   `global_ridge` but does not close it (§3).
3. Fitting one joint ridge to the total correction closes most of the remainder
   (§4). Even so it only ties `global_ridge` at every support size up to fs500
   (§5). For this t5-base → t5-large pair, $B$'s intermediate blocks add no
   usable signal beyond $f_B$.
4. $\alpha$ matters only below about 0.1. There it damps noisy predictors
   usefully, but it never changes the ranking (§7). The $\alpha$ sweep also
   covers `stage1_lambda` through $\alpha/(1+\lambda_1)$.
5. The regime choice is correct and not a cause (§6).

**Caveats.**
- All results are in cached-feature space. The live "rebased" evaluation path
  was not rerun.
- There is one model pair.
- rte is too noisy to read small differences from.
- The $\alpha$ peaks were read off test.
- The unselected train rows used for choosing $\beta$ are more labelled data than
  a strict few-shot budget would allow. This is appropriate for a diagnosis, not
  for a method comparison.

---

## Reproducing

```bash
source .venv/bin/activate

# Artifact-saving reruns into t5enc_block_ridge_diag/ (never the grid's own root)
python scripts/slurm/submit_block_ridge_diag.py --group ntk-br-fs200   # also ntk-br-fs300, ntk-br-fs500

# §1: diagnostics + validation gate, one run
python scripts/diagnose_block_ridge.py \
  --exp-dir /work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag/mnli_linear_noreg-ntk_block_ridge_fs200_seed33 \
  --output  /work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag/_reports/mnli_linear_noreg-ntk_block_ridge_fs200_seed33.json

# §3, §4, §7: one Slurm job per run
scripts/block_ridge_experiments/submit.sh trace_lambda.py
scripts/block_ridge_experiments/submit.sh joint_ridge.py
scripts/block_ridge_experiments/submit.sh alpha_sweep.py --alphas 0,0.2,0.4,0.6,0.8,1
scripts/block_ridge_experiments/submit.sh alpha_sweep.py --alphas 0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,1 --out-subdir alpha_sweep_fine

# every table in this document
python scripts/block_ridge_experiments/summarize.py all
```

**Notes on running these:**
- `alpha_sweep.py` reads the $\beta$ picks from the `trace_lambda/` and
  `joint_ridge/` reports, so run it after those two.
- `submit_block_ridge_diag.py`'s `--dry-run` still creates the experiment
  directory. `submit_text_rebase.sh` resolves the config before it checks
  `DRY_RUN`, so point `--results-root` at scratch for a dry run.
- The refit scripts are CPU-bound. The GPU is requested only because the
  partition expects one.

To run without a Slurm scheduler, see [Appendix A](#appendix-a-running-the-experiments-with-plain-python).

**Where things are:**
- Per-run JSON reports are under `t5enc_block_ridge_diag/_reports/`, in
  `trace_lambda/`, `joint_ridge/`, `alpha_sweep/` and `alpha_sweep_fine/`.
- The per-run diagnostic reports are the `*.json` files at the top level of that
  directory.
- The 54 reruns' artifacts take about 12 GB.

---

## Appendix A: running the experiments with plain Python

Everything above can be run without Slurm. Only step A.2 needs the models,
and a GPU in practice. Everything after it reads saved tensors and runs on CPU.
Run all commands from the repository root with the project's environment
active (`source .venv/bin/activate`).

### A.1 What you need, and where it is looked up

Each experiment works on **run directories**, one per (task, few_shot, seed),
each holding:

| file | written by | used for |
|---|---|---|
| `config.json` | A.2 | task, seed, `few_shot`, `ridge_lambda`, feature-cache location |
| `summary.json` | A.2 | the recorded `steer_diagnostics` that every script checks itself against |
| `<task>_steer_artifacts.pt` | A.2 | Stage-1 map $L$, $P_B$, $W_B$, $b_B$, support indices, saved block_ridge coefficients |

The scripts also need the **feature cache** that the run read:

```
<feature_cache_dir>/google-t5__t5-base__encoder_classification_to_google-t5__t5-large__encoder_classification/<task>/linear/{train,test}/
    features_A.pt  delta_A.pt  delta_A_blocks.pt  features_B.pt  features_B_blocks.pt  y_A.pt
```

Two environment variables relocate everything without editing any file:

| variable | default | meaning |
|---|---|---|
| `DIAG_ROOT` | `/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag` | where results are written (`$DIAG_ROOT/_reports/...`), and where `alpha_sweep.py` and `summarize.py` read them |
| `FEATURE_CACHE_DIR` | each run's `method_params.feature_cache_dir` | the cache root. Only the root changes; the layout under it stays as above |

Run directories can live anywhere, because every script takes the run
directory as an argument.

Resources:
- Run **one run per process**. A single process looping over many runs keeps
  every run's tensors alive and was killed on a login node.
- Peak memory is a few GB for `diagnose_block_ridge.py` and `trace_lambda.py`.
  It goes up to about 20 GB for `joint_ridge.py` and `alpha_sweep.py` at fs500,
  because the 25,600-dimensional concatenated features are materialized for the
  unselected train rows.
- Each run takes about a minute.

### A.2 Producing a run directory (the only step that needs the models)

A grid run's own `config.json` is already the right configuration. Only
`save_steer_artifacts_dir` must point at the new directory, so that the fitted
transforms are kept:

```bash
GRID=/work/intesasanpaolo_phd/merge-and-rebase/t5enc_steer_text
NAME=mnli_linear_noreg-ntk_block_ridge_fs200_seed33
RUN=$DIAG_ROOT/$NAME
mkdir -p "$RUN"

python - "$GRID/$NAME/config.json" "$RUN" <<'PY'
import json, sys
src, run = sys.argv[1], sys.argv[2]
cfg = json.load(open(src))
cfg["save_steer_artifacts_dir"] = run
json.dump(cfg, open(f"{run}/config.json", "w"), indent=2)
PY

python -m merge_and_rebase.eval.text_rebase \
  --config "$RUN/config.json" --local-log-dir "$RUN" --run-name summary
```

This is exactly what the Slurm job runs (`scripts/slurm/text_rebase_job.sbatch`).

- With a warm feature cache it loads the three models, refits Stage 1 and 2,
  and evaluates, which takes minutes.
- With a cold cache it first computes features, running one forward-mode `jvp`
  per source block per batch. That is much slower, and should be done once per
  task before running several seeds.
- `config.json` says `"device": "cuda"`. `--device cpu` overrides it, but that
  path has not been timed or verified for this model pair.

For a run that is not in the grid, start from `configs/text_rebase_t5enc_steer.json`
and set the per-cell fields that `grid.py`'s `overrides_for` sets:

- `tasks`
- `seed`
- `tuned_ckpts.<task>`
- `target_task_heads`
- `method_params.{feature_cache_dir, feature_regime, stage_2_strategy, few_shot}`

This prints them for any cell:

```bash
python -c "import sys; sys.path.insert(0, 'scripts/slurm'); import grid; \
c = grid.Cell('ntk-br-fs200', grid.NTK, 'mnli', 'block_ridge', 200, 33); \
print('\n'.join(grid.overrides_for(c, save_artifacts=True)))"
```

The output is `dotted.key=value` lines, such as `tasks=mnli` and
`method_params.few_shot=200`. Apply them to the base config's JSON as in the
snippet above.

### A.3 Regression diagnostics (§1)

```bash
python scripts/diagnose_block_ridge.py \
  --exp-dir "$RUN" \
  --output  "$DIAG_ROOT/_reports/$NAME.json"
```

The script prints a summary table and writes the JSON report. It exits non-zero
without writing anything unless the recomputed stage0, stage1 and stage2
accuracies match the run's own `summary.json`. It also compares against the
original grid cell and its `global_ridge` twin under `--grid-root` (default
`/work/.../t5enc_steer_text`), but only when those exist. On a machine without
them, only the self-check runs.

### A.4 Trace-scaled $\lambda$ (§3), joint ridge (§4), $\alpha$ sweeps (§7)

`alpha_sweep.py` reuses the $\beta$ that the first two scripts chose on
validation. Run those two first, for every run you want to sweep:

```bash
for RUN in "$DIAG_ROOT"/*_seed*; do
  python scripts/block_ridge_experiments/trace_lambda.py "$RUN"
  python scripts/block_ridge_experiments/joint_ridge.py  "$RUN"
done

for RUN in "$DIAG_ROOT"/*_seed*; do
  python scripts/block_ridge_experiments/alpha_sweep.py "$RUN" --alphas 0,0.2,0.4,0.6,0.8,1
  python scripts/block_ridge_experiments/alpha_sweep.py "$RUN" \
    --alphas 0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,1 --out-subdir alpha_sweep_fine
done
```

Each invocation handles one run and writes one JSON file under
`$DIAG_ROOT/_reports/{trace_lambda,joint_ridge,alpha_sweep,alpha_sweep_fine}/<task>_fs<k>_seed<s>.json`.
The loops start a fresh process per run, as A.1 recommends.
`trace_lambda.py` accepts several run directories in one call, but that shares
one process and its memory.

Built-in checks. Each script aborts rather than reporting if a check fails:

- `trace_lambda.py`: the unscaled refit reproduces the saved coefficients
  (relative error $<10^{-6}$) and the recorded `stage2_test_acc`.
- `alpha_sweep.py`: $\alpha=0$ reproduces stage0 exactly for every predictor.
  $\alpha=1$ of the saved fit reproduces the recorded `stage2_test_acc`
  whenever $1$ is in `--alphas`.
- `joint_ridge.py` has no recorded value to compare against. Its `global`
  entries at $\beta\approx1/6$ should bracket the grid's `global_ridge`
  accuracy (§4).

### A.5 Tables

```bash
python scripts/block_ridge_experiments/summarize.py all     # or: trace | joint | alpha | alpha_fine | margins
```

It reads `$DIAG_ROOT/_reports`, so set `DIAG_ROOT` if the results are somewhere
else, for instance after unpacking the results tarball. The `margins` table
reads the §1 reports (`$DIAG_ROOT/_reports/*fs200_seed33.json`).

### A.6 Expected reproducibility

Rerunning A.4 on a different machine reproduces the stored results up to
floating-point summation order. Checked on mnli, fs200, seed 33:

- `alpha_sweep` came out identical.
- `trace_lambda` and `joint_ridge` differed by at most $2\times10^{-12}$ in any
  reported accuracy or $R^2$.

Support selection is deterministic in (labels, `few_shot`, seed), and every fit
is a closed-form solve, so there is no other source of variation.
