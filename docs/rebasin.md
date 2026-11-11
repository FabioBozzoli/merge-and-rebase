# Rebasin

Rebasin transports a task vector from a source base model to a target base model. The current evaluation entrypoint operates on one task vector at a time; merging and transport can be chained through checkpoint artifacts, but are not yet a single configurable multi-task pipeline.

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision8_gradfix_hf.json
```

Useful starter configurations include:

- [GradFix over Vision8](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_gradfix.json)
- [Theseus over Vision8](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_theseus_all.json)
- [TransFusion transport](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_transfusion_rebase.json)
- [BiCo alpha sweep](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_bico_all_alpha_sweep.json)

## Core Fields

| Field | Meaning |
|---|---|
| `source_clip_model`, `source_clip_pretrained` | Base model defining the source task vector. |
| `target_clip_model`, `target_clip_pretrained` | Base model receiving the transported update. |
| `tuned_ckpts` | Per-task fine-tuned checkpoint references. |
| `tasks` | `"all"` or a comma-separated task subset. |
| `alpha` / `alpha_search` | Fixed task-vector scale or an alpha sweep. |
| `transport_method` | Registered transport method. |

`identity` and `orthogonal_shift` are data-free. `gradfix`, `theseus`, `transfusion`, and `bico` use method-specific model or data inputs during preparation. See [methods.md](methods.md) for behavior and parameters.

## Linear probing after transport

Set `"linear_probe_head": true` to refit the target's classification head once the
chosen method has produced the target backbone. The backbone is frozen; only the
zero-shot head (`_zs_text_features`, the `[C, D]` matrix `OpenClipClassifier.forward`
multiplies image features by) is trained, by Adam on a class-balanced few-shot support
set drawn from the task's train split.

Two properties are deliberate:

- **The probe starts from the zero-shot head, never from a random draw.** Its epoch-0
  log line is therefore the plain zero-shot accuracy of whatever backbone it is sitting
  on, and everything above that line is what probing added. For `steer` this is also
  load-bearing: Stage 1 builds its correction through `pinv(w_b)` of that exact head, so
  a fresh draw would discard the readout the correction only works through.
- **The probed head replaces the zero-shot head for both the baseline and the rebased
  column.** The classifier is identical on both sides, so the columns keep isolating what
  the transport did rather than mixing in a change of classifier.

Under `steer` the probe trains inside `steer_correction_context`, i.e. on the corrected
features — every other method has already baked its transport into the weights.

| Field | Meaning |
|---|---|
| `linear_probe_head` | Enable probing after transport. |
| `linear_probe_shots_per_class` | Support-set size per class. Defaults to `method_params.shots_per_class` / `few_shot` when the method defines one, so probe and transport see the same budget; required otherwise (e.g. `identity`). |
| `linear_probe_epochs` | Passes over the support set (default `50`). |
| `linear_probe_lr` | Adam LR (default `1e-3`). Keep it small: head rows are L2-normalized (entries ~`1/sqrt(D)`) and Adam's step is ~`lr` per coordinate regardless of gradient magnitude, so a large LR erases the zero-shot head in a couple of steps. |
| `linear_probe_log_every` | Epoch interval for the loss + support/val/test log lines (default: ~10 lines per run). |

Prefer `alpha_search: false`: the head is probed once, on the backbone at the configured
`alpha`, so a sweep would score every other alpha with a head fit for a different
backbone. The run warns when both are enabled.

Starter configs: [steer](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_steer_linear_probe.json),
[theseus](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_theseus_linear_probe.json) and
[bico](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_bico_linear_probe.json).

### Zero-shot control (no rebasin)

The baseline those runs are measured against is a separate entrypoint, not a degenerate
transport:

```bash
python -m merge_and_rebase.eval.vision_linear_probe \
  --config configs/vision8_zeroshot_linear_probe.json
```

`eval/vision_linear_probe.py` builds exactly one model — the `clip_model` /
`clip_pretrained` pair from the config — and uses only that pretrain's own backbone and
its own zero-shot heads. No source model, no fine-tuned checkpoint, no task vector is
constructed anywhere in it. It reports zero-shot and probed accuracy per task, so the
gap between the two columns is what probing alone buys. Point `clip_model` /
`clip_pretrained` at the *target* of a rebasin run and keep the probe budget identical
to compare them fairly.

## Linear probe distillation (`stage1_mode="logit_distill"`): opt-in baseline, not steer

Disabled by default (`stage1_mode="delta_projection"`, the original two-stage `steer`). It is a
baseline/ablation; **do not use it in place of `steer`** (Stage 1 + Stage 2, typically
`block_ridge` with `block_ridge_mode="smoothed_residual"`, `rho=0.9`) unless that is the point.

From the same few-shot support set (K per class) and the same feature cache, it skips Stage 1
and fits one linear map on B's final (L2-normalized) feature:

```
min_L || H_A^ft T_A - (H_B T_B + H_B L) ||_F^2 + lambda_eff ||L||_F^2
```

with `H_A^ft = f_A + delta_A` (A fine-tuned; linearized under `feature_regime="linear"`), `T_A`, `T_B`
the cached heads and `H_B` B's pretrained feature. At inference only B and `L` are used.

- `H_B T_B + H_B L = H_B (T_B + L)`: this is a new head shrunk towards B's zero-shot head. With
  `ridge_lambda=0` (allowed here: minimum-norm least squares) and `n > d_B`, `T_B` drops out
  (plain probe distillation); in the few-shot regime (`n <= d_B`) the solution interpolates the
  support residual with minimum `||L||`.
- `correction_space="logit"` adds `H_B L` to the logits; `"feature"` regresses `E pinv(W_B)^T` and
  adds it to the feature. Since `W_B pinv(W_B) = I_C`, both give the same logit offset (the feature
  path renormalizes, a positive per-row scale: identical predictions).
- `global_ridge_lambda_scaling="trace"`: `lambda_eff = ridge_lambda * tr(H_B H_B^T)/n`. For L2-normalized
  CLIP features this equals `ridge_lambda` (no-op); it matters for text.
- Input of the ridge: `stage_2_strategy="global_ridge"` regresses E on B's final feature (default); `stage_2_strategy="concat_ridge"`
  regresses the same E on the **concatenation of all of B's block activations** (each block scaled to unit mean squared norm, as in steer's
  `concat_ridge`; `ridge_lambda` must be > 0). Same target, Stage 1 still skipped: only the input of the linear map changes. Other strategies are rejected.
- `stage1_lambda`, `stage1_energy_threshold`,
  `stage1_regularization_mode` are ignored. Diagnostics add `teacher_test_acc` (A's cached accuracy),
  `global_ridge_lambda_effective` (and `lpd_input`, `concat_input_dim`, `concat_effective_lambda` for `concat_ridge`).
- Relation to steer: with `stage1_regularization_mode="standard"`, `stage1_lambda=0`, logit space and
  a support set with `n <= d_A` and full-row-rank `delta_A`, Stage 1 reproduces the residual exactly and
  steer's `global_ridge` coincides with LPD; otherwise (e.g. `n > d_A`) they differ.

```json
"method": "steer",
"method_params": {"stage1_mode": "logit_distill", "stage_2_strategy": "global_ridge",
                  "correction_space": "logit", "ridge_lambda": 0.01, "few_shot": 5, "feature_regime": "linear", ...}
```

## Integrated per-block residuals (`feature_regime="integrated"`): nonlinear sources, opt-in

`block_ridge` needs A's fine-tuning residual split into per-block contributions. Under
`feature_regime="linear"` these are first-order (Taylor at `theta0`) JVPs, which is what an
NTK-linearized source actually computes. For a **nonlinearly fine-tuned** source the opt-in regime
`"integrated"` uses the path-integral decomposition along `theta(t) = theta0 + t * dtheta`
(`dtheta = theta1 - theta0`, `f` = A's L2-normalized final feature, `M_l` = mask of block `l`):

```
D(x)   = f(x; theta1) - f(x; theta0) = sum_l D_l(x)
D_l(x) = int_0^1 J_theta f(x; theta0 + t dtheta) (M_l dtheta) dt
       ~ (1/K) sum_{k=0}^{K-1} JVP(f, primals = theta0 + t_k dtheta, tangents = M_l dtheta),   t_k = (k + 1/2) / K
```

- Primals interpolate **all** parameters (the model at `t_k` is the same for every block); only the
  tangent is masked, and it is not multiplied by `t_k`. Masking adds no approximation (the JVP is linear
  in the direction): what remains is the quadrature error and floating point.
- `integration_points` = K (required, integer >= 1) is the number of midpoint nodes: a hyperparameter.
  Cost of the extraction: `K * num_blocks` JVPs per batch (ViT-B/16: `13 K`), paid on the train split
  only; the test split runs two plain forwards unless `block_ridge_diagnostics=true`.
- Stage 1 is fit on the **exact** residual (`delta_A`, two forwards, bit-identical to the `standard`
  regime); the integrated `D_l`, mapped through Stage 1 (`D_l R^T pinv(W_B)^T`), are the per-block
  Stage 2 targets. Everything downstream (`_fit_block_ridge`, pairing, eval) is unchanged.
- Cache: `<feature_cache_dir>/<pair>/<Task>/integrated_k<K>/<split>/` (one cache per K; same files as
  the linear regime plus `integration_rel_error.pt`). `prepared["feature_regime"]` reports
  `integrated_k<K>`.
- Diagnostics: `integration_points`, `integration_rel_error_{mean,max}` = per-example
  `||sum_l D_l - D|| / ||D||` on the train features. It measures the error of the *sum* only: errors of
  individual blocks can cancel, so check the stability of each `D_l` when K doubles before trusting a K.
- Assumptions: eval mode, buffers common to both endpoints (true for CLIP ViTs: no BatchNorm), every
  fine-tuned visual parameter assigned to a block (`_parameter_blocks_for_visual` raises otherwise). The
  split of the nonlinear interactions between blocks depends on the path (here the straight segment).
  The identity requires `f` to be smooth along the path: a near-singular point (e.g. a LayerNorm fed an
  all-zero token at `theta0`, as in the tiny test model) breaks the quadrature at any small K.
- Works with every Stage 2 strategy; only `block_ridge` uses the integrated blocks. Vision only
  (`steer`), not `steer_text`.

```json
"method": "steer",
"method_params": {"feature_regime": "integrated", "integration_points": 4, "stage_2_strategy": "block_ridge",
                  "block_ridge_mode": "smoothed_residual", "rho": 0.9, "block_ridge_lambda_scaling": "trace",
                  "ridge_lambda": 0.1, "few_shot": 5, "train_features_per_class": 50, ...}
```

Configs: `configs/vision8_steer_integrated/` (`make_configs.py --k 4 8 16`), launcher
`scripts/slurm/steer_integrated_vision.sh`.

For HuggingFace text models (T5, Qwen, Llama, ...) instead of CLIP, see [Text Rebasin](text-rebasin.md), which drives the same rebase methods through `eval/text_rebase.py`.
