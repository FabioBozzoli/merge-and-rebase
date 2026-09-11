# Text Rebasin

`eval/text_rebase.py` is the text counterpart of [Rebasin](rebasin.md): it transports a
task vector from a source base language model **A** to a target base language model **B**,
for any HuggingFace `transformers` model the repo already supports (`AutoModelForCausalLM`,
`AutoModelForSeq2SeqLM`, `AutoModelForSequenceClassification` — see
[`models/text_lm.py`](../src/merge_and_rebase/models/text_lm.py)), e.g. T5, Qwen, Llama.

It is a structural twin of `eval/vision_rebase.py`: same config surface, same alpha grid and
early stopping, same `untransported` / `target_zeroshot` baselines, same summary JSON. A text
run and a vision run of the same method are directly comparable and read by the same
`scripts/audit_rebase_summary.py`.

```bash
python -m merge_and_rebase.eval.text_rebase \
  --config configs/text_rebase_t5_theseus.json
```

Starter configs:

- [`configs/text_rebase_t5_gradfix.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_t5_gradfix.json) — same-shape pair, `untransported` baseline available.
- [`configs/text_rebase_t5_theseus.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_t5_theseus.json) — cross-width pair (t5-v1_1-small → t5-v1_1-base).
- [`configs/text_rebase_t5_bico.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_t5_bico.json)
- [`configs/text_rebase_t5_steer.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_t5_steer.json)
- [`configs/text_rebase_qwen_theseus.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_qwen_theseus.json) — decoder-only template (checkpoints not included).
- [`configs/text_rebase_t5base_t5large_theseus_nearestmean.json`](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/text_rebase_t5base_t5large_theseus_nearestmean.json) — cross-width pair (t5-v1_1-base → t5-v1_1-large) where B's head is *not* trained, but a nearest-class-mean head built by `scripts/build_nearest_mean_head.py` (see below).

## Which methods it drives

Nothing under `rebase/methods/` was changed. Each existing method reaches text models
through an adapter in `rebase/text/`:

| `method` | Status | How |
|---|---|---|
| `gradfix` | Works unmodified | Model-agnostic; driven by `causal_lm_recipe` / `seq_classification_recipe` from `models/grad_recipes.py`. |
| `theseus` | Works via a shim | `TextEncoderShim` wraps the HF model so `theseus`'s hooks and `.encode_image`-style call see a plain forward with a correct `attention_mask`. |
| `bico` | Works via the same shim | Its forward already goes through an injected recipe; the shim only satisfies a batch-size check. |
| `steer_text` | Text port, separate file | `steer` itself is CLIP-only (zero-shot text heads, ViT block tables). `rebase/text/steer_text.py` rebuilds the scaffolding for a linear classification head and imports steer's Stage 1/2 math unchanged. Use `"method": "steer_text"`, **not** `"steer"`. |
| `identity`, `orthogonal_shift` | Work unmodified | Already architecture-agnostic. |
| `transfusion` | **Not supported** | Its permutation spec (`rebase/permutations/spec.py`) only knows OpenCLIP ViT key names. Raises a clear error. |
| `bico_gradin` | **Not supported** | Marks the model input as requiring grad, which integer `input_ids` reject. Use `bico` instead. |

## Core config fields

| Field | Meaning |
|---|---|
| `source_model_name_or_path`, `source_model_arch` | Base model A (HF hub id or local path). |
| `target_model_name_or_path`, `target_model_arch` | Base model B receiving the transported update. |
| `model_kind` | `"sequence_classification"` or `"causal_lm"`. |
| `eval_mode` | `"auto"` \| `"head_logits"` \| `"prompt"` — same semantics as `eval/llm_merge.py`. |
| `source_task_heads`, `target_task_heads` | Per-task classification heads (`{task: {param_name: tensor}}`, produced by `finetune/train_text.py`), required for `eval_mode="head_logits"`. |
| `tuned_ckpts` | Per-task fine-tuned checkpoints for the **source** model A. |
| `tasks` | `"all"` or a comma-separated subset of the `nli6` suite (`snli, mnli, sick, qnli, rte, scitail`). |
| `method` / `method_params` | Registered rebase method and its kwargs. |
| `feature_cache_dir` | Top-level shortcut for `steer_text`'s `method_params.feature_cache_dir` (its on-disk pooled-feature cache). Set here, not only inside `method_params`, so it survives a `--method-params` override, which replaces that whole dict. |
| `alpha` / `alpha_search` / `alpha_selection` | Fixed scale, alpha sweep, and `"shared"` vs `"per_task"` selection — identical to vision. |
| `val_fraction` | Vision carves val out of test with a fixed seed so val/test never leak into each other; text does the same here (see below), since some NLI tasks' HF `test` split is literally their `validation` split. |

Full field list, in `eval/text_rebase.py`'s `main()`, mirrors `vision_rebase.py`'s CLI/config
merge pattern: every value can come from `--config`, overridden by the matching CLI flag.

## `eval_mode`: `head_logits` vs `prompt`

- **`head_logits`** (recommended, matches the example configs): the model is
  `AutoModelForSequenceClassification`, a per-task linear head is injected from
  `target_task_heads` before each evaluation, and accuracy comes from the head's logits.
  This is what `steer_text` requires — its Stage 1 fits a correction in the pooled-feature
  space in front of that head.
- **`prompt`**: the model is a causal/seq2seq LM, evaluated by comparing the log-probability
  of each label string after a prompt (`nli_accuracy`). No head is needed, but there is no
  `steer_text` support in this mode.

## Baselines

Same three baselines as vision, with the same fallback rule:

- **`untransported`** — `target_base + alpha * (source delta, unmodified)`. Only available
  when source and target share every delta key's shape (e.g. two checkpoints of the same
  architecture). `text_rebase.py` prints, per task, how many delta keys actually match the
  target base by name and shape.
- **`target_zeroshot`** — plain B, alpha-independent. Used automatically whenever shapes
  differ (e.g. t5-v1_1-small → t5-v1_1-base) or for `steer_text`, which never writes weights.
- **`mixed_baseline`** — per-task choice between the two.

## Val/test split

HF's NLI loaders (`data/text_loaders.py`) hand back real dataset splits, and for `qnli`,
`rte`, and `mnli` the `test` split **is** `validation`. Searching alpha on `val` and reporting
on `test` would then read the same rows. `text_rebase.py` instead carves a `val_fraction`
slice out of the evaluation split with a fixed seed (independent of `cfg["seed"]`), the same
way `data/vision_loaders.py` carves val out of test — val and test are always disjoint.

## Cross-architecture pairs

There is no text equivalent of the ViT-only block-extension preprocess, and what happens on a
depth mismatch (e.g. t5-v1_1-base, 12 layers → t5-v1_1-large, 24 layers) differs **per method**
— it is not one blanket rule. `text_rebase.py` prints both depths and a delta/target
key-coverage report at the start of each task so the actual overlap is visible either way.

- **`theseus` / `bico`** — width mismatches (different `d_model`) are exactly what these
  methods are built to handle: they compute per-layer alignment maps from activation
  statistics (procrustes/whitening) and transport across dimensions. Depth mismatches are a
  separate problem they do *not* solve: matching is by **exact block name**, so A's block `i`
  aligns only to B's block `i`. If B has more blocks than A, B's extra trailing blocks have no
  source block to align against and are transported **zero delta** — they keep their
  pretrained weights. This is a partial transport, not a failure: it runs, but only touches
  the first `len(A's blocks)` of B's blocks.
- **`gradfix`** — has no per-layer alignment at all. It masks A's raw delta against B's own
  gradient signs key-by-key, which requires the two tensors for the same parameter name to
  have the **same shape**. Any width mismatch (not just depth) raises immediately
  (`torch.where` on incompatible shapes). gradfix therefore only works for same-architecture
  pairs (e.g. `t5-v1_1-base` → `flan-t5-base`), never across model sizes.
- **`steer_text`** — never touches weights, so it is largely immune to this. `global_ridge` /
  `global_mlp` fit Stage 1/2 from A's feature dimension to B's, whatever those are. `block_ridge`
  explicitly handles a depth mismatch instead of dropping it: when B has more residual blocks
  than A, `block_group_strategy` (`concat` / `sum_avg`) groups B's blocks down to A's block
  count first, so every one of B's blocks participates in the fit — not just a name-matching
  prefix.

## Producing the inputs

- **Fine-tuned checkpoints and task heads** come from
  `python -m merge_and_rebase.finetune.train_text --text-config <config>` (see
  `finetune/configs/text.yaml` for a template). Set `output.extract_heads: true` to get a
  `heads.pt` usable as `target_task_heads` / `source_task_heads`.
- **`target_task_heads` must be heads trained *for B*.** The example configs use a
  linear-probe of B per task (`strategy.name: linear_probe`) — B's no-backbone-training head,
  the text analogue of CLIP's free zero-shot head.
- **`tuned_ckpts` can also name a full HF Hub model instead of a local checkpoint file.** A
  bare repo id with no local file and no weight-file extension (e.g.
  `"snli": "varun-v-rao/t5-base-snli"`) is loaded as a complete, already fine-tuned
  `AutoModelForSequenceClassification` in its own right — useful for testing the rebase
  methods against a checkpoint known to have actually learned its task, without waiting on a
  local fine-tuning run. `source_model_name_or_path` must then be the **exact pretrained base**
  that Hub checkpoint was fine-tuned from (matching tokenizer/vocab size and architecture), or
  the computed task delta mixes the real fine-tuning delta with a spurious cross-pretraining
  offset — this cannot be verified from the config alone, check the Hub model card.
- **`source_input_template` / `target_input_template`** cover a checkpoint fine-tuned on a
  single formatted string (e.g. `"premise: {premise} hypothesis: {hypothesis}"`) rather than
  this repo's default `tokenizer(premise, hypothesis)` two-segment pair encoding — the two
  produce different token sequences, and a model trained on one performs at chance on the
  other. `scripts/probe_nli_input_format.py` finds the right template empirically for a given
  checkpoint (scores a fixed sample under several candidates, including this repo's default);
  whichever clears chance by a wide margin is very likely the one it was trained on. Each
  field applies only to that model's own tokenization — a template for A does not affect B.

## Head-free baseline: nearest-class-mean (cosine) head

When B has no fine-tune to linear-probe from at all, `scripts/build_nearest_mean_head.py`
builds a `target_task_heads`-compatible `heads.pt` without training anything:

```bash
python -m scripts.build_nearest_mean_head \
  --model-name-or-path google/t5-v1_1-large --model-arch t5 \
  --task mnli --few-shot 8 --seed 33 --num-labels 3 \
  --output /path/to/t5-v1_1-large_mnli_nearest_mean.pt
```

Each class row is the L2-normalized centroid of B's own pooled features (the tensor B's real
head would consume — post-dense-tanh for T5, the last non-pad hidden state for a decoder-only
model) over a small few-shot support set, with the bias forced to zero:

```
mu_c = normalize( mean_{x in support(c)} normalize(pooled_feature(x)) )
```

No custom "cosine head" module is needed at eval time: for a fixed query `x`,
`cos_sim(x, mu_c) = (x · mu_c) / (|x| |mu_c|)`. Every `mu_c` is unit-norm, so this reduces to
`(x · mu_c) / |x|`, and `|x|` is the same positive scalar for every class — it cannot change
which class scores highest. So the raw dot product `feature @ weight.T` that `head_logits`
evaluation already computes (`TextLM.sequence_classification_accuracy`) is argmax-equivalent
to nearest-cosine classification once the weight rows are unit-norm and the bias is zero.

The pooled feature is extracted the same way `rebase/text/steer_text.py` already does it
(swap the head's final `nn.Linear` for `nn.Identity`, read what would have been its input),
and the script finishes by injecting the built head back into the live model and scoring it
on its own support set through `TextLM.sequence_classification_accuracy` — the exact function
`text_rebase.py` calls in `eval_mode="head_logits"` — as an end-to-end sanity check before the
file is written.

Point `target_task_heads` at the resulting file exactly as you would a trained one; nothing
else in the pipeline needs to know the head wasn't trained.

## Alternative: linear probing after rebasin

Set `"linear_probe_head": true` (instead of `target_task_heads`, they're mutually
exclusive) to skip the nearest-mean head entirely: B's classification head starts at
whatever fresh random init `AutoModelForSequenceClassification.from_pretrained` gives it,
and is trained from scratch **after** the rebase method's `prepare()`/`transport()` have
run, on the exact same few-shot support set (same count, same seed) the method itself
used — `method_params.shots_per_class` for `theseus`/`bico`, `method_params.few_shot` for
`steer_text`. Supported for `theseus`, `bico`, and `steer_text` only, and requires
`eval_mode: "head_logits"` set explicitly (`"auto"` resolves to `"prompt"` when no
`target_task_heads` path is given). `linear_probe_epochs` (default `200`, an epoch here is
one full pass over the tiny few-shot support set, since training is full-batch) and
`linear_probe_lr` (default `1e-2`) control the probe's Adam training loop.

Mechanically: for `theseus`/`bico`, the backbone is loaded as `target_base + alpha *
transported_delta` (the same state the eval loop would score); for `steer_text` (which
never writes weights) it's the plain target base under `steer_text_correction_context`.
Only the head's own parameters get gradients (`rebase/text/adapters.py`'s
`train_linear_probe_head`, full-batch Adam, backbone frozen); the trained head is then
stored exactly like a `target_task_heads` file would be, so the rest of the pipeline
(the alpha sweep's per-call head re-injection) is unaware anything changed. One
consequence specific to `steer_text`: Stage 1 reads `w_b` off B's live head at
`prepare()` time, which under this mode is still the untrained random init, not a real
head — rebasin happens first, the probe only fits afterward.

See `configs/text_rebase_t5base_t5large_theseus_linearprobe.json`,
`configs/text_rebase_t5base_flant5large_bico_linearprobe.json`, and
`configs/text_rebase_t5base_t5large_steer_linearprobe.json` for complete examples.

## Diagnosing a near-chance result

A near-chance accuracy through B's head is ambiguous on its own: it could be the
rebase method, the head, or the representation underneath both. Two controls
separate them.

`eval_source_finetuned: true` (or `--eval-source-finetuned`) scores **A's own
fine-tuned checkpoint with its own trained head** on the same task split, and
reports pooled-feature class separability for A fine-tuned, A pretrained, and
(via `scripts/build_nearest_mean_head.py`) B pretrained. Separability is the mean
cosine similarity among same-class pairs minus that among different-class pairs:

| Reading | Means |
|---|---|
| A fine-tuned gap large, A pretrained gap ~0 | Pooling and tokenization are fine. Class structure is created by fine-tuning, so a head built on B's *untrained* features has nothing to work with. |
| A fine-tuned gap ~0 but A still classifies well | The tensor being read as "the pooled feature" is not the one the classifier uses -- a pooling-position bug upstream. |
| Every gap ~0 and A classifies poorly too | The checkpoint, tokenization or task wiring is wrong, not the rebase method. |

This matters because every method here reads through B's head: `steer_text` fits
its Stage 2 correction as a function of B's pooled feature, so if that feature
carries no class signal the correction cannot be class-dependent either, and no
Stage 1 quality can recover it.

## Limits

- `steer_text` requires `eval_mode="head_logits"`.
- `steer_text`'s `stage_2_strategy="block_ridge"` requires `method_params.feature_regime="linear"`,
  which runs one forward-mode `jvp` per source transformer block per batch — affordable for a
  small model, expensive for a multi-billion-parameter one. `text_rebase.py` warns above a
  parameter-count threshold.
- `theseus` / `bico` default `method_params.seq_align` to `"interpolate"` (linear) instead of
  vision's `"interpolate2d"`, which assumes a square patch grid.
- Source and target must share the same tokenizer/vocabulary for the transported embedding
  delta to mean anything (true for same-family pairs: t5-v1_1-small/base, Qwen2.5-*).

## Testing

```bash
pytest tests/test_text_rebase.py -v
```

No GPU or real checkpoints required — the suite builds tiny in-memory T5/Qwen models via
`transformers`' own config classes.
