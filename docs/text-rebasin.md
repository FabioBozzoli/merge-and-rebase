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

There is no text equivalent of the ViT-only block-extension preprocess. If A and B have a
different number of transformer blocks, only name-matching parameters are transported and
the target's extra blocks keep their pretrained weights; `text_rebase.py` prints both depths
and a delta/target key-coverage report at the start of each task so this is visible, not
silent.

## Producing the inputs

- **Fine-tuned checkpoints and task heads** come from
  `python -m merge_and_rebase.finetune.train_text --text-config <config>` (see
  `finetune/configs/text.yaml` for a template). Set `output.extract_heads: true` to get a
  `heads.pt` usable as `target_task_heads` / `source_task_heads`.
- **`target_task_heads` must be heads trained *for B*.** The example configs use a
  linear-probe of B per task (`strategy.name: linear_probe`) — B's no-backbone-training head,
  the text analogue of CLIP's free zero-shot head.

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
