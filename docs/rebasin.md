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

For HuggingFace text models (T5, Qwen, Llama, ...) instead of CLIP, see [Text Rebasin](text-rebasin.md), which drives the same rebase methods through `eval/text_rebase.py`.
