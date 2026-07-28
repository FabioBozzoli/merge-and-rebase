# Rebasin

Rebasin transports a task vector from a source base model to a target base model. The current evaluation entrypoint operates on one task vector at a time; merging and transport can be chained through checkpoint artifacts, but are not yet a single configurable multi-task pipeline.

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision8_gradfix_hf.json
```

Useful starter configurations include:

- [GradFix over Vision8](../configs/vision8_gradfix.json)
- [Theseus over Vision8](../configs/vision8_theseus_all.json)
- [TransFusion transport](../configs/vision8_transfusion_rebase.json)
- [BiCo alpha sweep](../configs/vision8_bico_all_alpha_sweep.json)

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
