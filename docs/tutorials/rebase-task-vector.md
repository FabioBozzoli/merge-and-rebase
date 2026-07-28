# Tutorial: Transport a Task Vector

This tutorial transports task vectors from one pretrained OpenCLIP base to another with GradFix.

## 1. Inspect the Example

The [Vision8 GradFix configuration](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_gradfix_hf.json) specifies an OpenAI source base, a LAION target base, hosted checkpoints, and gradient-sign parameters.

## 2. Run Rebasin

```bash
python -m merge_and_rebase.eval.vision_rebase \
  --config configs/vision8_gradfix_hf.json
```

GradFix prepares target-model gradient signs from its data budget, then masks each transported task vector according to `mask_mode` and `vote`. `alpha: 1.0` applies the resulting transported update without an alpha sweep.

## 3. Adapt the Run

- Use `--tasks Cars,DTD` to restrict tasks.
- Change `method` to another registered transport method when its required preparation inputs are available.
- Enable `alpha_search` and choose `alpha_selection: "shared"` or `"per_task"` to select scales from validation results.

See [rebasin.md](../rebasin.md) and [configuration.md](../configuration.md) for the complete field reference.
