# Tutorial: Merge Local Checkpoints

This workflow fine-tunes compatible tasks, then uses the resulting paths in a merge configuration.

## 1. Fine-Tune Tasks

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --datasets Cars,DTD
```

The two tasks must use the same base model and compatible parameterization. By default, outputs are under `src/checkpoints/finetune/<model>/<pretrained>/<task>/`.

## 2. Create a Configuration

Create `configs/local_task_arithmetic.json`:

```json
{
  "suite": "vision8",
  "clip_model": "ViT-B-32",
  "clip_pretrained": "openai",
  "method": "task_arithmetic",
  "alpha": 0.3,
  "tuned_ckpts": {
    "Cars": "src/checkpoints/finetune/ViT-B-32/openai/Cars/full_best_ep.pt",
    "DTD": "src/checkpoints/finetune/ViT-B-32/openai/DTD/full_best_ep.pt"
  }
}
```

Use the checkpoint filenames actually written by your selected strategy and save format.

## 3. Evaluate

```bash
python -m merge_and_rebase.eval.vision_merge \
  --config configs/local_task_arithmetic.json
```

Use [methods.md](../methods.md) to select a different method or [configuration.md](../configuration.md) to add alpha and parameter search.
