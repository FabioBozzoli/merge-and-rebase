# Tutorial: Post-Merge Adaptation

Post-merge methods adapt an already merged vision model. This example uses AdaMerging to optimize bounded task-level coefficients against the configured entropy objective.

## 1. Start from an Existing Merge Configuration

Use [configs/vision8_task_arithmetic_adamerging_alpha03.json](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_task_arithmetic_adamerging_alpha03.json), or add this block to another compatible vision merge configuration:

```json
{
  "postmerge": {
    "method": "adamerging",
    "alpha_mode": "task",
    "steps": 500,
    "lr": 0.001,
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "init_alpha": 0.3
  }
}
```

## 2. Run It

```bash
python -m merge_and_rebase.eval.vision_merge \
  --config configs/vision8_task_arithmetic_adamerging_alpha03.json
```

The evaluator first builds the configured merge, then runs the post-merge method and records its metadata in the result summary. Disable `alpha_search` and `hyperparam_search` when using a post-merge block.

## 3. Choose a Method

`adamerging` optimizes task- or layer-level coefficients. `task_vector_finetune`, `merged_delta_finetune`, and `vision_head_probe` use vision training objectives instead. Their scope and restrictions are listed in [methods.md](../methods.md).
