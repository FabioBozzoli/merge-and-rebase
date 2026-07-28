# Tutorial: Reproduce a Vision8 Merge

This tutorial runs the released Task Arithmetic configuration with eight hosted fine-tuned ViT-B-32/OpenAI checkpoints.

## 1. Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[data]"
```

## 2. Run the Configuration

```bash
python -m merge_and_rebase.eval.vision_merge \
  --config configs/vision8_task_arithmetic_hf_release.json
```

The loader resolves the `hf-hub:` references in the configuration and downloads files as needed. The method prepares a Task Arithmetic direction once, applies `alpha: 0.3`, and evaluates the selected Vision8 test tasks.

## 3. Inspect Results

The command writes a JSON summary to its configured result location and prints task-level and aggregate metrics. Treat reported accuracies as raw test accuracy unless the result field explicitly says it is normalized.

For a downloaded bundle with manifest verification, use [artifacts.md](../artifacts.md).
