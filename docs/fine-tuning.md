# Fine-Tuning

## Vision

Vision fine-tuning uses YAML files in `src/merge_and_rebase/finetune/configs/`.

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --datasets CIFAR10,CIFAR100,EuroSAT
```

Use `--suite vision8` to select a named benchmark suite. Outputs default to `src/checkpoints/finetune/<model>/<pretrained>/<task>/` and include checkpoints, summaries, and append-only event logs.

Available strategies are `full`, `linear_probe`, and `peft_lora`. Forward modes, including `linearized_ntk`, are configured through the strategy configuration. The main vision presets include `vision.yaml`, `vision-peft.yaml`, `vision-ntk.yaml`, and two-stage variants.

## Text

```bash
python -m merge_and_rebase.finetune.train_text \
  --text-config src/merge_and_rebase/finetune/configs/text-peft.yaml \
  --suite nli6
```

Text configurations support full fine-tuning, linear probing, LoRA adapters, task heads, and PEFT export.

## Regularization and Logging

Available regularizers are `distillation`, `kfac_ggn`, and `ekfac_ggn`. Configure one regularizer in the same fine-tuning configuration; its prepared state is rebuilt when relevant method parameters change.

All entrypoints write local structured logs. Configure optional Weights & Biases logging with:

```yaml
logging:
  use_wandb: false
  project: null
  entity: null
  tags: []
  mode: online
  local_log_dir: null
  log_every_n_steps: 50
```

See [methods](methods.md) for the runtime behavior of merge, transport, and post-merge methods.
