from __future__ import annotations

from typing import Any

import torch.nn as nn


def run(
    tasks: list[str],
    model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    num_fewshot: int = 0,
    batch_size: str = "auto",
    limit: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a causal-LM model on lm-eval harness tasks.

    Parameters
    ----------
    tasks : List of task names (e.g. "hellaswag", "piqa", etc.)
    model : A HuggingFace causal-LM model.
    tokenizer : Corresponding tokenizer.
    device : Target device string.
    num_fewshot : Number of few-shot examples.
    batch_size : Batch size ("auto" or int).
    limit : Optional max eval examples per task.

    Returns
    -------
    dict[str, float] : Per-task accuracy dict.
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        raise ImportError(
            "lm-eval harness requires `lm-eval` to be installed. "
            "Install with: pip install 'merge-and-rebase[harness]'"
        ) from None

    model.eval()
    if hasattr(model, "to"):
        model.to(device)

    try:
        results = simple_evaluate(
            model="hf",
            model_args={
                "pretrained": model,
                "tokenizer": tokenizer,
            },
            tasks=list(tasks),
            num_fewshot=int(num_fewshot),
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
    except TypeError:
        results = simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=list(tasks),
            num_fewshot=int(num_fewshot),
            batch_size=batch_size,
            device=device,
            limit=limit,
        )

    out: dict[str, float] = {}
    if results is None:
        return out

    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none")
        if acc is not None:
            out[task_name] = float(acc)
        acc_norm = task_results.get("acc_norm,none")
        if acc_norm is not None:
            out[f"{task_name}_norm"] = float(acc_norm)

    return out
