#!/usr/bin/env python
"""
Probe which text-pair input format a HF Hub sequence-classification checkpoint
actually expects, by scoring the same small sample under several plausible
tokenizations and comparing accuracy.

Motivation: varun-v-rao/t5-base-snli's model card claims 89.82% eval accuracy,
but evaluating it through this repo's standard tokenizer(premise, hypothesis)
pair encoding (build_nli_tokenized_loader's convention) measures ~35% --
chance for 3-way SNLI. Neither model card gives the exact training input
format, and this repo's pooled-feature diagnostic
(rebase/text/adapters.feature_separability) already showed the checkpoint's
own representations barely separate the classes under our encoding --
consistent with the input shape itself being wrong, not the model or the
rebase methods. google-t5/t5-base's own pretraining mixture used MNLI with the
convention "mnli hypothesis: {hypothesis} premise: {premise}" (for a
generative head, not a classification one, but plausibly imitated anyway) --
one of the candidates below.

This script tests each candidate format against the same fixed sample in one
pass, so the answer is a plain accuracy comparison, not another guess.
Deliberately standalone: it does not touch data/text_loaders.py (shared by
every other entrypoint) while the correct format is still unconfirmed.

Usage:
    python -m scripts.probe_nli_input_format \
        --model-name-or-path varun-v-rao/t5-base-snli \
        --task snli --num-examples 300 --device cuda
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from datasets import load_dataset


# Each candidate returns either (text_a, text_b) for a tokenizer text-pair
# call, or a single string for a one-segment call -- both are handled below.
def _pair_premise_first(premise: str, hypothesis: str) -> tuple[str, str]:
    return premise, hypothesis


def _pair_hypothesis_first(premise: str, hypothesis: str) -> tuple[str, str]:
    return hypothesis, premise


def _single_premise_hypothesis(premise: str, hypothesis: str) -> str:
    return f"premise: {premise} hypothesis: {hypothesis}"


def _single_hypothesis_premise(premise: str, hypothesis: str) -> str:
    return f"hypothesis: {hypothesis} premise: {premise}"


def _single_t5_mnli_style(premise: str, hypothesis: str) -> str:
    # google-t5/t5-base's own pretraining mixture used MNLI with exactly this
    # template (for a generative head, not a classification one -- plausible
    # if whoever fine-tuned t5-base-snli imitated the convention anyway).
    return f"mnli hypothesis: {hypothesis} premise: {premise}"


def _single_bare_concat(premise: str, hypothesis: str) -> str:
    return f"{premise} {hypothesis}"


CANDIDATES: dict[str, Callable[[str, str], tuple[str, str] | str]] = {
    "pair(premise, hypothesis)  [this repo's default]": _pair_premise_first,
    "pair(hypothesis, premise)  [swapped order]": _pair_hypothesis_first,
    "single 'premise: ... hypothesis: ...'": _single_premise_hypothesis,
    "single 'hypothesis: ... premise: ...'": _single_hypothesis_premise,
    "single 'mnli hypothesis: ... premise: ...'  [T5-paper style]": _single_t5_mnli_style,
    "single bare concat 'premise hypothesis'": _single_bare_concat,
}

_LABEL_NAMES = ("entailment", "neutral", "contradiction")


def load_sample(task: str, num_examples: int, seed: int) -> list[dict]:
    ds = load_dataset(task, split="test")
    ds = ds.filter(lambda ex: ex["label"] in (0, 1, 2))  # drop the -1 "no gold label" rows SNLI has
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))
    return [{"premise": ex["premise"], "hypothesis": ex["hypothesis"], "label": ex["label"]} for ex in ds]


@torch.no_grad()
def score_candidate(
    *,
    model,
    tokenizer,
    examples: list[dict],
    make_input: Callable[[str, str], tuple[str, str] | str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        made = [make_input(ex["premise"], ex["hypothesis"]) for ex in batch]
        if made and isinstance(made[0], tuple):
            text_a = [m[0] for m in made]
            text_b = [m[1] for m in made]
            enc = tokenizer(text_a, text_b, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        else:
            enc = tokenizer(made, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        if logits.shape[-1] != 3:
            raise ValueError(f"Expected a 3-way head, got logits shape {tuple(logits.shape)}.")
        preds = logits.argmax(dim=-1).cpu().tolist()
        labels = [ex["label"] for ex in batch]
        correct += sum(1 for p, y in zip(preds, labels, strict=True) if p == y)
    return correct / len(examples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name-or-path", type=str, required=True)
    p.add_argument("--task", type=str, default="snli", help="HF datasets id with premise/hypothesis/label fields.")
    p.add_argument("--num-examples", type=int, default=300)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    ).to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None)
    print(f"Model config id2label: {id2label}")
    if id2label is not None:
        recovered = tuple(str(id2label[i]).lower() for i in sorted(id2label, key=int))
        if recovered != _LABEL_NAMES:
            print(
                f"  NOTE: this differs from this repo's assumed order {_LABEL_NAMES} -- if it differs only by\n"
                "  permutation, that alone would produce a near-chance result independent of input format."
            )

    examples = load_sample(args.task, args.num_examples, args.seed)
    print(f"Loaded {len(examples)} examples from '{args.task}' test split.\n")

    print(f"{'candidate format':<55} accuracy")
    print("-" * 65)
    for name, make_input in CANDIDATES.items():
        acc = score_candidate(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            make_input=make_input,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        flag = "  <-- clearly above chance" if acc > 0.5 else ""
        print(f"{name:<55} {acc:.4f}{flag}")

    print(
        "\nChance for this label set is ~0.333. Whichever format clears that by a wide margin is (very likely) the\n"
        "one the checkpoint was trained on -- update the tokenization used by whatever config points at this\n"
        "checkpoint to match it."
    )


if __name__ == "__main__":
    main()
