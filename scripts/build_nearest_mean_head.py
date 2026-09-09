#!/usr/bin/env python
"""
Build a nearest-class-mean (cosine) classification head for a HuggingFace
sequence-classification model, saved in the ``heads.pt`` format
``eval/text_rebase.py`` reads via ``target_task_heads``/``source_task_heads``.

Each class row is the L2-normalized centroid of the model's own pooled
features over a few-shot support set:

    mu_c = normalize( mean_{x in support(c)} normalize(pooled_feature(x)) )

with bias forced to zero. A raw (non-normalized) dot product against these
rows -- the model's own head, exactly as ``TextLM.sequence_classification_accuracy``
computes it (``model(...).logits``, argmax) -- is then argmax-equivalent to
nearest-cosine classification: for a fixed query x, cos_sim(x, mu_c) =
(x . mu_c) / (|x| |mu_c|); every mu_c here has |mu_c| = 1, so cos_sim(x, mu_c)
= (x . mu_c) / |x|, and |x| is the same positive scalar for every class c.
It cannot change which class scores highest, so
argmax_c (x . mu_c) == argmax_c cos_sim(x, mu_c). No custom "cosine head"
module is needed: the raw-dot-product head_logits path already used by
``eval/text_rebase.py`` and ``eval/llm_merge.py`` gives exact nearest-mean
predictions once the weight rows are unit-norm and the bias is zero.

The pooled feature -- whatever tensor the model's real head would consume
(T5: post-dense-tanh; decoder-only: the last non-pad hidden state) -- is
obtained the same way ``rebase/text/steer_text.py`` already does it: swap the
head's final ``nn.Linear`` for ``nn.Identity`` (``_head_as_identity``) and
read the identity's output. That is the only place in the repo capable of
handing back a pre-head feature architecture-agnostically, so it is reused
here rather than reimplemented.

After building the head, this script injects it into the live model with
``_inject_task_head`` (the same helper ``text_rebase.py`` uses at eval time)
and reports accuracy on its own support set via
``TextLM.sequence_classification_accuracy`` -- the exact function
``text_rebase.py`` calls in ``eval_mode="head_logits"`` -- as a sanity check
that the saved weights reproduce the intended nearest-mean predictions
end-to-end, not just in isolation.

Usage:
    python -m scripts.build_nearest_mean_head \
        --model-name-or-path google/t5-v1_1-large --model-arch t5 \
        --task mnli --few-shot 8 --seed 33 \
        --output src/checkpoints/finetune_text/nearest_mean_heads/t5-v1_1-large_mnli_head.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from merge_and_rebase.data.text_loaders import (
    NLITaskData,
    build_nli_task_data,
    build_nli_tokenized_loader,
    default_head_class_ids_for_task,
)
from merge_and_rebase.eval.llm_merge import _inject_task_head
from merge_and_rebase.models.text_lm import TextBuildConfig, TextLM
from merge_and_rebase.rebase.text import balanced_indices, head_intermediate_linears, head_linear
from merge_and_rebase.rebase.text.steer_text import _head_as_identity, _pooled_features


def _neutralize_intermediate_head_layers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Overwrite every ``nn.Linear`` between the model's pretrained representation
    and the final classification ``nn.Linear`` (see
    :func:`~merge_and_rebase.rebase.text.head_intermediate_linears`) with an
    identity transform, in place, and return the tensors written.

    Necessary for a nearest-mean head: T5's ``T5ClassificationHead`` inserts a
    ``dense`` Linear + ``tanh`` between the decoder's pretrained eos-pooled
    hidden state and ``out_proj``. ``dense`` is never present in a base
    checkpoint (``AutoModelForSequenceClassification`` always initializes it
    randomly), so centroids built from its output are class means of a
    pretrained representation passed through an untrained random rotation --
    not of the representation itself. Setting it to the identity (square
    layers only; T5's is d_model -> d_model) makes the feature space
    ``pooled_feature(x)`` actually captures equal to ``tanh(pretrained_hidden)``,
    and doing it *before* feature extraction (not just at save time) keeps
    centroid construction and later injection in the same space. A no-op for
    architectures with no such layer (e.g. a bare decoder-only ``score``
    Linear): :func:`head_intermediate_linears` returns ``[]`` for those.
    """
    written: dict[str, torch.Tensor] = {}
    for name, module in head_intermediate_linears(model):
        if module.weight.shape[0] != module.weight.shape[1]:
            raise ValueError(
                f"Cannot neutralize non-square intermediate head layer '{name}' "
                f"(shape {tuple(module.weight.shape)}) to an identity transform."
            )
        eye = torch.eye(module.weight.shape[0], dtype=module.weight.dtype, device=module.weight.device)
        with torch.no_grad():
            module.weight.copy_(eye)
            written[f"{name}.weight"] = eye.detach().cpu()
            if module.bias is not None:
                module.bias.zero_()
                written[f"{name}.bias"] = module.bias.detach().cpu().clone()
    return written


def _select_few_shot(task_data: NLITaskData, *, per_class: int, seed: int, max_candidates: int | None) -> NLITaskData:
    examples = task_data.examples
    if max_candidates is not None and len(examples) > max_candidates:
        # balanced_indices only needs a candidate pool a few times larger than
        # per_class * num_classes; capping it keeps loading a huge split (e.g.
        # MNLI train, ~393k rows) from dominating runtime. NLI datasets are not
        # label-sorted, so a head slice stays balanced.
        examples = examples[:max_candidates]

    local_labels = [int(ex.label) for ex in examples]
    selected = balanced_indices(local_labels, per_class, seed=seed)
    counts = {c: 0 for c in range(len(task_data.labels))}
    for idx in selected:
        counts[local_labels[idx]] += 1
    short = {c: n for c, n in counts.items() if n < per_class}
    if short:
        raise ValueError(
            f"Not enough support examples for classes {short} (need {per_class} each); "
            "raise --max-candidates or lower --few-shot."
        )

    return NLITaskData(
        task=task_data.task,
        examples=[examples[i] for i in selected],
        labels=list(task_data.labels),
        label_texts=list(task_data.label_texts),
        meta={**task_data.meta, "num_examples": len(selected), "few_shot_per_class": per_class},
    )


def build_head(
    *,
    model_name_or_path: str,
    model_arch: str,
    task: str,
    few_shot: int,
    seed: int,
    num_labels: int,
    max_length: int,
    max_candidates: int | None,
    device: str,
    dtype: str | None,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
    batch_size: int = 16,
) -> tuple[dict[str, torch.Tensor], dict[str, object], TextLM]:
    torch.manual_seed(seed)

    build_cfg = TextBuildConfig(
        model_name_or_path=model_name_or_path,
        model_arch=model_arch,
        device=device,
        dtype=dtype,
        model_kind="sequence_classification",
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
    )
    llm = TextLM.build(build_cfg)

    neutralized = _neutralize_intermediate_head_layers(llm.model)
    if neutralized:
        print(
            f"Neutralized {len(neutralized) // 2} untrained intermediate head layer(s) to identity "
            f"before feature extraction: {sorted({k.rsplit('.', 1)[0] for k in neutralized})}"
        )

    head_class_ids = default_head_class_ids_for_task(task, num_labels=num_labels)
    task_data = build_nli_task_data(task=task, split="train")
    print(f"Loaded {len(task_data.examples)} '{task}' train examples ({task_data.labels}).")

    few_shot_data = _select_few_shot(task_data, per_class=few_shot, seed=seed, max_candidates=max_candidates)
    print(f"Selected {len(few_shot_data.examples)} support examples ({few_shot} per class, seed={seed}).")

    tokenized = build_nli_tokenized_loader(
        task_data=few_shot_data,
        tokenizer=llm.tokenizer,
        batch_size=int(batch_size),
        num_workers=0,
        max_length=max_length,
        shuffle=False,
        head_class_ids=head_class_ids,
    )

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    llm.model.eval()
    # Mini-batched on purpose: a single forward pass over the whole support
    # set (previously batch_size=len(support), i.e. everything at once) OOMs
    # once few_shot grows past a handful of shots -- 300/class on a 3-way
    # task is a 900-example batch through t5-v1_1-large in one shot. Looping
    # over torch's own DataLoader batches and concatenating on CPU also fixes
    # a latent correctness bug: a single next(iter(loader)) call only reads
    # the *first* batch, silently dropping the rest of the support set for
    # any batch_size smaller than the whole support.
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad(), _head_as_identity(llm.model):
        for batch in tokenized.loader:
            feats = _pooled_features(llm.model, batch, dev).cpu().double()
            feature_chunks.append(feats)
            label_chunks.append(batch["labels"].cpu())
    features = torch.cat(feature_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)

    normed = torch.nn.functional.normalize(features, dim=-1)
    centroids = torch.zeros(num_labels, features.shape[-1], dtype=torch.float64)
    for class_id in range(num_labels):
        rows = normed[labels == class_id]
        if rows.shape[0] == 0:
            raise ValueError(f"No support examples ended up in class {class_id} after tokenization.")
        centroids[class_id] = torch.nn.functional.normalize(rows.mean(dim=0), dim=-1)

    head_name, head_module = head_linear(llm.model)
    weight = centroids.to(dtype=head_module.weight.dtype, device="cpu")
    payload: dict[str, torch.Tensor] = {f"{head_name}.weight": weight}
    if head_module.bias is not None:
        payload[f"{head_name}.bias"] = torch.zeros(num_labels, dtype=head_module.bias.dtype)
    # Bake the same neutralization into the saved head: injecting it at eval
    # time (_inject_task_head) must reproduce the exact feature space the
    # centroids above were computed in, or the two disagree again.
    payload.update(neutralized)

    meta = {
        "model_name_or_path": model_name_or_path,
        "model_arch": model_arch,
        "task": task,
        "few_shot": few_shot,
        "seed": seed,
        "num_labels": num_labels,
        "head_class_ids": head_class_ids,
        "head_param_prefix": head_name,
        "neutralized_intermediate_layers": sorted({k.rsplit(".", 1)[0] for k in neutralized}),
        "construction": "nearest_class_mean_cosine",
        "note": (
            "weight rows are L2-normalized class centroids of the model's own pooled "
            "features over the support set, computed after neutralizing any untrained "
            "intermediate head layers (see neutralized_intermediate_layers) to identity; "
            "bias is zero throughout. A raw dot product against these rows is "
            "argmax-equivalent to cosine nearest-mean classification."
        ),
    }

    # Sanity check end to end: inject the head we just built into the live
    # model with the same helper text_rebase.py uses at eval time, then score
    # it on its own support set through TextLM.sequence_classification_accuracy
    # -- the exact function text_rebase.py calls in eval_mode="head_logits".
    # A saved-tensor round trip could silently disagree with this (dtype
    # cast, a bias the head expects but we didn't zero, wrong param name);
    # this catches that before the file is written.
    _inject_task_head(
        model=llm.model,
        task=task,
        task_heads={task: payload},
        head_key_pattern=head_name.rsplit(".", 1)[0] if "." in head_name else head_name,
        head_class_ids=head_class_ids,
    )
    support_acc = llm.sequence_classification_accuracy(
        tokenized.loader, device=device, mask_class=tokenized.mask_class
    )
    print(f"Support-set accuracy after injecting the built head (sanity check, not a generalization estimate): {support_acc:.4f}")
    meta["support_set_accuracy"] = support_acc

    return payload, meta, llm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name-or-path", type=str, required=True, help="The model whose pooled features define the centroids (usually the target base model B).")
    p.add_argument("--model-arch", type=str, default="auto", choices=["llama", "t5", "auto"])
    p.add_argument("--task", type=str, required=True, help="One of the nli6 tasks (snli, mnli, sick, qnli, rte, scitail).")
    p.add_argument("--few-shot", type=int, required=True, help="Support examples per class used to build each centroid.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-labels", type=int, default=3, help="Head width; must match the model_kind='sequence_classification' build used at eval time.")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-candidates", type=int, default=20000, help="Cap on train rows loaded before balanced sampling (0 = no cap).")
    p.add_argument("--batch-size", type=int, default=16, help="Forward-pass batch size for feature extraction and the sanity-check eval; lower this if you hit CUDA OOM with a large --few-shot.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default=None, choices=[None, "fp16", "bf16", "fp32"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-fast-tokenizer", action="store_true")
    p.add_argument("--output", type=str, required=True, help="Where to write the heads.pt (a {task: {param_name: tensor}} dict).")
    args = p.parse_args()

    max_candidates = None if args.max_candidates in (0, None) else int(args.max_candidates)
    payload, meta, _llm = build_head(
        model_name_or_path=args.model_name_or_path,
        model_arch=args.model_arch,
        task=str(args.task).strip().lower(),
        few_shot=args.few_shot,
        seed=args.seed,
        num_labels=args.num_labels,
        max_length=args.max_length,
        max_candidates=max_candidates,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=not args.no_fast_tokenizer,
        batch_size=args.batch_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heads = {str(args.task).strip().lower(): {**payload, "_meta": meta}}
    torch.save(heads, out_path)
    print(f"Wrote nearest-mean-cosine head for task '{args.task}' -> {out_path}")
    for name, tensor in payload.items():
        print(f"  {name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
