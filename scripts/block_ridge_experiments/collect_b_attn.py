"""Collect B's per-block attention outputs (the input of SelfAttention.o), pooled, and exit.

The residual stream is cumulative and, mean-pooled, near-identical from block to block.
The attention output before W_O is not: each token's vector there is
sum_j a_tj v_j, so its mean over tokens weights the value vectors by how much attention
each token *receives* -- a token weighting computed by the model itself, per head, which
no per-token linear map of the residual stream can reproduce. (Anything per-token linear,
Q/K/V or W_O itself, commutes with mean pooling and adds nothing for a linear Stage 2.)

Pooled three ways, over real tokens:

  mean      masked mean of the [B, T, n_heads * d_head] input of `o`
  unitnorm  each token's full vector divided by its L2 norm first
  headnorm  each head's d_head slice divided by its own L2 norm first (heads with large
            norms no longer dominate), then scaled by 1/sqrt(n_heads) so tokens have unit norm

Saved as <local-log-dir>/<task>_b_attn.pt in collect_b_pooled.py's format
({split: {pooling: {block: [N, D]}}}), so preprocess_curve.py --b-pooled reads it as is.
Check printed: W_O applied to the mean-pooled input equals the mean-pooled output of `o`
(mean pooling is linear) -- the hook captured the tensor `o` actually consumes.

Launched as a text_rebase entrypoint:
    ENTRYPOINT=scripts.block_ridge_experiments.collect_b_attn \
      scripts/slurm/submit_text_rebase.sh <exp-name> <config> [key=value ...]
"""
import os
import runpy
import sys

import torch
import torch.nn as nn

import merge_and_rebase.rebase.text.steer_text as st

POOLINGS = ("mean", "unitnorm", "headnorm")
_O_PATHS = ("layer.0.SelfAttention.o",)  # T5Block; add others (e.g. self_attn.o_proj) when needed


def _attention_out_projections(model: nn.Module) -> list[tuple[nn.Linear, int]]:
    """(o, n_heads) for every block, in block_modules order."""
    found = []
    for block in st.block_modules(model):
        for path in _O_PATHS:
            try:
                o = block.get_submodule(path)
            except AttributeError:
                continue
            attn = block.get_submodule(path.rsplit(".", 1)[0])
            found.append((o, int(attn.n_heads)))
            break
        else:
            raise ValueError(f"no attention output projection ({_O_PATHS}) in block {type(block).__name__}")
    return found


def _collect(self, **kwargs):
    model = kwargs["llm_target"].model
    loaders = kwargs["target_loaders"]
    dev = torch.device(kwargs.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    projections = _attention_out_projections(model)
    model.eval()
    state = {"mask": None, "pooled": {}, "o_out": {}}

    def pre_hook(b, n_heads):
        def hook(_m, inputs):
            x = inputs[0]
            m = state["mask"]
            unit = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            heads = x.unflatten(-1, (n_heads, -1))
            headn = (heads / heads.norm(dim=-1, keepdim=True).clamp_min(1e-12)).flatten(-2) / n_heads**0.5
            state["pooled"].setdefault("mean", {})[b] = st._masked_mean(x, m).float().cpu()
            state["pooled"].setdefault("unitnorm", {})[b] = st._masked_mean(unit, m).float().cpu()
            state["pooled"].setdefault("headnorm", {})[b] = st._masked_mean(headn, m).float().cpu()
        return hook

    def post_hook(b):
        def hook(_m, _i, out):
            state["o_out"][b] = st._masked_mean(out, state["mask"]).double().cpu()
        return hook

    handles = []
    for b, (o, n_heads) in enumerate(projections):
        handles.append(o.register_forward_pre_hook(pre_hook(b, n_heads)))
        handles.append(o.register_forward_hook(post_hook(b)))

    payload = {"task": kwargs["task"], "source": "SelfAttention.o input"}
    try:
        with torch.no_grad(), st._head_as_identity(model):
            for split in ("train", "test"):
                acc = {p: {b: [] for b in range(len(projections))} for p in POOLINGS}
                worst = 0.0
                for batch in getattr(loaders, split):
                    state["mask"] = batch["attention_mask"].to(dev) if "attention_mask" in batch else None
                    state["pooled"], state["o_out"] = {}, {}
                    st._pooled_features(model, batch, dev)
                    for b, (o, _) in enumerate(projections):
                        for p in POOLINGS:
                            acc[p][b].append(state["pooled"][p][b])
                        via_pool = state["pooled"]["mean"][b].double() @ o.weight.detach().double().cpu().T
                        err = float((via_pool - state["o_out"][b]).norm() / state["o_out"][b].norm().clamp_min(1e-30))
                        worst = max(worst, err)
                payload[split] = {p: {b: torch.cat(v) for b, v in acc[p].items()} for p in POOLINGS}
                n = payload[split]["mean"][0].shape[0]
                print(f"[collect_b_attn] {split}: n={n}, blocks={len(projections)}, dim={payload[split]['mean'][0].shape[1]}; "
                      f"max rel err W_O(mean-pooled input) vs mean-pooled o output = {worst:.2e}", flush=True)
                payload[split]["checks"] = {"o_linearity": worst}
    finally:
        for h in handles:
            h.remove()

    out = os.path.join(log_dir, f"{kwargs['task']}_b_attn.pt")
    torch.save(payload, out)
    print(f"[collect_b_attn] wrote {out}", flush=True)
    raise SystemExit(0)


log_dir = sys.argv[sys.argv.index("--local-log-dir") + 1]
st.SteerTextRebase.prepare = _collect
sys.argv = ["text_rebase", *sys.argv[1:]]
runpy.run_module("merge_and_rebase.eval.text_rebase", run_name="__main__")
