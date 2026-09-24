"""Recollect B's per-block features with normalize-then-pool, and exit before prepare().

steer_text's cache holds B's block outputs mean-pooled over tokens straight from the
unnormalized residual stream. f_B instead goes through the encoder's final RMSNorm token
by token before pooling. This reruns B (a plain forward -- nothing here needs A's jvps)
over the exact loaders steer_text uses and pools every block three ways:

  mean       masked mean of the raw block output         (what the cache holds)
  unitnorm   each token divided by its L2 norm, then masked mean
  rmsnorm    each token through B's own final_layer_norm (T5 RMSNorm, learned gain), then masked mean
  center_unitnorm  (x - mu_b) / ||x - mu_b|| per token, then masked mean
  center_rmsnorm   final_layer_norm(x - mu_b) per token, then masked mean

mu_b is block b's mean token vector over every real token of the *train* split
(unlabelled; test never enters it), computed in a first pass over the train loader.
Centering first means each token contributes its deviation from the typical token at
unit scale, instead of a vector dominated by the component all tokens share.
center_rmsnorm is no longer the model's own operation (f_B is not centered); it only
reuses RMSNorm's learned per-channel gain.

Everything but `mean` (which the cache already holds) is saved to
<local-log-dir>/<task>_b_pooled.pt, with mu_b. Two checks are printed:
`mean` against the cached features_B_blocks (proves rows line up with the cache), and
`rmsnorm` of the last block against f_B (the model's own pooled feature).

Launched as a text_rebase entrypoint:
    ENTRYPOINT=scripts.block_ridge_experiments.collect_b_pooled \
      scripts/slurm/submit_text_rebase.sh <exp-name> <config> [key=value ...]
"""
import os
import runpy
import sys

import torch

import merge_and_rebase.rebase.text.steer_text as st
from merge_and_rebase.rebase.methods.steer import _cache_split_dir, _load_cached_split

POOLINGS = ("mean", "unitnorm", "rmsnorm", "center_unitnorm", "center_rmsnorm")


def _collect(self, **kwargs):
    model = kwargs["llm_target"].model
    loaders = kwargs["target_loaders"]
    dev = torch.device(kwargs.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    final_norm = model.transformer.encoder.final_layer_norm
    blocks = st.block_modules(model)
    model.eval()

    state: dict = {"mask": None, "pooled": {}, "mode": "mu", "sum": {}, "count": {}, "mu": {}}
    unit = lambda z: z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # noqa: E731

    def make_hook(b):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            m = state["mask"]
            mk = torch.ones(h.shape[:2], device=h.device) if m is None else m.to(h.dtype)
            if state["mode"] == "mu":
                state["sum"][b] = state["sum"].get(b, 0) + (h.double() * mk[..., None]).sum((0, 1))
                state["count"][b] = state["count"].get(b, 0) + float(mk.sum())
                return
            hc = h - state["mu"][b].to(h.dtype)
            pooled = state["pooled"]
            pooled.setdefault("mean", {})[b] = st._masked_mean(h, m).float().cpu()
            pooled.setdefault("unitnorm", {})[b] = st._masked_mean(unit(h), m).float().cpu()
            pooled.setdefault("rmsnorm", {})[b] = st._masked_mean(final_norm(h), m).float().cpu()
            pooled.setdefault("center_unitnorm", {})[b] = st._masked_mean(unit(hc), m).float().cpu()
            pooled.setdefault("center_rmsnorm", {})[b] = st._masked_mean(final_norm(hc), m).float().cpu()
        return hook

    payload = {"task": kwargs["task"]}
    handles = [mod.register_forward_hook(make_hook(b)) for b, mod in enumerate(blocks)]
    try:
        with torch.no_grad(), st._head_as_identity(model):
            # pass 1: mu_b over every real train token
            for batch in loaders.train:
                state["mask"] = batch["attention_mask"].to(dev) if "attention_mask" in batch else None
                st._pooled_features(model, batch, dev)
            state["mu"] = {b: (state["sum"][b] / state["count"][b]).float() for b in range(len(blocks))}
            payload["mu"] = {b: v.cpu() for b, v in state["mu"].items()}
            state["mode"] = "pool"
            for split in ("train", "test"):
                acc = {p: {b: [] for b in range(len(blocks))} for p in POOLINGS}
                f_b = []
                for batch in getattr(loaders, split):
                    state["mask"] = batch["attention_mask"].to(dev) if "attention_mask" in batch else None
                    state["pooled"] = {}
                    f_b.append(st._pooled_features(model, batch, dev).float().cpu())
                    for p in POOLINGS:
                        for b in range(len(blocks)):
                            acc[p][b].append(state["pooled"][p][b])
                payload[split] = {p: {b: torch.cat(v) for b, v in acc[p].items()} for p in POOLINGS}
                payload[split]["f_B"] = torch.cat(f_b)
    finally:
        for h in handles:
            h.remove()

    # Checks against the steer_text cache and against f_B.
    c = kwargs
    for split in ("train", "test"):
        cached = _load_cached_split(
            _cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], c["task"], c["feature_regime"], split),
            need_blocks=True,
        )
        cb = cached["features_B_blocks"]
        last = len(blocks) - 1
        mean_err = max(
            float((payload[split]["mean"][b] - cb[b].float()).norm() / cb[b].float().norm()) for b in range(len(blocks))
        )
        fb_err = float((payload[split]["f_B"] - cached["features_B"].float()).norm() / cached["features_B"].float().norm())
        rms_err = float((payload[split]["rmsnorm"][last] - payload[split]["f_B"]).norm() / payload[split]["f_B"].norm())
        print(f"[collect_b_pooled] {split}: n={payload[split]['f_B'].shape[0]} "
              f"max rel err mean-pool vs cache = {mean_err:.2e}; f_B vs cache = {fb_err:.2e}; "
              f"rmsnorm(block {last}) vs f_B = {rms_err:.2e}", flush=True)
        payload[split]["checks"] = {"mean_vs_cache": mean_err, "fB_vs_cache": fb_err, "rmsnorm_last_vs_fB": rms_err}

    for split in ("train", "test"):
        del payload[split]["mean"]  # the cache holds it; only used for the check above
    out = os.path.join(log_dir, f"{kwargs['task']}_b_pooled.pt")
    torch.save(payload, out)
    print(f"[collect_b_pooled] wrote {out}", flush=True)
    raise SystemExit(0)


log_dir = sys.argv[sys.argv.index("--local-log-dir") + 1]
st.SteerTextRebase.prepare = _collect
sys.argv = ["text_rebase", *sys.argv[1:]]
runpy.run_module("merge_and_rebase.eval.text_rebase", run_name="__main__")
