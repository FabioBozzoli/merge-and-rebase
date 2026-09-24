"""Geometry of B's pooled block features under each pooling rule (train split).

Per block: share of energy in the dataset mean, share of variance on the top principal
direction, energy in the 5 largest channels, share of variance explained by class.

Usage: python pool_geometry.py <task>   (reads $BR_LABELFREE_ROOT/{dump,b_pooled})
"""
import os
import sys, torch
from merge_and_rebase.rebase.methods.steer import _cache_split_dir, _load_cached_split

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")

task = sys.argv[1]
bd = torch.load(f"{W}/dump/dump_{task}/{task}_prepare_inputs.pt", weights_only=False)
c = bd["cache"]
tr = _load_cached_split(_cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], "train"), need_blocks=True)
bp = torch.load(f"{W}/b_pooled/bpool_{task}/{task}_b_pooled.pt", weights_only=False)["train"]
y = tr["y_A"]
feats = {"mean": {int(k): v for k, v in tr["features_B_blocks"].items() if int(k) < 24}}
for p in ("unitnorm", "rmsnorm", "center_unitnorm", "center_rmsnorm"):
    feats[p] = {int(k): v for k, v in bp[p].items()}


def stats(x):
    x = x.double(); mu = x.mean(0)
    mean_share = float(mu.square().sum() / x.square().sum(1).mean())
    xc = x - mu
    ev = torch.linalg.eigvalsh(xc.T @ xc / x.shape[0])
    top1 = float(ev[-1] / ev.sum())
    e = x.square().mean(0); top5 = float(e.topk(5).values.sum() / e.sum())
    tot = float(xc.square().sum(1).mean())
    btw = sum(float((y == k).double().mean()) * float(xc[y == k].mean(0).square().sum()) for k in y.unique())
    return mean_share, top1, top5, btw / tot


print(f"task={task}   columns: mean-share / top1-direction share / top5-channel energy / class share")
for b in (0, 6, 12, 17, 23):
    print(f"block {b:2d}: " + "  ".join(f"{p}=" + "/".join(f"{v:.3f}" for v in stats(feats[p][b])) for p in feats))
fb = stats(tr["features_B"])
print("f_B     : " + "/".join(f"{v:.3f}" for v in fb))
