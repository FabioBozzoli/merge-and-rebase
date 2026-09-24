"""Geometry of B's mean-pooled block features, every block (from the feature cache).

Usage: python pooling_collapse.py <task>   (reads $BR_LABELFREE_ROOT/dump)
"""
import os
import sys, torch
from merge_and_rebase.rebase.methods.steer import _cache_split_dir, _load_cached_split

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")
bd = torch.load(f"{W}/dump/dump_{sys.argv[1]}/{sys.argv[1]}_prepare_inputs.pt", weights_only=False)
c = bd["cache"]
tr = _load_cached_split(_cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], sys.argv[1], c["feature_regime"], "train"), need_blocks=True)
B = {int(k): v.double() for k, v in tr["features_B_blocks"].items()}
y = tr["y_A"]
print(f"task={sys.argv[1]}  n={B[0].shape[0]}")
print("block  mean-share  cos(raw)  cos(centered)  top1-eig(cov)  top5-dims-energy  class-sep(centered)  cos(prev block)")
prev = None
for b in sorted(B):
    x = B[b]; mu = x.mean(0)
    mean_share = float(mu.square().sum() / x.square().sum(1).mean())
    xn = torch.nn.functional.normalize(x, dim=1); cos_raw = float((xn @ xn.T).mean())
    xc = x - mu; xcn = torch.nn.functional.normalize(xc, dim=1); cos_c = float((xcn @ xcn.T).mean())
    ev = torch.linalg.eigvalsh(xc.T @ xc / x.shape[0]); top1 = float(ev[-1] / ev.sum())
    e = x.square().mean(0); top5 = float(e.topk(5).values.sum() / e.sum())
    # between-class / total variance of centered features (how much of the spread is class-related)
    tot = float(xc.square().sum(1).mean())
    btw = sum(float((y == k).double().mean()) * float((xc[y == k].mean(0)).square().sum()) for k in y.unique())
    cp = "" if prev is None else f"{float((torch.nn.functional.normalize(prev, dim=1) * xn).sum(1).mean()):.3f}"
    tag = "f_B" if b == max(B) else str(b)
    print(f"{tag:>5s}  {mean_share:10.4f}  {cos_raw:8.4f}  {cos_c:13.4f}  {top1:13.4f}  {top5:16.4f}  {btw / tot:19.4f}  {cp:>15s}")
    prev = x
