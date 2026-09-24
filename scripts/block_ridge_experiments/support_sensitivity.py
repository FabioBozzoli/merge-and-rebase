"""How much does joint ridge depend on *which* rows form the support?

Refits joint ridge (given pooling / zscore / lambda) and global_ridge (given beta) on four
supports per seed: the first 3400 rows of the seeded permutation (what production's
_random_sample draws), the last 3400 (what the offline curves used), a middle 3400, and all
4000 rows. Also reproduces production's cached stage-2 accuracy exactly on the first draw.

Usage: python support_sensitivity.py <task> <pooling> <zscore 0|1> <joint lambda> <global beta>
"""
import os
import torch
from merge_and_rebase.rebase.methods.steer import _BLOCK_GROUP_STRATEGIES, _cache_split_dir, _load_cached_split, _ridge, _stage1_projection, _random_sample
from merge_and_rebase.rebase.text.steer_text import _accuracy

def run(task, pooling, zscore, lam, support):
    bd = torch.load(f"{W}/dump/dump_{task}/{task}_prepare_inputs.pt", weights_only=False); c = bd["cache"]
    tr, te = (_load_cached_split(_cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], s), need_blocks=True) for s in ("train", "test"))
    bp = torch.load(f"{W}/b_pooled/bpool_{task}/{task}_b_pooled.pt", weights_only=False)
    w_a, w_b, b_b, mask = bd["w_a"], bd["w_b"], bd["b_b"], bd["mask_class"]
    f_a, dA, f_b = (tr[k].double() for k in ("features_A", "delta_A", "features_B"))
    def blocks(split, d):
        f = {int(k): v.double() for k, v in d["features_B_blocks"].items()}
        for b, v in bp[split][pooling].items(): f[int(b)] = v.double()
        g = _BLOCK_GROUP_STRATEGIES["concat"]({b: f[b] for b in range(24)}, 12); g[12] = f[24]; return g
    Xtr, Xte = blocks("train", tr), blocks("test", te)
    S = support
    M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=S, regularization=1.0)
    p_b = torch.linalg.pinv(w_b); T = dA[S] @ M.T @ p_b.T
    xs = {b: v[S] for b, v in Xtr.items()}; xt = dict(Xte)
    if zscore:
        for b in xs:
            mu, sd = xs[b].mean(0), xs[b].std(0); sd = sd.clamp_min(1e-2 * float(sd.median()))
            xs[b], xt[b] = (xs[b] - mu) / sd, (xt[b] - mu) / sd
    sc = [float(xs[b].square().sum(1).mean().sqrt()) for b in range(13)]
    zs = torch.cat([xs[b] / sc[b] for b in range(13)], 1); zt = torch.cat([xt[b] / sc[b] for b in range(13)], 1)
    m = T.mean(0) if zscore else 0
    P = zt @ _ridge(zs, T - m, lam) + m
    return _accuracy(te["features_B"].double() + P, w_b, b_b, te["y_A"].long(), mask_class=mask)

def run_global(task, lam_beta, support):
    bd = torch.load(f"{W}/dump/dump_{task}/{task}_prepare_inputs.pt", weights_only=False); c = bd["cache"]
    tr, te = (_load_cached_split(_cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], s), need_blocks=False) for s in ("train", "test"))
    w_a, w_b, b_b, mask = bd["w_a"], bd["w_b"], bd["b_b"], bd["mask_class"]
    f_a, dA, f_b = (tr[k].double() for k in ("features_A", "delta_A", "features_B"))
    M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=support, regularization=1.0)
    p_b = torch.linalg.pinv(w_b); T = dA[support] @ M.T @ p_b.T
    lam = lam_beta * float(f_b[support].square().sum(1).mean())
    P = te["features_B"].double() @ _ridge(f_b[support], T, lam)
    return _accuracy(te["features_B"].double() + P, w_b, b_b, te["y_A"].long(), mask_class=mask)

import sys

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")
task, pooling, zscore, lam, gbeta = sys.argv[1], sys.argv[2], sys.argv[3] == "1", float(sys.argv[4]), float(sys.argv[5])
n = 4000
for seed in (33, 54, 89):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    supports = {"first3400": perm[:3400], "last3400": perm[600:], "middle3400": torch.cat([perm[:300], perm[900:]]), "all4000": perm}
    out = {k: (run(task, pooling, zscore, lam, S), run_global(task, gbeta, S)) for k, S in supports.items()}
    print(f"{task} seed {seed}: " + "  ".join(f"{k}: joint={a:.4f} global={g:.4f}" for k, (a, g) in out.items()), flush=True)
