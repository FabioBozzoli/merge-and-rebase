"""Oracle vs block_ridge(reuse) along alpha, seed 33 fs200: magnitude, quality, and flips."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, "/homes/pmoriello/merge-and-rebase/scripts/block_ridge_experiments")
from merge_and_rebase.rebase.methods.steer import (_BLOCK_GROUP_STRATEGIES, _cache_split_dir, _few_shot,
    _fit_block_ridge, _load_cached_split, _predict_block_ridge, _stage1_projection)

ALPHAS = [0, 1/64, 1/16, 1/4, 1/2, 1, 2, 4]
def grouped(fb, n):
    f = {int(b): v.double() for b, v in fb.items()}; nt = len(f) - 1
    r = _BLOCK_GROUP_STRATEGIES["concat"]({b: f[b] for b in range(nt)}, n); r[n] = f[nt]; return r

for bp in sorted(Path(sys.argv[1]).glob("*/*_prepare_inputs.pt")):
    bd = torch.load(bp, weights_only=False); task, c = bd["task"], bd["cache"]
    w_a, w_b, b_b, mask = bd["w_a"], bd["w_b"], bd["b_b"], bd["mask_class"]
    cls = torch.tensor(sorted(mask)); head = w_b[cls]; bias = b_b[cls] if b_b is not None else 0
    tr, te = (_load_cached_split(_cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task,
              c["feature_regime"], s), need_blocks=True) for s in ("train", "test"))
    f_a, dA, f_b, dAb = (tr[k].double() for k in ("features_A", "delta_A", "features_B", "delta_A_blocks"))
    fbt, dAt, y = te["features_B"].double(), te["delta_A"].double(), te["y_A"].long()
    L = dAb.shape[1]; Xtr, Xte = grouped(tr["features_B_blocks"], L - 1), grouped(te["features_B_blocks"], L - 1)
    sel = _few_shot(bd["local_labels_train"], 200, 33); p_b = torch.linalg.pinv(w_b)
    M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=sel, regularization=1.0)
    coefs = _fit_block_ridge({b: v[sel] for b, v in Xtr.items()}, dAb[sel] @ M.T @ p_b.T,
                             selected=torch.arange(sel.numel()), regularization=1.0, mode="independent")
    base = fbt @ head.T + bias                        # B's own logits (task classes)
    Y = (dAt @ M.T @ p_b.T) @ head.T                  # oracle correction, logit space
    P = _predict_block_ridge(coefs, Xte) @ head.T     # stage-2 correction, logit space
    yl = (y.unsqueeze(1) == cls).float().argmax(1)
    def margin(z):  # true-class logit minus best other
        t = z.gather(1, yl[:, None]).squeeze(1); o = z.clone(); o.scatter_(1, yl[:, None], -1e30); return t - o.max(1).values
    acc = lambda z: float((z.argmax(1) == yl).double().mean())
    r2 = 1 - float(((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    slope = float((P * Y).sum() / (P * P).sum())
    b_ok = margin(base) > 0
    print(f"\n== {task} (n_test={len(y)})  |oracle|/|B| = {float(Y.norm()/base.norm()):.2f}  |stage2|/|B| = {float(P.norm()/base.norm()):.2f}"
          f"  stage2 vs oracle: R2={r2:.3f} cos={float((P*Y).sum()/(P.norm()*Y.norm())):.3f} slope<P,Y>/<P,P>={slope:.3f}")
    cen = lambda z: z - z.mean(1, keepdim=True)
    print(f"   B logits: norm share of class-shared part = {float(1 - cen(base).norm()**2 / base.norm()**2):.4f}; "
          f"median |B margin| = {float(margin(base).abs().median()):.4g}, median |B logit| = {float(base.abs().median()):.4g}")
    print(f"   median |margin shift|: oracle = {float((margin(base+Y)-margin(base)).abs().median()):.4g}, stage2 = {float((margin(base+P)-margin(base)).abs().median()):.4g}"
          f"  -> ratio to median |B margin|: oracle {float((margin(base+Y)-margin(base)).abs().median()/margin(base).abs().median()):.1f}x, stage2 {float((margin(base+P)-margin(base)).abs().median()/margin(base).abs().median()):.1f}x")
    print(f"   corr(stage2 margin shift, oracle margin shift) = {float(torch.corrcoef(torch.stack([margin(base+P)-margin(base), margin(base+Y)-margin(base)]))[0,1]):.3f}")
    print("   alpha   oracle  stage2   | stage2: B-right->wrong  B-wrong->right")
    for a in ALPHAS:
        zs = base + a * P
        print(f"   {a:6.4f}  {acc(base + a*Y):.4f}  {acc(zs):.4f}   |   {float(((margin(zs) <= 0) & b_ok).double().mean()):.3f}                {float(((margin(zs) > 0) & ~b_ok).double().mean()):.3f}")
    # where do flips at small alpha happen? B's |margin| quartiles, alpha=1/16 vs 1
    q = torch.quantile(margin(base).abs(), torch.tensor([.25, .5, .75], dtype=torch.float64))
    bins = torch.bucketize(margin(base).abs(), q)
    for a in (1/16, 1):
        zs = base + a * P
        print(f"   net fixes by |B margin| quartile (low->high), alpha={a:g}: " +
              " ".join(f"{int(((margin(zs)>0)&~b_ok&(bins==k)).sum()) - int(((margin(zs)<=0)&b_ok&(bins==k)).sum()):+d}" for k in range(4)))
