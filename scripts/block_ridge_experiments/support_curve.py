"""block_ridge learning curve across the interpolation threshold, label-free.

The pipeline never uses labels to fit: Stage 1 fits A's logits minus B's, Stage 2 fits
Stage 1's targets. So the support can be any unlabelled rows, and the ntk cache holds
4000 train rows per task (rte: 2490) -- more than the 2048 input dims of a
concat-grouped B block, which no fs<=500 run ever reached (n_S <= 1500 keeps every
intermediate block interpolating its support).

Per task and seed: shuffle the train rows (no labels), hold out the first N_VAL as V,
and draw nested supports of growing size from the rest. For each support, refit Stage 1
(production _stage1_projection) and every Stage-2 variant below; pick each variant's
penalty on V by logit-space R^2 against A's Stage-1 target (label-free), then report
test accuracy at alpha=1, at the alpha picked on V (by accuracy -- the one step that
uses labels), and test logit R^2.

Variants (all with reuse_logitmap per-block targets unless noted):
  br_asrun         lambda=1, concat grouping, independent  (the grid's block_ridge)
  br_trace         lambda_b = beta * tr(K_b)/n
  br_trace_carry   br_trace + smoothed_residual rho=1 (leftovers carried to later blocks)
  br_sumavg_trace  sum_avg grouping (1024-d blocks), trace lambda
  joint            one ridge on all trace-normalized concat blocks, total target
  global           ridge on normalized f_B, total target

Ridge is solved in kernel form from one eigendecomposition per (support, block): for
n <= d that is _ridge's own dual solution, and for n > d the primal solution _ridge
uses is the same function. Targets are carried in B's logit space (T W_B^T, C columns)
and mapped back with pinv(W_B): ridge is linear in its targets and every Stage-1 target
already lies in pinv(W_B)'s column space, so this is exact and ~300x cheaper. A guard checks br_asrun against production
_fit_block_ridge/_predict_block_ridge on the smallest support.

Usage: python support_curve.py <dump-dir> --task T [--seeds ...] [--sizes ...] [--out f.json]
"""
import argparse
import json
from pathlib import Path

import torch

from merge_and_rebase.rebase.methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _cache_split_dir,
    _fit_block_ridge,
    _load_cached_split,
    _predict_block_ridge,
    _stage1_projection,
)
from merge_and_rebase.rebase.text.steer_text import _accuracy

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--task", required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[33, 54, 89])
parser.add_argument("--sizes", type=int, nargs="+", default=[600, 1200, 2000, 2600, 3400])
parser.add_argument("--n-val", type=int, default=600)
parser.add_argument("--out", default=None)
args = parser.parse_args()

BETAS = [10.0, 1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
ALPHAS = [0.0] + [2.0**k for k in range(-10, 3)]

bd = torch.load(Path(args.dump_dir) / f"dump_{args.task}" / f"{args.task}_prepare_inputs.pt", weights_only=False)
task, c, mask = bd["task"], bd["cache"], bd["mask_class"]
cls = torch.tensor(sorted(mask))
w_a, w_b, b_b = bd["w_a"], bd["w_b"], bd["b_b"]
head = w_b[cls]
p_b = torch.linalg.pinv(w_b)
tr, te = (
    _load_cached_split(
        _cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], s),
        need_blocks=True,
    )
    for s in ("train", "test")
)
f_a, dA, f_b, dAb = (tr[k].double() for k in ("features_A", "delta_A", "features_B", "delta_A_blocks"))
y_tr, f_te, dA_te, y_te = tr["y_A"].long(), te["features_B"].double(), te["delta_A"].double(), te["y_A"].long()
L = dAb.shape[1]


def grouped(features_b_blocks, strategy):
    feats = {int(b): v.double() for b, v in features_b_blocks.items()}
    n_tgt = len(feats) - 1
    res = _BLOCK_GROUP_STRATEGIES[strategy]({b: feats[b] for b in range(n_tgt)}, L - 1)
    res[L - 1] = feats[n_tgt]
    return res


X = {g: (grouped(tr["features_B_blocks"], g), grouped(te["features_B_blocks"], g)) for g in ("concat", "sum_avg")}
del tr, te


def r2(y, p):
    return 1 - float(((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


class Kern:
    """Kernel ridge on one feature matrix: support S, cross-kernels to V and test."""

    def __init__(self, x_tr, x_te, S, V, scale=1.0):
        xs = x_tr[S] / scale
        self.K = xs @ xs.T
        self.s, self.U = torch.linalg.eigh(self.K)
        self.s = self.s.clamp_min(0)
        self.Kv = (x_tr[V] / scale) @ xs.T
        self.Kt = (x_te / scale) @ xs.T
        self.trace_mean = float(self.s.sum()) / xs.shape[0]

    def dual(self, T, lam):
        return self.U @ ((self.U.T @ T) / (self.s + lam)[:, None])


rows = []
for seed in args.seeds:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(f_b.shape[0], generator=g)
    V, pool = perm[: args.n_val], perm[args.n_val :]
    sizes = sorted({min(n, pool.numel()) for n in args.sizes})
    for n in sizes:
        S = pool[:n]
        M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=S, regularization=1.0)
        proj = lambda d: d @ M.T @ p_b.T  # noqa: E731
        T_S, T_V = proj(dA[S]), proj(dA[V])
        Tb_S = dAb[S] @ M.T @ p_b.T  # [n, L, d]
        yV, yT = T_V @ head.T, proj(dA_te) @ head.T
        # logit-space targets (exact, see docstring); predictions are mapped back with p_b
        T_Sl, Tb_Sl = T_S @ w_b.T, Tb_S @ w_b.T
        back = lambda z: z @ p_b.T  # noqa: E731
        acc = lambda f, p, y, a=1.0: _accuracy(f + a * p, w_b, b_b, y, mask_class=mask)  # noqa: E731
        row = {"task": task, "seed": seed, "n_support": n, "oracle": acc(f_te, proj(dA_te), y_te), "stage0": acc(f_te, torch.zeros_like(f_te), y_te)}

        kern = {gname: [Kern(X[gname][0][b], X[gname][1][b], S, V) for b in range(L)] for gname in X}

        def per_block(gname, lam_fn, carry):
            ks = kern[gname]
            pv = torch.zeros(V.numel(), T_Sl.shape[1], dtype=torch.float64)
            pt = torch.zeros(f_te.shape[0], T_Sl.shape[1], dtype=torch.float64)
            state = torch.zeros_like(T_Sl)
            for b, k in enumerate(ks):
                fitted = Tb_Sl[:, b] + (state if carry else 0)
                a = k.dual(fitted, lam_fn(k))
                if carry:
                    state = fitted - k.K @ a
                pv, pt = pv + k.Kv @ a, pt + k.Kt @ a
            return back(pv), back(pt)

        def total(k, lam):
            a = k.dual(T_Sl, lam)
            return back(k.Kv @ a), back(k.Kt @ a)

        # joint: sum of per-block kernels, each normalized by its own mean eigenvalue
        jk = kern["concat"]
        joint = Kern.__new__(Kern)
        joint.K = sum(k.K / k.trace_mean for k in jk)
        joint.s, joint.U = torch.linalg.eigh(joint.K)
        joint.s = joint.s.clamp_min(0)
        joint.Kv = sum(k.Kv / k.trace_mean for k in jk)
        joint.Kt = sum(k.Kt / k.trace_mean for k in jk)
        gscale = float(f_b[S].square().sum(1).mean().sqrt())
        glob = Kern(f_b, f_te, S, V, scale=gscale)

        variants = {
            "br_asrun": {None: per_block("concat", lambda k: 1.0, False)},
            "br_trace": {bt: per_block("concat", lambda k, bt=bt: bt * k.trace_mean, False) for bt in BETAS},
            "br_trace_carry": {bt: per_block("concat", lambda k, bt=bt: bt * k.trace_mean, True) for bt in BETAS},
            "br_sumavg_trace": {bt: per_block("sum_avg", lambda k, bt=bt: bt * k.trace_mean, False) for bt in BETAS},
            "joint": {bt: total(joint, bt) for bt in BETAS},
            "global": {bt: total(glob, bt) for bt in BETAS},
        }

        if n == sizes[0]:  # guard: kernel form == production block_ridge
            Xc_tr, Xc_te = X["concat"]
            coefs = _fit_block_ridge({b: Xc_tr[b][S] for b in range(L)}, Tb_S, selected=torch.arange(n),
                                     regularization=1.0, mode="independent")
            ref = _predict_block_ridge(coefs, Xc_te)
            gap = float((ref - variants["br_asrun"][None][1]).norm() / ref.norm())
            assert gap < 1e-6, f"kernel-form block_ridge differs from production ({gap:.2e})"
            row["guard_rel_err"] = gap

        for name, by_pen in variants.items():
            pen = max(by_pen, key=lambda p: r2(yV, by_pen[p][0] @ head.T))
            pv, pt = by_pen[pen]
            a_best = max(ALPHAS, key=lambda a: (acc(f_b[V], pv, y_tr[V], a), -a))
            row[name] = {
                "penalty": pen, "val_r2": r2(yV, pv @ head.T), "test_r2": r2(yT, pt @ head.T),
                "test_acc_a1": acc(f_te, pt, y_te), "val_alpha": a_best, "test_acc_valalpha": acc(f_te, pt, y_te, a_best),
            }
        rows.append(row)
        print(json.dumps({"task": task, "seed": seed, "n": n, "oracle": round(row["oracle"], 4),
                          **{k: (round(row[k]["test_acc_a1"], 4), round(row[k]["test_r2"], 3), row[k]["penalty"])
                             for k in variants}}), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1))
        del kern, joint, glob, variants
