"""Does conditioning B's pooled block features help block_ridge? Label-free, offline.

Mean-pooled intermediate blocks of t5-large are badly conditioned: 80-96% of each
vector's squared norm is the dataset mean, one principal direction carries up to 98%
of the remaining variance, and five channels hold about half the energy. f_B, which
goes through the final RMSNorm before pooling, is not. This refits the block variants
of support_curve.py after preprocessing every block (and f_B, for global) with
statistics from the support rows only:

  none          raw pooled features (reproduces support_curve.py)
  center        subtract the support mean; ridge with an unpenalized intercept
  zscore        center, then divide each channel by its support std (floored)
  center_drop1  center, then project out the support's top principal direction
  rownorm       LayerNorm over channels per example (no affine), then center

Same protocol as support_curve.py: seeded label-free shuffle, first N_VAL rows held
out as V, nested supports; penalty picked on V by logit-space R^2 against A's Stage-1
target; test accuracy at alpha=1 and at the alpha picked on V (by accuracy).

--b-pooled FILE --pooling P swaps B's per-block features (not f_B) for the ones
collect_b_pooled.py recollected with another pooling rule (unitnorm, rmsnorm and their
centered versions),
so global -- which only sees f_B -- stays an unchanged reference. collect_b_attn.py's file
(attention outputs, poolings mean / unitnorm / headnorm) has the same format.

--add-pooled FILE --add-pooling P appends a *second* set of B blocks (grouped the same way)
to the joint ridge only, reported as the extra variant "joint_plus": does that source carry
information the first one lacks? The per-block fits keep their one-to-one map to A's blocks.

Usage: python preprocess_curve.py <dump-dir> --task T [--seeds ...] [--sizes ...] [--out f.json]
                                  [--b-pooled <task>_b_pooled.pt --pooling rmsnorm]
"""
import argparse
import json
from pathlib import Path

import torch

from merge_and_rebase.rebase.methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _cache_split_dir,
    _load_cached_split,
    _stage1_projection,
)
from merge_and_rebase.rebase.text.steer_text import _accuracy

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--task", required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[33, 54, 89])
parser.add_argument("--sizes", type=int, nargs="+", default=[600, 1500, 3400])
parser.add_argument("--n-val", type=int, default=600)
parser.add_argument("--preps", nargs="+", default=["none", "center", "zscore", "center_drop1", "rownorm"])
parser.add_argument("--out", default=None)
parser.add_argument("--b-pooled", default=None)
parser.add_argument("--pooling", default="rmsnorm",
                    choices=["mean", "unitnorm", "rmsnorm", "center_unitnorm", "center_rmsnorm", "headnorm"])
parser.add_argument("--add-pooled", default=None)
parser.add_argument("--add-pooling", default="unitnorm",
                    choices=["mean", "unitnorm", "rmsnorm", "center_unitnorm", "center_rmsnorm", "headnorm"])
args = parser.parse_args()

BETAS = [1000.0, 100.0, 10.0, 1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
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
if args.b_pooled:
    bp = torch.load(args.b_pooled, weights_only=False)
    for split, d in (("train", tr), ("test", te)):
        keys = {int(k): k for k in d["features_B_blocks"]}
        for b, v in bp[split][args.pooling].items():
            old = d["features_B_blocks"][keys[int(b)]]
            assert v.shape == old.shape, (split, b, v.shape, old.shape)
            d["features_B_blocks"][keys[int(b)]] = v
    print(f"[preprocess_curve] B blocks replaced with '{args.pooling}' pooling from {args.b_pooled}; "
          f"checks: {bp['train']['checks']}", flush=True)


def grouped(features_b_blocks):
    feats = {int(b): v.double() for b, v in features_b_blocks.items()}
    n_tgt = len(feats) - 1
    res = _BLOCK_GROUP_STRATEGIES["concat"]({b: feats[b] for b in range(n_tgt)}, L - 1)
    res[L - 1] = feats[n_tgt]
    return res


X_tr, X_te = grouped(tr["features_B_blocks"]), grouped(te["features_B_blocks"])
ADD_tr = ADD_te = None
if args.add_pooled:
    ap = torch.load(args.add_pooled, weights_only=False)
    n_add = len(ap["train"][args.add_pooling])
    group = lambda d: _BLOCK_GROUP_STRATEGIES["concat"]({int(b): v.double() for b, v in d.items()}, L - 1)  # noqa: E731
    ADD_tr, ADD_te = group(ap["train"][args.add_pooling]), group(ap["test"][args.add_pooling])
    assert ADD_tr[0].shape[0] == X_tr[0].shape[0] and ADD_te[0].shape[0] == X_te[0].shape[0]
    print(f"[preprocess_curve] joint_plus adds {n_add} '{args.add_pooling}' blocks from {args.add_pooled} "
          f"as {len(ADD_tr)} groups", flush=True)
    del ap
del tr, te


def r2(y, p):
    return 1 - float(((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def rownorm(x):
    x = x - x.mean(1, keepdim=True)
    return x / x.std(1, keepdim=True).clamp_min(1e-12)


def preprocess(x_tr, x_te, S, V, prep):
    """(support, V, test) matrices after `prep`, with statistics from the support only."""
    xs, xv, xt = x_tr[S], x_tr[V], x_te
    if prep == "none":
        return xs, xv, xt
    if prep == "rownorm":
        xs, xv, xt = rownorm(xs), rownorm(xv), rownorm(xt)
    mu = xs.mean(0)
    xs, xv, xt = xs - mu, xv - mu, xt - mu
    if prep == "zscore":
        sd = xs.std(0)
        sd = sd.clamp_min(1e-2 * float(sd.median()))
        xs, xv, xt = xs / sd, xv / sd, xt / sd
    elif prep == "center_drop1":
        u = torch.linalg.svd(xs, full_matrices=False).Vh[0]
        drop = lambda z: z - (z @ u)[:, None] * u[None, :]  # noqa: E731
        xs, xv, xt = drop(xs), drop(xv), drop(xt)
    return xs, xv, xt


class Kern:
    """Kernel ridge on preprocessed support/V/test matrices; intercept when centered."""

    def __init__(self, xs, xv, xt, intercept, scale=1.0):
        xs, xv, xt = xs / scale, xv / scale, xt / scale
        self.K, self.Kv, self.Kt = xs @ xs.T, xv @ xs.T, xt @ xs.T
        self.s, self.U = torch.linalg.eigh(self.K)
        self.s = self.s.clamp_min(0)
        self.trace_mean = float(self.s.sum()) / xs.shape[0]
        self.intercept = intercept

    @classmethod
    def from_kernels(cls, K, Kv, Kt, intercept):
        k = cls.__new__(cls)
        k.K, k.Kv, k.Kt, k.intercept = K, Kv, Kt, intercept
        k.s, k.U = torch.linalg.eigh(K)
        k.s = k.s.clamp_min(0)
        k.trace_mean = float(k.s.sum()) / K.shape[0]
        return k

    def fit(self, T, lam):
        """Dual coefficients and intercept; the intercept is not penalized."""
        m = T.mean(0) if self.intercept else torch.zeros(T.shape[1], dtype=T.dtype)
        a = self.U @ ((self.U.T @ (T - m)) / (self.s + lam)[:, None])
        return a, m


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
        yV, yT = proj(dA[V]) @ head.T, proj(dA_te) @ head.T
        # logit-space targets (exact; see support_curve.py), mapped back with p_b
        T_Sl = proj(dA[S]) @ w_b.T
        Tb_Sl = (dAb[S] @ M.T @ p_b.T) @ w_b.T
        back = lambda z: z @ p_b.T  # noqa: E731
        acc = lambda f, p, y, a=1.0: _accuracy(f + a * p, w_b, b_b, y, mask_class=mask)  # noqa: E731
        row = {"task": task, "seed": seed, "n_support": n, "oracle": acc(f_te, proj(dA_te), y_te)}

        for prep in args.preps:
            icpt = prep != "none"
            ks = [Kern(*preprocess(X_tr[b], X_te[b], S, V, prep), icpt) for b in range(L)]
            glob = Kern(*preprocess(f_b, f_te, S, V, prep), icpt)
            joint = Kern.from_kernels(
                sum(k.K / k.trace_mean for k in ks), sum(k.Kv / k.trace_mean for k in ks),
                sum(k.Kt / k.trace_mean for k in ks), icpt,
            )
            joint_plus = None
            if ADD_tr is not None:
                ks_add = [Kern(*preprocess(ADD_tr[g], ADD_te[g], S, V, prep), icpt) for g in range(len(ADD_tr))]
                all_k = ks + ks_add
                joint_plus = Kern.from_kernels(
                    sum(k.K / k.trace_mean for k in all_k), sum(k.Kv / k.trace_mean for k in all_k),
                    sum(k.Kt / k.trace_mean for k in all_k), icpt,
                )
                del ks_add, all_k

            def per_block(beta, carry):
                pv = torch.zeros(V.numel(), T_Sl.shape[1], dtype=torch.float64)
                pt = torch.zeros(f_te.shape[0], T_Sl.shape[1], dtype=torch.float64)
                state = torch.zeros_like(T_Sl)
                for b, k in enumerate(ks):
                    fitted = Tb_Sl[:, b] + (state if carry else 0)
                    a, m = k.fit(fitted, beta * k.trace_mean)
                    if carry:
                        state = fitted - (k.K @ a + m)
                    pv, pt = pv + k.Kv @ a + m, pt + k.Kt @ a + m
                return back(pv), back(pt)

            def total(k, lam):
                a, m = k.fit(T_Sl, lam)
                return back(k.Kv @ a + m), back(k.Kt @ a + m)

            variants = {
                "br_trace": {bt: per_block(bt, False) for bt in BETAS},
                "br_trace_carry": {bt: per_block(bt, True) for bt in BETAS},
                "joint": {bt: total(joint, bt) for bt in BETAS},
                **({"joint_plus": {bt: total(joint_plus, bt) for bt in BETAS}} if joint_plus is not None else {}),
                # global: lambda relative to f_B's own mean eigenvalue, the same scale-free grid
                "global": {bt: total(glob, bt * glob.trace_mean) for bt in BETAS},
            }
            out = {}
            for name, by_pen in variants.items():
                pen = max(by_pen, key=lambda p: r2(yV, by_pen[p][0] @ head.T))
                pv, pt = by_pen[pen]
                a_best = max(ALPHAS, key=lambda a: (acc(f_b[V], pv, y_tr[V], a), -a))
                out[name] = {
                    "penalty": pen, "val_r2": r2(yV, pv @ head.T), "test_r2": r2(yT, pt @ head.T),
                    "test_acc_a1": acc(f_te, pt, y_te), "val_alpha": a_best,
                    "test_acc_valalpha": acc(f_te, pt, y_te, a_best),
                }
            row[prep] = out
            del ks, glob, joint, joint_plus, variants
            print(json.dumps({"task": task, "seed": seed, "n": n, "prep": prep,
                              **{k: (round(v["test_acc_a1"], 4), round(v["test_r2"], 3), v["penalty"]) for k, v in out.items()}}),
                  flush=True)
        rows.append(row)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1))
