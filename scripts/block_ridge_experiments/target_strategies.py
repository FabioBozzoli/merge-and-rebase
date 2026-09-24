"""Why do block_ridge's reuse_logitmap and last_only targets score the same?

Refits every block_ridge target strategy offline -- production Stage 1, grouping,
_fit_block_ridge and _predict_block_ridge, on the cached features -- from the inputs
dump_prepare_inputs.py saved, and compares the *predictions*, not just accuracies:

- argmax agreement, cosine and norm ratio of the reuse vs last_only test corrections;
- whether the correction dominates B's own logits (correction-only accuracy);
- accuracy along alpha for each strategy (does one curve equal the other rescaled?);
- the per-block hat matrices H_b (test x support map of an interpolating ridge):
  reuse = sum_b H_b T_b and last_only = (sum_b H_b) T, so near-identical H_b make
  last_only ~= L * reuse.

Usage: python target_strategies.py <dump-dir> [--seeds 33 54 89] [--few-shot 200]
"""
import argparse
import json
from pathlib import Path

import torch

from merge_and_rebase.rebase.methods.steer import (
    _BLOCK_GROUP_STRATEGIES,
    _cache_split_dir,
    _few_shot,
    _fit_block_ridge,
    _load_cached_split,
    _predict_block_ridge,
    _stage1_projection,
)
from merge_and_rebase.rebase.text.steer_text import _accuracy

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--seeds", type=int, nargs="+", default=[33, 54, 89])
parser.add_argument("--few-shot", type=int, default=200)
parser.add_argument("--out", default=None)
args = parser.parse_args()

ALPHAS = [1 / 64, 1 / 32, 1 / 16, 1 / 13, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 13, 16, 64]


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def grouped(features_b_blocks, n_src_res):
    feats = {int(b): v.double() for b, v in features_b_blocks.items()}
    n_tgt_res = len(feats) - 1
    res = {b: feats[b] for b in range(n_tgt_res)}
    if n_tgt_res != n_src_res:
        res = _BLOCK_GROUP_STRATEGIES["concat"](res, n_src_res)
    out = dict(res)
    out[n_src_res] = feats[n_tgt_res]
    return out


rows = []
for bundle_path in sorted(Path(args.dump_dir).glob("*/*_prepare_inputs.pt")):
    bd = torch.load(bundle_path, weights_only=False)
    task, c = bd["task"], bd["cache"]
    w_a, w_b, b_b, mask = bd["w_a"], bd["w_b"], bd["b_b"], bd["mask_class"]
    head = w_b[sorted(mask)] if mask is not None else w_b
    splits = {}
    for split in ("train", "test"):
        d = _cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], split)
        splits[split] = _load_cached_split(d, need_blocks=True)
    tr, te = splits["train"], splits["test"]
    f_a, dA, f_b = tr["features_A"].double(), tr["delta_A"].double(), tr["features_B"].double()
    dAb = tr["delta_A_blocks"].double()
    f_b_te, y_te = te["features_B"].double(), te["y_A"].long()
    L = dAb.shape[1]
    Xtr, Xte = grouped(tr["features_B_blocks"], L - 1), grouped(te["features_B_blocks"], L - 1)
    p_b = torch.linalg.pinv(w_b)
    lam1 = float(bd["method_params"].get("stage1_lambda", 1.0))
    ridge_lam = float(bd["method_params"].get("ridge_lambda", 1.0))

    for seed in args.seeds:
        sel = _few_shot(bd["local_labels_train"], args.few_shot, seed)
        M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=sel, regularization=lam1)
        T = dA[sel] @ M.T @ p_b.T
        targets = {
            "reuse_logitmap": dAb[sel] @ M.T @ p_b.T,
            "last_only": T.unsqueeze(1).expand(-1, L, -1),
        }
        loc = torch.arange(sel.numel())
        Xs = {b: v[sel] for b, v in Xtr.items()}
        preds, block_preds = {}, {}
        for name, tgt in targets.items():
            coefs = _fit_block_ridge(Xs, tgt, selected=loc, regularization=ridge_lam, mode="independent")
            preds[name] = _predict_block_ridge(coefs, Xte)
            block_preds[name] = [Xte[b] @ coefs[b] for b in range(L)]
            # train fit quality per block: does each ridge interpolate its support?
            block_preds[name + "_train_relres"] = [
                float((Xs[b] @ coefs[b] - tgt[:, b]).norm() / tgt[:, b].norm().clamp_min(1e-300)) for b in range(L)
            ]

        R, O = preds["reuse_logitmap"], preds["last_only"]
        lg = lambda x: x @ head.T  # noqa: E731
        acc = lambda p, a=1.0: _accuracy(f_b_te + a * p, w_b, b_b, y_te, mask_class=mask)  # noqa: E731
        pr = lambda p: (f_b_te + p) @ w_b.T + (0 if b_b is None else b_b)  # noqa: E731
        cls = torch.tensor(sorted(mask) if mask is not None else list(range(w_b.shape[0])))
        am = lambda logits: cls[logits[:, cls].argmax(1)]  # noqa: E731  full-head logits
        amh = lambda logits: cls[logits.argmax(1)]  # noqa: E731  logits already restricted to `head`

        # hat matrices: H_b = Xte_b Xs_b^T (Xs_b Xs_b^T + lam I)^-1 (exact for n <= d ridge)
        H = []
        for b in range(L):
            K = Xs[b] @ Xs[b].T
            H.append(Xte[b] @ Xs[b].T @ torch.linalg.inv(K + ridge_lam * torch.eye(K.shape[0], dtype=K.dtype)))
        Hsum = sum(H)
        Hmean = Hsum / L
        hat_spread = [float((h - Hmean).norm() / Hmean.norm()) for h in H]
        hat_scale = [float(h.norm() / Hmean.norm()) for h in H]

        row = {
            "task": task, "seed": seed, "few_shot": args.few_shot, "n_support": int(sel.numel()),
            "n_test": int(y_te.numel()), "L": L,
            "acc_stage0": acc(torch.zeros_like(R)),
            "acc_reuse": acc(R), "acc_last_only": acc(O),
            "argmax_agree_reuse_vs_last": float((am(pr(R)) == am(pr(O))).double().mean()),
            "argmax_agree_correction_only": float((amh(lg(R)) == amh(lg(O))).double().mean()),
            "acc_correction_only_reuse": float((amh(lg(R)) == y_te).double().mean()),
            "acc_correction_only_last": float((amh(lg(O)) == y_te).double().mean()),
            "cos_logits_reuse_vs_last": cos(lg(R), lg(O)),
            "norm_ratio_last_over_reuse": float(lg(O).norm() / lg(R).norm()),
            "correction_over_base_logit_norm_reuse": float(lg(R).norm() / lg(f_b_te).norm()),
            "correction_over_base_logit_norm_last": float(lg(O).norm() / lg(f_b_te).norm()),
            "acc_vs_alpha_reuse": {a: acc(R, a) for a in ALPHAS},
            "acc_vs_alpha_last": {a: acc(O, a) for a in ALPHAS},
            "block_logit_norm_share_reuse": [round(float(lg(p).norm() / sum(lg(q).norm() for q in block_preds["reuse_logitmap"])), 3) for p in block_preds["reuse_logitmap"]],
            "block_logit_norm_share_last": [round(float(lg(p).norm() / sum(lg(q).norm() for q in block_preds["last_only"])), 3) for p in block_preds["last_only"]],
            "train_relres_reuse": [round(x, 4) for x in block_preds["reuse_logitmap_train_relres"]],
            "train_relres_last": [round(x, 4) for x in block_preds["last_only_train_relres"]],
            "hat_rel_dev_from_mean": [round(x, 3) for x in hat_spread],
            "hat_norm_over_mean": [round(x, 3) for x in hat_scale],
            # reuse == sum_b H_b T_b ; if H_b ~ H then reuse ~ Hmean T and last ~ L * Hmean T
            "reuse_vs_Hmean_T_relerr": float((lg(R) - lg(Hmean @ T)).norm() / lg(R).norm()),
            "last_vs_Hsum_T_relerr": float((lg(O) - lg(Hsum @ T)).norm() / lg(O).norm()),
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith(("acc_vs_alpha", "block_", "train_relres", "hat_"))}), flush=True)
        print("   alpha  reuse  last_only")
        for a in ALPHAS:
            print(f"   {a:7.4f} {row['acc_vs_alpha_reuse'][a]:.4f} {row['acc_vs_alpha_last'][a]:.4f}")
        for k in ("block_logit_norm_share_reuse", "block_logit_norm_share_last", "train_relres_reuse",
                  "train_relres_last", "hat_rel_dev_from_mean", "hat_norm_over_mean"):
            print(f"   {k}: {row[k]}")
        print(flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1))

if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))
