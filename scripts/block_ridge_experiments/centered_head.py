"""Plain vs centered nearest-mean B head, through Stage 1 and Stage 2, offline.

B's pooled features in the steer_text cache do not depend on B's head (they are
captured with the head swapped for Identity), so a different head only changes
w_B and b_B. This refits the whole pipeline -- production Stage 1, global_ridge,
block_ridge (reuse_logitmap targets) -- from the cache for each head variant and
reports, per task and seed:

- B alone: test accuracy, median |margin|, share of logit energy common to all classes;
- the Stage-1 oracle at alpha=1;
- global_ridge and block_ridge at alpha=1, and at the alpha picked on the
  unselected train rows (never on test), over a log grid.

Stage 1 fits the residual A_logits - f_B W_B^T without B's bias (steer_text's design),
so at alpha=1 the corrected logits are A/2 + B/2 + b_B/2: half of B's bias is never
corrected. The plain head has b_B = 0; the centered one does not, so the
"centered_biasfix" variant adds the missing -alpha * b_B/(1+stage1_lambda) at eval.

The plain head comes from dump_prepare_inputs.py's bundle (the live model's head);
the centered one from build_nearest_mean_head.py --center.

Usage: python centered_head.py <dump-dir> <centered-head-dir> [--seeds ...] [--few-shot N] [--out f.json]
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
    _ridge,
    _stage1_projection,
)
from merge_and_rebase.rebase.text.steer_text import _accuracy

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("head_dir")
parser.add_argument("--seeds", type=int, nargs="+", default=[33, 54, 89])
parser.add_argument("--few-shot", type=int, default=200)
parser.add_argument("--out", default=None)
args = parser.parse_args()

ALPHAS = [0.0] + [2.0**k for k in range(-10, 3)]  # 1/1024 .. 4


def grouped(features_b_blocks, n_src_res, rows=None):
    feats = {int(b): (v if rows is None else v[rows]).double() for b, v in features_b_blocks.items()}
    n_tgt_res = len(feats) - 1
    res = {b: feats[b] for b in range(n_tgt_res)}
    if n_tgt_res != n_src_res:
        res = _BLOCK_GROUP_STRATEGIES["concat"](res, n_src_res)
    out = dict(res)
    out[n_src_res] = feats[n_tgt_res]
    return out


def head_stats(f, w_b, b_b, labels, cls):
    z = f @ w_b[cls].T + (0 if b_b is None else b_b[cls])
    yl = (labels.unsqueeze(1) == cls).float().argmax(1)
    true = z.gather(1, yl[:, None]).squeeze(1)
    other = z.clone()
    other.scatter_(1, yl[:, None], -1e30)
    margin = true - other.max(1).values
    centered = z - z.mean(1, keepdim=True)
    return float(margin.abs().median()), float(1 - centered.norm() ** 2 / z.norm() ** 2)


rows_out = []
for bundle_path in sorted(Path(args.dump_dir).glob("*/*_prepare_inputs.pt")):
    bd = torch.load(bundle_path, weights_only=False)
    task, c, mask = bd["task"], bd["cache"], bd["mask_class"]
    cls = torch.tensor(sorted(mask))
    head_file = Path(args.head_dir) / f"t5-large_{task}_nearest_mean_centered_enc_seed33_fewshot300.pt"
    payload = torch.load(head_file, weights_only=False)[task]
    w_c = payload["classification_head.out_proj.weight"].double()
    b_c = payload["classification_head.out_proj.bias"].double()
    heads = {"plain": (bd["w_b"], bd["b_b"], False), "centered": (w_c, b_c, False), "centered_biasfix": (w_c, b_c, True)}

    tr, te = (
        _load_cached_split(
            _cache_split_dir(c["feature_cache_dir"], c["source_tag"], c["target_tag"], task, c["feature_regime"], s),
            need_blocks=True,
        )
        for s in ("train", "test")
    )
    f_a, dA, f_b, dAb = (tr[k].double() for k in ("features_A", "delta_A", "features_B", "delta_A_blocks"))
    y_tr = tr["y_A"].long()
    f_te, dA_te, y_te = te["features_B"].double(), te["delta_A"].double(), te["y_A"].long()
    L = dAb.shape[1]
    Xtr_all = grouped(tr["features_B_blocks"], L - 1)
    Xte = grouped(te["features_B_blocks"], L - 1)
    w_a = bd["w_a"]

    for seed in args.seeds:
        sel = _few_shot(bd["local_labels_train"], args.few_shot, seed)
        unsel = torch.ones(f_b.shape[0], dtype=torch.bool)
        unsel[sel] = False
        unsel = unsel.nonzero().squeeze(1)
        Xs = {b: v[sel] for b, v in Xtr_all.items()}
        Xv = {b: v[unsel] for b, v in Xtr_all.items()}

        for head_name, (w_b, b_b, bias_fix) in heads.items():
            M = _stage1_projection(f_a=f_a, delta_a=dA, w_a=w_a, f_b=f_b, w_b=w_b, selected=sel, regularization=1.0)
            p_b = torch.linalg.pinv(w_b)
            # feature-space vector whose logits are -b_B/(1+lambda) on the task's rows
            shift = -(p_b @ b_b) / 2.0 if bias_fix else torch.zeros(w_b.shape[1], dtype=torch.float64)
            acc = lambda f, p, y, a, _s=shift: _accuracy(f + a * (p + _s), w_b, b_b, y, mask_class=mask)  # noqa: E731
            T = dA[sel] @ M.T @ p_b.T
            g = _ridge(f_b[sel], T, 1.0)
            coefs = _fit_block_ridge(
                Xs, dAb[sel] @ M.T @ p_b.T, selected=torch.arange(sel.numel()), regularization=1.0, mode="independent"
            )
            preds = {
                "global_ridge": (f_b[unsel] @ g, f_te @ g),
                "block_ridge": (_predict_block_ridge(coefs, Xv), _predict_block_ridge(coefs, Xte)),
            }
            margin, shared = head_stats(f_te, w_b, b_b, y_te, cls)
            row = {
                "task": task, "seed": seed, "head": head_name,
                "stage0": _accuracy(f_te, w_b, b_b, y_te, mask_class=mask),
                "median_abs_margin": margin, "shared_logit_share": shared,
                "oracle": acc(f_te, dA_te @ M.T @ p_b.T, y_te, 1.0),
            }
            for strat, (pv, pt) in preds.items():
                val = {a: acc(f_b[unsel], pv, y_tr[unsel], a) for a in ALPHAS}
                best = max(ALPHAS, key=lambda a: (val[a], -a))  # ties -> smaller alpha
                row[f"{strat}_a1"] = acc(f_te, pt, y_te, 1.0)
                row[f"{strat}_valalpha"] = best
                row[f"{strat}_at_valalpha"] = acc(f_te, pt, y_te, best)
                row[f"{strat}_test_curve"] = {a: acc(f_te, pt, y_te, a) for a in ALPHAS}
            rows_out.append(row)
            print(
                f"{task:8s} {seed:3d} {head_name:9s} stage0={row['stage0']:.4f} |m|={margin:.4g} shared={shared:.4f} "
                f"oracle={row['oracle']:.4f} | global a1={row['global_ridge_a1']:.4f} "
                f"val-a={row['global_ridge_valalpha']:.4g}->{row['global_ridge_at_valalpha']:.4f} | "
                f"block a1={row['block_ridge_a1']:.4f} val-a={row['block_ridge_valalpha']:.4g}->{row['block_ridge_at_valalpha']:.4f}",
                flush=True,
            )
            if args.out:
                Path(args.out).write_text(json.dumps(rows_out, indent=1))
