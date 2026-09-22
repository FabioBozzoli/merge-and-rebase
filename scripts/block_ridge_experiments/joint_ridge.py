"""One ridge on all 13 trace-normalized blocks, concatenated, fit to the *total* correction.

Dividing block b by sqrt(tr(X_b X_bᵀ)/n) (support rows only) and fitting one
ridge with penalty `base` is the same as penalizing block b's slice of the
coefficient with base * tr(X_b X_bᵀ)/n -- i.e. block_ridge's trace scaling,
but one joint fit to the summed target instead of 13 fits to per-block slices.
global_ridge gets the same treatment (its single block normalized, same bases)
so both sides are tuned identically. Bases are picked on the unselected train
rows; test is only reported.

Usage: python scripts/block_ridge_experiments/joint_ridge.py <run-dir>
"""
import json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_block_ridge import _load_split, grouped_blocks, r2_scores  # noqa: E402
from merge_and_rebase.rebase.methods.steer import _ridge  # noqa: E402
from merge_and_rebase.rebase.text.steer_text import _accuracy  # noqa: E402

ROOT = Path(os.environ.get("DIAG_ROOT", "/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag"))
BASES = [10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]

d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
task = cfg["tasks"]
art = torch.load(d / f"{task}_steer_artifacts.pt", map_location="cpu", weights_only=False)
state = art["stage2_state"]
lm, pb, w_b = art["stage1_logit_map"].double(), art["stage1_pinv_w_b"].double(), art["w_b"].double()
b_b = art["b_b"].double() if art["b_b"] is not None else None
sel = art["selected"].long()
mask = sorted(set(int(c) for c in art["head_class_ids"]))
head = w_b[mask]
train, test = _load_split(cfg, task, "train"), _load_split(cfg, task, "test")
n = train["features_B"].shape[0]
unsel = torch.ones(n, dtype=torch.bool); unsel[sel] = False; unsel = unsel.nonzero().squeeze(1)


def total_target(data, rows):
    delta = data["delta_A"] if rows is None else data["delta_A"][rows]
    return delta.double() @ lm.T @ pb.T


# Per-block scale from the support only, applied unchanged to val/test.
support_blocks = grouped_blocks(train["features_B_blocks"], state, sel)
num_blocks = len(support_blocks)
scale = {b: (support_blocks[b].square().sum() / support_blocks[b].shape[0]).sqrt() for b in range(num_blocks)}
global_scale = (train["features_B"][sel].double().square().sum() / sel.numel()).sqrt()


def joint_features(data, rows):
    blocks = support_blocks if (data is train and rows is sel) else grouped_blocks(data["features_B_blocks"], state, rows)
    return torch.cat([blocks[b] / scale[b] for b in range(num_blocks)], dim=1)


def global_features(data, rows):
    f = data["features_B"] if rows is None else data["features_B"][rows]
    return f.double() / global_scale


y_support = total_target(train, sel)
splits = {"val": (train, unsel), "test": (test, None)}
labels = {k: (data["y_A"] if rows is None else data["y_A"][rows]).long() for k, (data, rows) in splits.items()}
f_b = {k: (data["features_B"] if rows is None else data["features_B"][rows]).double() for k, (data, rows) in splits.items()}
y = {k: total_target(data, rows) for k, (data, rows) in splits.items()}

row = {"task": task, "seed": cfg["seed"], "few_shot": cfg["method_params"]["few_shot"], "joint_dim": None}
for name, feat_fn in (("joint", joint_features), ("global", global_features)):
    z_support = feat_fn(train, sel)
    row["joint_dim" if name == "joint" else "global_dim"] = int(z_support.shape[1])
    z = {k: feat_fn(data, rows) for k, (data, rows) in splits.items()}
    for base in BASES:
        coef = _ridge(z_support, y_support, base)
        res = {}
        for k in splits:
            p = z[k] @ coef
            res[k] = (_accuracy(f_b[k] + p, w_b, b_b, labels[k], mask_class=mask),
                      r2_scores(y[k] @ head.T, p @ head.T)["variance_weighted"])
        res["support_r2"] = r2_scores(y_support @ head.T, (z_support @ coef) @ head.T)["variance_weighted"]
        row[f"{name}{base:g}"] = res
    del z, z_support

out = ROOT / "_reports" / "joint_ridge"
out.mkdir(parents=True, exist_ok=True)
(out / f"{task}_fs{row['few_shot']}_seed{cfg['seed']}.json").write_text(json.dumps(row, indent=1))
print(task, cfg["seed"], {k: v["test"] for k, v in row.items() if isinstance(v, dict)})
