"""Sweep steer_text's alpha (f_B + alpha * correction) for every stage-2 predictor we compared.

Cached-feature space, i.e. the stage2_test_acc diagnostic with alpha != 1.
Tuned predictors use the base their earlier run picked on the unselected
train rows (trace_lambda/ and joint_ridge/ reports); nothing is picked on test.
alpha = 0 must reproduce stage0 (B alone) exactly -- asserted; so must
alpha = 1 of the saved fit reproduce the recorded stage2_test_acc, when 1 is swept.

Usage: python scripts/block_ridge_experiments/alpha_sweep.py <run-dir> [--alphas 0,0.2,...] [--out-subdir alpha_sweep]
"""
import argparse, json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_block_ridge import _load_split, grouped_blocks  # noqa: E402
from merge_and_rebase.rebase.methods.steer import _fit_block_ridge, _predict_block_ridge, _ridge  # noqa: E402
from merge_and_rebase.rebase.text.steer_text import _accuracy  # noqa: E402

ROOT = Path(os.environ.get("DIAG_ROOT", "/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag"))
_p = argparse.ArgumentParser()
_p.add_argument("run_dir", type=Path)
_p.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1")
_p.add_argument("--out-subdir", default="alpha_sweep")
_args = _p.parse_args()
ALPHAS = [float(a) for a in _args.alphas.split(",")]
assert 0.0 in ALPHAS, "alpha=0 is the stage0 sanity check; keep it in the sweep"

d = _args.run_dir
cfg = json.loads((d / "config.json").read_text())
task, seed, fs = cfg["tasks"], cfg["seed"], cfg["method_params"]["few_shot"]
lam = float(cfg["method_params"]["ridge_lambda"])
art = torch.load(d / f"{task}_steer_artifacts.pt", map_location="cpu", weights_only=False)
state = art["stage2_state"]
lm, pb, w_b = art["stage1_logit_map"].double(), art["stage1_pinv_w_b"].double(), art["w_b"].double()
b_b = art["b_b"].double() if art["b_b"] is not None else None
sel = art["selected"].long()
mask = sorted(set(int(c) for c in art["head_class_ids"]))
train, test = _load_split(cfg, task, "train"), _load_split(cfg, task, "test")


def picked(sub, prefix):
    r = json.loads((ROOT / "_reports" / sub / f"{task}_fs{fs}_seed{seed}.json").read_text())
    keys = [k for k in r if k.startswith(prefix) and isinstance(r[k], dict) and "val" in r[k]]
    return float(max(keys, key=lambda k: r[k]["val"][0])[len(prefix):])


f_b_test = test["features_B"].double()
labels = test["y_A"].long()
y_support_total = train["delta_A"][sel].double() @ lm.T @ pb.T
y_support_blocks = train["delta_A_blocks"][sel].double() @ lm.T @ pb.T
sup_blocks = grouped_blocks(train["features_B_blocks"], state, sel)
test_blocks = grouped_blocks(test["features_B_blocks"], state, None)
nb = len(sup_blocks)

preds, bases = {}, {}
# 1. block_ridge exactly as the grid ran it (saved coefficients).
preds["block_ridge_saved"] = _predict_block_ridge([c.double() for c in state["coefficients"]], test_blocks)
# 2. per-block trace-scaled lambda, val-picked base.
bases["per_block_trace"] = picked("trace_lambda", "trace")
coefs = _fit_block_ridge(sup_blocks, y_support_blocks, selected=torch.arange(sel.numel()),
                         regularization=bases["per_block_trace"], mode=state["block_ridge_mode"],
                         rho=state["rho"], regularization_scaling="trace")
preds["per_block_trace"] = _predict_block_ridge(coefs, test_blocks)
# 3. joint ridge on trace-normalized concatenated blocks, val-picked base.
bases["joint"] = picked("joint_ridge", "joint")
scale = [(sup_blocks[b].square().sum() / sup_blocks[b].shape[0]).sqrt() for b in range(nb)]
z_sup = torch.cat([sup_blocks[b] / scale[b] for b in range(nb)], dim=1)
z_test = torch.cat([test_blocks[b] / scale[b] for b in range(nb)], dim=1)
preds["joint"] = z_test @ _ridge(z_sup, y_support_total, bases["joint"])
del z_sup, z_test
# 4. global_ridge, normalized + val-picked base; 5. global_ridge as the grid ran it (lambda=1).
bases["global_tuned"] = picked("joint_ridge", "global")
f_sup = train["features_B"][sel].double()
gs = (f_sup.square().sum() / sel.numel()).sqrt()
preds["global_tuned"] = (f_b_test / gs) @ _ridge(f_sup / gs, y_support_total, bases["global_tuned"])
preds["global_grid"] = f_b_test @ _ridge(f_sup, y_support_total, lam)

stage0 = _accuracy(f_b_test, w_b, b_b, labels, mask_class=mask)
oracle = test["delta_A"].double() @ lm.T @ pb.T
row = {"task": task, "seed": seed, "few_shot": fs, "bases": bases, "stage0": stage0,
       "stage1_oracle": {f"{a:g}": _accuracy(f_b_test + a * oracle, w_b, b_b, labels, mask_class=mask) for a in ALPHAS}}
for name, p in preds.items():
    row[name] = {f"{a:g}": _accuracy(f_b_test + a * p, w_b, b_b, labels, mask_class=mask) for a in ALPHAS}
    assert row[name]["0"] == stage0, f"{name}: alpha=0 gives {row[name]['0']} != stage0 {stage0}"
# alpha=1 of the saved fit must be the recorded stage2_test_acc.
if "1" in row["block_ridge_saved"]:
    assert abs(row["block_ridge_saved"]["1"] - art["diagnostics"]["stage2_test_acc"]) <= 1 / labels.numel() + 1e-12

out = ROOT / "_reports" / _args.out_subdir
out.mkdir(parents=True, exist_ok=True)
(out / f"{task}_fs{fs}_seed{seed}.json").write_text(json.dumps(row, indent=1))
print(task, fs, seed, {k: v for k, v in row.items() if isinstance(v, dict)})
