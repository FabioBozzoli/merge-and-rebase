"""Refit block_ridge with trace-scaled lambda; compare to the saved fit and global_ridge.

For each run directory (a block_ridge run with <task>_steer_artifacts.pt), refit
the per-block ridges with lambda_b = base * tr(X_b X_bᵀ)/n for every base in
BASES, using the production _fit_block_ridge on the run's own support, Stage-1
map and cached features. First asserts that the unscaled refit reproduces the
saved coefficients and the recorded stage2_test_acc. Scores accuracy and
logit-space R² on the unselected train rows ("val", for picking a base) and on
test. See docs/block-ridge-regression-diagnostics.md.

Usage: python scripts/block_ridge_experiments/trace_lambda.py <run-dir> [<run-dir> ...]
"""
import json, os, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_block_ridge import _load_split, grouped_blocks, r2_scores  # noqa: E402
from merge_and_rebase.rebase.methods.steer import _fit_block_ridge, _predict_block_ridge, _ridge  # noqa: E402
from merge_and_rebase.rebase.text.steer_text import _accuracy  # noqa: E402

ROOT = os.environ.get("DIAG_ROOT", "/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag")
BASES = [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
rows = []
for d in sys.argv[1:]:
    d = Path(d)
    cfg = json.loads((d / "config.json").read_text())
    task = cfg["tasks"]
    art = torch.load(d / f"{task}_steer_artifacts.pt", map_location="cpu", weights_only=False)
    state = art["stage2_state"]
    lm, pb, w_b = art["stage1_logit_map"].double(), art["stage1_pinv_w_b"].double(), art["w_b"].double()
    b_b = art["b_b"].double() if art["b_b"] is not None else None
    sel = art["selected"].long()
    mask = sorted(set(int(c) for c in art["head_class_ids"]))
    head = w_b[mask]
    lam = float(cfg["method_params"]["ridge_lambda"])
    train, test = _load_split(cfg, task, "train"), _load_split(cfg, task, "test")
    n = train["features_B"].shape[0]
    unsel = torch.ones(n, dtype=torch.bool); unsel[sel] = False; unsel = unsel.nonzero().squeeze(1)

    tgt = train["delta_A_blocks"][sel].double() @ lm.T @ pb.T
    X = grouped_blocks(train["features_B_blocks"], state, sel)
    local = torch.arange(sel.numel())

    def fit(scaling, base):
        return _fit_block_ridge(X, tgt, selected=local, regularization=base, mode=state["block_ridge_mode"],
                                rho=state["rho"], regularization_scaling=scaling)

    # Guard: the unscaled refit must reproduce the saved production coefficients.
    ref = fit("none", lam)
    gap = max(float((a - b.double()).norm() / b.double().norm()) for a, b in zip(ref, state["coefficients"]))
    assert gap < 1e-6, f"{d.name}: unscaled refit differs from saved coefs ({gap:.2e})"

    evals = {}
    for split, data, r in (("val", train, unsel), ("test", test, None)):
        f_b = (data["features_B"][r] if r is not None else data["features_B"]).double()
        y = (data["delta_A"][r] if r is not None else data["delta_A"]).double() @ lm.T @ pb.T
        labels = (data["y_A"][r] if r is not None else data["y_A"]).long()
        evals[split] = (grouped_blocks(data["features_B_blocks"], state, r), f_b, y, labels)

    def score(pred_fn):
        out = {}
        for split, (blocks, f_b, y, labels) in evals.items():
            p = pred_fn(blocks, f_b)
            out[split] = (_accuracy(f_b + p, w_b, b_b, labels, mask_class=mask),
                          r2_scores(y @ head.T, p @ head.T)["variance_weighted"])
        return out

    g_coef = _ridge(train["features_B"][sel].double(), train["delta_A"][sel].double() @ lm.T @ pb.T, lam)
    row = {"task": task, "seed": cfg["seed"], "few_shot": cfg["method_params"]["few_shot"],
           "saved": score(lambda B, f: _predict_block_ridge(ref, B)),
           "global": score(lambda B, f: f @ g_coef)}
    for base in BASES:
        coefs = fit("trace", base)
        row[f"trace{base:g}"] = score(lambda B, f, c=coefs: _predict_block_ridge(c, B))
    assert abs(row["saved"]["test"][0] - art["diagnostics"]["stage2_test_acc"]) <= 1 / len(evals["test"][3]) + 1e-12
    rows.append(row)
    print(d.name, {k: v["test"] for k, v in row.items() if isinstance(v, dict)}, flush=True)

Path(f"{ROOT}/_reports/trace_lambda").mkdir(parents=True, exist_ok=True)
for r in rows: json.dump(r, open(f"{ROOT}/_reports/trace_lambda/{r["task"]}_fs{r["few_shot"]}_seed{r["seed"]}.json", "w"), indent=1)
