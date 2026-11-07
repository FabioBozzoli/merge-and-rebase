#!/usr/bin/env python
"""Tables for docs/block-ridge-regression-diagnostics.md from the per-run JSON reports.

    python scripts/block_ridge_experiments/summarize.py [trace|joint|alpha|alpha_fine|margins|all]

Every tuned predictor's base is picked per run on the unselected train rows
("val"); test numbers are means over the seeds (and, for the alpha tables,
over the six tasks too).
"""
import glob
import os
import json
import statistics as st
import sys

REPORTS = os.path.join(os.environ.get("DIAG_ROOT", "/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag"), "_reports")
TASKS = ("mnli", "sick", "rte", "scitail", "qnli", "snli")
FEW_SHOTS = (200, 300, 500)


def _load(sub):
    return {(r["task"], r["few_shot"], r["seed"]): r for r in map(lambda f: json.load(open(f)), glob.glob(f"{REPORTS}/{sub}/*.json"))}


def _pick(r, prefix):
    keys = [k for k in r if k.startswith(prefix) and isinstance(r[k], dict) and "val" in r[k]]
    best = max(keys, key=lambda k: r[k]["val"][0])
    return r[best]["test"][0], r[best]["test"][1], best[len(prefix):]


def trace_table(fs=200):
    trace = _load("trace_lambda")
    bases = sorted({k for r in trace.values() for k in r if k.startswith("trace")}, key=lambda k: -float(k[5:]))
    print(f"| task | saved | " + " | ".join("×" + b[5:] for b in bases) + " | val-picked | global_ridge (λ=1) |")
    print("|---" * (len(bases) + 4) + "|")
    for t in TASKS:
        rs = [trace[k] for k in sorted(trace) if k[:2] == (t, fs)]
        if not rs:
            continue
        acc = lambda k: st.mean(r[k]["test"][0] for r in rs)
        vp = st.mean(_pick(r, "trace")[0] for r in rs)
        print(f"| {t} | {acc('saved'):.3f} | " + " | ".join(f"{acc(b):.3f}" for b in bases) + f" | {vp:.3f} | {acc('global'):.3f} |")


def joint_table():
    joint, trace = _load("joint_ridge"), _load("trace_lambda")
    print("| task | fs | block_ridge saved | per-block trace | joint concat | global tuned | global grid λ=1 | test logit R²: per-block / joint / global tuned | joint picks |")
    print("|---" * 9 + "|")
    for t in TASKS:
        for fs in FEW_SHOTS:
            keys = sorted(k for k in joint if k[:2] == (t, fs) and k in trace)
            if not keys:
                continue
            pb = [_pick(trace[k], "trace") for k in keys]
            jt = [_pick(joint[k], "joint") for k in keys]
            gl = [_pick(joint[k], "global") for k in keys]
            m = st.mean
            print(f"| {t} | {fs} | {m(trace[k]['saved']['test'][0] for k in keys):.3f} | {m(x[0] for x in pb):.3f} "
                  f"| {m(x[0] for x in jt):.3f} | {m(x[0] for x in gl):.3f} | {m(trace[k]['global']['test'][0] for k in keys):.3f} "
                  f"| {m(x[1] for x in pb):.2f} / {m(x[1] for x in jt):.2f} / {m(x[1] for x in gl):.2f} | {','.join(x[2] for x in jt)} |")


METHODS = ["stage1_oracle", "block_ridge_saved", "per_block_trace", "joint", "global_tuned", "global_grid"]


def alpha_table(sub):
    rows = list(_load(sub).values())
    alphas = sorted({a for r in rows for a in r["joint"]}, key=float)
    for fs in FEW_SHOTS:
        rs = [r for r in rows if r["few_shot"] == fs]
        if not rs:
            continue
        print(f"\nfs{fs} (n={len(rs)} runs), mean test accuracy over tasks × seeds")
        print("| method | " + " | ".join(f"α={a}" for a in alphas) + " | best α |")
        print("|---" * (len(alphas) + 2) + "|")
        for m in METHODS:
            v = [st.mean(r[m][a] for r in rs) for a in alphas]
            best = max(range(len(v)), key=v.__getitem__)
            print(f"| {m} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {alphas[best]} |")


def margins():
    print("| task | B stage0 margin range | oracle logit-correction norm range | Q1/Q2 boundary |")
    print("|---|---|---|---|")
    for f in sorted(glob.glob(f"{REPORTS}/*fs200_seed33.json")):
        d = json.load(open(f))
        mg, cn = d["hard_vs_easy"]["by_stage0_margin"], d["hard_vs_easy"]["by_logit_correction_norm"]
        print(f"| {d['task']} | [{mg[0]['key_min']:.3f}, {mg[-1]['key_max']:.3f}] | [{cn[0]['key_min']:.2f}, {cn[-1]['key_max']:.2f}] | {cn[0]['key_max']:.2f} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("trace", "all"):
        trace_table()
    if what in ("joint", "all"):
        print(); joint_table()
    if what in ("alpha", "all"):
        alpha_table("alpha_sweep")
    if what in ("alpha_fine", "all"):
        alpha_table("alpha_sweep_fine")
    if what in ("margins", "all"):
        print(); margins()
