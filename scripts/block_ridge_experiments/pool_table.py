"""Tables for the normalize-then-pool refit (pool_curve/); 'mean' rows come from preprocess_curve/.

Usage: python pool_table.py   (reads $BR_LABELFREE_ROOT)
"""
import os
import json, glob, os, statistics as st
from collections import defaultdict

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")

POOLS = ["mean", "unitnorm", "rmsnorm", "center_unitnorm", "center_rmsnorm"]
VARS = ["br_trace", "br_trace_carry", "joint", "global"]
PREPS = ["none", "zscore"]

rows = defaultdict(list)  # pooling -> rows
for f in glob.glob(f"{W}/preprocess_curve/*.json"):
    rows["mean"] += json.load(open(f))
for f in glob.glob(f"{W}/pool_curve/*.json"):
    task_pool = os.path.basename(f)[:-5]
    pool = task_pool.split("_", 1)[1]
    rows[pool] += json.load(open(f))

def mean_over(rs, prep, var, metric):
    return st.mean(r[prep][var][metric] for r in rs)

for metric in ("test_acc_a1", "test_acc_valalpha", "test_r2"):
    print(f"\n=== {metric}: mean over 5 tasks (excl. rte) x 3 seeds ===")
    print(f"{'n':>5s} {'variant':15s} {'prep':6s} " + " ".join(f"{p:>15s}" for p in POOLS))
    for n in (600, 1500, 3400):
        for var in VARS:
            for prep in PREPS:
                cells = []
                for p in POOLS:
                    rs = [r for r in rows[p] if r["n_support"] == n and r["task"] != "rte"]
                    cells.append(f"{mean_over(rs, prep, var, metric):15.4f}" if rs and len({r['task'] for r in rs}) == 5 else f"{'-':>15s}")
                print(f"{n:5d} {var:15s} {prep:6s} " + " ".join(cells))

print("\n=== test_acc_a1 per task at the largest n, prep=none (mean over seeds) ===")
for t in ("mnli", "qnli", "rte", "scitail", "sick", "snli"):
    print(t)
    for var in VARS:
        cells = []
        for p in POOLS:
            rs = [r for r in rows[p] if r["task"] == t]
            if not rs:
                cells.append(f"{p}=-"); continue
            nmax = max(r["n_support"] for r in rs)
            cells.append(f"{p}={st.mean(r['none'][var]['test_acc_a1'] for r in rs if r['n_support'] == nmax):.4f}")
        print(f"   {var:15s} " + " ".join(cells))
