"""Tables for preprocess_curve/ (pooled-feature preprocessing, mean pooling).

Usage: python preprocess_table.py   (reads $BR_LABELFREE_ROOT)
"""
import os
import json, glob, statistics as st
from collections import defaultdict

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")

P = ["none", "center", "zscore", "center_drop1", "rownorm"]
V = ["br_trace", "br_trace_carry", "joint", "global"]
rows = [r for f in sorted(glob.glob(f"{W}/preprocess_curve/*.json")) for r in json.load(open(f))]

for metric in ("test_acc_a1", "test_acc_valalpha", "test_r2"):
    print(f"\n=== {metric}: mean over 5 tasks (excl. rte) x 3 seeds ===")
    print(f"{'n':>5s} {'variant':15s} " + " ".join(f"{p:>13s}" for p in P))
    for n in sorted({r["n_support"] for r in rows if r["task"] != "rte"}):
        rs = [r for r in rows if r["n_support"] == n and r["task"] != "rte"]
        for v in V:
            print(f"{n:5d} {v:15s} " + " ".join(f"{st.mean(r[p][v][metric] for r in rs):13.4f}" for p in P))

print("\n=== test_acc_a1 per task at the largest n (mean over seeds) ===")
tasks = sorted({r["task"] for r in rows})
for t in tasks:
    nmax = max(r["n_support"] for r in rows if r["task"] == t)
    rs = [r for r in rows if r["task"] == t and r["n_support"] == nmax]
    print(f"{t:8s} n={nmax}")
    for v in V:
        print(f"   {v:15s} " + " ".join(f"{p}={st.mean(r[p][v]['test_acc_a1'] for r in rs):.4f}" for p in P))

print("\n=== picked beta per (prep, variant) ===")
for p in P:
    for v in V:
        c = defaultdict(int)
        for r in rows:
            c[r[p][v]["penalty"]] += 1
        print(f"  {p:13s} {v:15s} {dict(sorted(c.items(), key=lambda kv: -kv[0]))}")
