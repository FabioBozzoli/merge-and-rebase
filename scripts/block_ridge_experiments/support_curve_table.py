"""Tables for support_curve/ (label-free learning curve of the block_ridge variants).

Usage: python support_curve_table.py   (reads $BR_LABELFREE_ROOT)
"""
import os
import json, glob, statistics as st
from collections import defaultdict

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")

V = ["br_asrun", "br_trace", "br_trace_carry", "br_sumavg_trace", "joint", "global"]
rows = [r for f in sorted(glob.glob(f"{W}/support_curve/*.json")) for r in json.load(open(f))]
by = defaultdict(list)
for r in rows:
    by[(r["task"], r["n_support"])].append(r)

for metric, label in (("test_acc_a1", "test acc, alpha=1"), ("test_acc_valalpha", "test acc, alpha picked on V"), ("test_r2", "test logit R^2")):
    print(f"\n=== {label} (mean over seeds) ===")
    print(f"{'task':8s} {'n':>5s} {'oracle':>7s} " + " ".join(f"{v:>15s}" for v in V))
    for (t, n), rs in sorted(by.items()):
        print(f"{t:8s} {n:5d} {st.mean(r['oracle'] for r in rs):7.4f} " + " ".join(f"{st.mean(r[v][metric] for r in rs):15.4f}" for v in V))

print("\n=== mean over tasks with 3400 rows (excl. rte), by n ===")
for metric in ("test_acc_a1", "test_acc_valalpha", "test_r2"):
    print(metric)
    for n in sorted({r["n_support"] for r in rows if r["task"] != "rte"}):
        rs = [r for r in rows if r["n_support"] == n and r["task"] != "rte"]
        print(f"  n={n:5d} oracle={st.mean(r['oracle'] for r in rs):.4f} " + " ".join(f"{v}={st.mean(r[v][metric] for r in rs):.4f}" for v in V))

print("\n=== picked penalties (beta) per variant, all runs ===")
for v in V[1:]:
    c = defaultdict(int)
    for r in rows:
        c[r[v]["penalty"]] += 1
    print(f"  {v:16s} {dict(sorted(c.items(), key=lambda kv: -kv[0]))}")
