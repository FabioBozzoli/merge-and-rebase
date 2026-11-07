"""Tables for the SelfAttention.o experiment (attn_curve/), against the residual-stream results.

Rows come from three runs of preprocess_curve.py that share seeds, held-out rows and supports:
  preprocess_curve/<task>.json            residual stream, mean pooling (the cache)
  pool_curve/<task>_unitnorm.json         residual stream, unit-norm pooling
  attn_curve/<task>_replace_<p>.json      attention outputs replace the residual blocks
  attn_curve/<task>_add_<p>.json          residual unitnorm blocks + attention outputs (joint_plus)

Usage: python attn_table.py   (reads $BR_LABELFREE_ROOT)
"""
import os
import glob
import json
import statistics as st
from collections import defaultdict

W = os.environ.get("BR_LABELFREE_ROOT", "/work/tesi_pmoriello/claude_debug_block_targets")
TASKS5 = ("mnli", "qnli", "snli", "scitail", "sick")


def load(pattern, exclude=None):
    out = defaultdict(list)
    for f in glob.glob(f"{W}/{pattern}"):
        if exclude and exclude in os.path.basename(f):
            continue
        for r in json.load(open(f)):
            out[r["task"]].append(r)
    return out


src = {
    "resid-mean": load("preprocess_curve/*.json"),
    # the glob would also match *_center_unitnorm.json
    "resid-unitnorm": load("pool_curve/*_unitnorm.json", exclude="center_"),
    **{f"attn-{p}": load(f"attn_curve/*_replace_{p}.json") for p in ("mean", "unitnorm", "headnorm")},
    **{f"resid-unitnorm+attn-{p}": load(f"attn_curve/*_add_{p}.json") for p in ("unitnorm", "headnorm")},
}
# (column label, source, variant)
COLS = [
    ("global", "resid-mean", "global"),
    ("resid-mean joint", "resid-mean", "joint"),
    ("resid-unit joint", "resid-unitnorm", "joint"),
    ("resid-unit carry", "resid-unitnorm", "br_trace_carry"),
    ("attn-mean joint", "attn-mean", "joint"),
    ("attn-unit joint", "attn-unitnorm", "joint"),
    ("attn-head joint", "attn-headnorm", "joint"),
    ("attn-unit carry", "attn-unitnorm", "br_trace_carry"),
    ("attn-head carry", "attn-headnorm", "br_trace_carry"),
    ("+attn-unit", "resid-unitnorm+attn-unitnorm", "joint_plus"),
    ("+attn-head", "resid-unitnorm+attn-headnorm", "joint_plus"),
]


def cell(source, var, prep, metric, tasks, n, *, seed=None):
    rs = [r for t in tasks for r in src[source].get(t, [])
          if r["n_support"] == n and prep in r and var in r[prep] and (seed is None or r["seed"] == seed)]
    return st.mean(r[prep][var][metric] for r in rs) if len({r["task"] for r in rs}) == len(tasks) else None


def fmt(x):
    return f"{x:.4f}" if x is not None else "  -   "


for prep in ("none", "zscore"):
    for metric in ("test_acc_a1", "test_acc_valalpha", "test_r2"):
        print(f"\n=== {metric}, prep={prep}: mean over {len(TASKS5)} tasks (excl. rte) x 3 seeds ===")
        print(f"{'n':>5s} " + " ".join(f"{c[0]:>17s}" for c in COLS))
        for n in (600, 1500, 3400):
            print(f"{n:5d} " + " ".join(f"{fmt(cell(s, v, prep, metric, TASKS5, n)):>17s}" for _, s, v in COLS))

# consistency: the add-mode runs fit "joint" on exactly the residual unit-norm features
gap = max(abs(r[p]["joint"]["test_acc_a1"] - q[p]["joint"]["test_acc_a1"])
          for t in src["resid-unitnorm"] for r in src["resid-unitnorm"][t]
          for q in src["resid-unitnorm+attn-unitnorm"].get(t, [])
          if (q["seed"], q["n_support"]) == (r["seed"], r["n_support"]) for p in ("none", "zscore"))
print(f"\n[check] add-mode 'joint' vs pool_curve unitnorm 'joint': max |diff| = {gap:.2e}")

print("\n=== per task, n = largest, prep=none, test_acc_a1 (mean over seeds) ===")
print(f"{'task':8s} " + " ".join(f"{c[0]:>17s}" for c in COLS))
for t in TASKS5 + ("rte",):
    n = 1890 if t == "rte" else 3400
    print(f"{t:8s} " + " ".join(f"{fmt(cell(s, v, 'none', 'test_acc_a1', (t,), n)):>17s}" for _, s, v in COLS))

print("\n=== paired differences at the largest n, prep=none (15 task-seeds, excl. rte), points ===")
pairs = [("+attn-unit", "resid-unit joint"), ("+attn-head", "resid-unit joint"),
         ("attn-unit joint", "resid-unit joint"), ("attn-head joint", "resid-unit joint"),
         ("+attn-unit", "global"), ("+attn-head", "global")]
lookup = {c[0]: c for c in COLS}
for a, b in pairs:
    for metric in ("test_acc_a1", "test_acc_valalpha"):
        d = []
        for t in TASKS5:
            for seed in (33, 54, 89):
                x = cell(lookup[a][1], lookup[a][2], "none", metric, (t,), 3400, seed=seed)
                y = cell(lookup[b][1], lookup[b][2], "none", metric, (t,), 3400, seed=seed)
                if x is not None and y is not None:
                    d.append(100 * (x - y))
        if len(d) > 1:
            print(f"  {a:>16s} - {b:<16s} {metric:17s} mean={st.mean(d):+.2f} se={st.stdev(d) / len(d) ** .5:.2f} "
                  f"wins={sum(x > 0 for x in d)}/{len(d)}")
