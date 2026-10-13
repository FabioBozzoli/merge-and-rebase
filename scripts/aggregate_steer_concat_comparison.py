"""Aggregate steer concat_ridge vs global_ridge vs block_ridge on vision8 (vitb->vitb laion2b, vitb->vitl datacomp).

Test accuracy (alpha=1 fixed), mean +- std over seeds 33/54/89, per (pair, method, dataset, K).
Sources (all under .log/):
  concat_ridge  linear / nonlinear : steer_concatridge_{linear,nonlinear}/<pair>/summaries/<T>_K<k>_seed<s>.json
  global_ridge  nonlinear          : steer_globalridge_nonlinear/<pair>/summaries/...   (SVHN excluded there: it ran
                                     without the 96px preprocessing its cache was built with, cached 0.625 vs live 0.496)
                SVHN (lin + nonlin): steer_svhn96_noaug_s1energy/global/<lin|nonlin>/vitb_<pair>/global_ridge/SVHN logs
  global_ridge  linear             : <pair>_globalridge_linear_subset (only DTD/GTSRB/RESISC45 K=10, SUN397 K=1) + SVHN above
  block_ridge   linear             : <pair>_trace (smoothed_residual rho=0.9, trace lambda=0.1). Older code: Stage 1 with the
                                     exact pinv (no 0.99 energy truncation); SVHN on the pre-96px cache -> SVHN excluded.

    python scripts/aggregate_steer_concat_comparison.py [--out .log/steer_concatridge_comparison]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = ["EuroSAT", "SVHN", "Cars", "RESISC45", "DTD", "GTSRB", "MNIST", "SUN397"]
NO_SVHN = [d for d in DATASETS if d != "SVHN"]
KS = [1, 2, 5, 10, 20]
SEEDS = [33, 54, 89]
PAIRS = {"vitb": "vitb16-datacomp_to_vitb16-laion2b", "vitl": "vitb16-datacomp_to_vitl14-datacomp"}
METHODS = ["CR-lin", "CR-nl", "GR-lin", "GR-nl", "BR-lin"]
LABELS = {
    "ZS": "zero-shot B",
    "CR-lin": "concat_ridge linear",
    "CR-nl": "concat_ridge nonlinear",
    "GR-lin": "global_ridge linear",
    "GR-nl": "global_ridge nonlinear",
    "BR-lin": "block_ridge linear (trace)",
}

# acc[(pair, method, dataset, K, seed)] = test accuracy; ZS stored as method "ZS".
acc: dict[tuple, float] = {}


def _from_json(path: str) -> tuple[str, float, float] | None:
    summary = json.load(open(path))
    if "test_results" not in summary:  # crashed run (error recorded in run_logging)
        return None
    tr = summary["test_results"]
    (task, rebased), = tr["per_task_absolute_accuracy"].items()
    return task, rebased, tr["per_task_baseline_accuracy"][task]


def _put(pair, method, task, k, seed, rebased, baseline=None):
    acc[(pair, method, task, k, seed)] = rebased
    if baseline is not None and method.startswith("CR"):
        acc[(pair, "ZS", task, k, seed)] = baseline


def load() -> None:
    for pair, pdir in PAIRS.items():
        for method, root in (("CR-lin", "steer_concatridge_linear"), ("CR-nl", "steer_concatridge_nonlinear"),
                             ("GR-nl", "steer_globalridge_nonlinear")):
            for f in glob.glob(f".log/{root}/{pdir}/summaries/*_K*_seed*.json"):
                m = re.search(r"/([A-Za-z0-9]+)_K(\d+)_seed(\d+)\.json$", f)
                task, k, seed = m.group(1), int(m.group(2)), int(m.group(3))
                if method == "GR-nl" and task == "SVHN":
                    continue
                if (res := _from_json(f)) is not None:
                    _put(pair, method, res[0], k, seed, res[1], res[2])
        sub = "vitb_vitb" if pair == "vitb" else "vitb_vitl"
        for f in glob.glob(f".log/{sub}_globalridge_linear_subset/summaries/*.json"):
            m = re.search(r"subset_([A-Za-z0-9]+)_fs(\d+)_seed(\d+)\.json$", f)
            if m and (res := _from_json(f)) is not None:
                _put(pair, "GR-lin", res[0], int(m.group(2)), int(m.group(3)), res[1])
        for f in glob.glob(f".log/{sub}_trace/summaries/*.json"):
            m = re.search(r"trace_([A-Za-z0-9]+)_fs(\d+)_seed(\d+)\.json$", f)
            if m and (res := _from_json(f)) is not None and res[0] != "SVHN":
                _put(pair, "BR-lin", res[0], int(m.group(2)), int(m.group(3)), res[1])
        for feats, method in (("linear", "GR-lin"), ("nonlinear", "GR-nl")):
            for f in glob.glob(f".log/steer_svhn96_noaug_s1energy/global/{feats}/{sub}/global_ridge/SVHN/*_shots/seed_*.log"):
                m = re.search(r"/(\d+)_shots/seed_(\d+)_job", f)
                rows = [line.split() for line in open(f) if line.startswith(" SVHN ")]
                if rows:
                    _put(pair, method, "SVHN", int(m.group(1)), int(m.group(2)), float(rows[-1][2]))


def cell(pair, method, datasets, k):
    """Mean +- std over seeds of the accuracy averaged over `datasets`; None unless every dataset/seed is present."""
    per_seed = []
    for seed in SEEDS:
        vals = [acc.get((pair, method, d, k, seed)) for d in datasets]
        if any(v is None for v in vals):
            return None
        per_seed.append(sum(vals) / len(vals))
    return statistics.mean(per_seed), statistics.stdev(per_seed)


def fmt(c, best=False):
    if c is None:
        return "—"
    s = f"{100 * c[0]:.1f} ± {100 * c[1]:.1f}"
    return f"**{s}**" if best else s


def table(pair, rows, datasets_for, ks=KS, row_label="method"):
    head = f"| {row_label} | " + " | ".join(f"K={k}" for k in ks) + " |"
    lines = [head, "|" + "---|" * (len(ks) + 1)]
    cells = {r: [cell(pair, r, datasets_for(r), k) for k in ks] for r in rows}
    best = [max((cells[r][i][0] for r in rows if r != "ZS" and cells[r][i] is not None), default=None) for i in range(len(ks))]
    for r in rows:
        lines.append(f"| {LABELS.get(r, r)} | " + " | ".join(
            fmt(c, best=c is not None and r != "ZS" and c[0] == best[i]) for i, c in enumerate(cells[r])) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=".log/steer_concatridge_comparison")
    args = p.parse_args()
    load()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ["ZS"] + METHODS
    md = ["# steer concat_ridge vs global_ridge vs block_ridge (vision8, test acc %, mean ± std over seeds 33/54/89)\n"]
    for pair, title in (("vitb", "ViT-B/16 datacomp → ViT-B/16 laion2b"), ("vitl", "ViT-B/16 datacomp → ViT-L/14 datacomp")):
        md.append(f"## {title}\n")
        md.append("### Average over 7 datasets (no SVHN: block_ridge SVHN used a different preprocessing)\n")
        md.append(table(pair, rows, lambda r: NO_SVHN) + "\n")
        md.append("### Average over all 8 datasets\n")
        md.append(table(pair, [r for r in rows if r != "BR-lin"], lambda r: DATASETS) + "\n")
        md.append("### Per dataset\n")
        for d in DATASETS:
            md.append(f"#### {d}\n")
            md.append(table(pair, rows, lambda r, d=d: [d]) + "\n")
    (out / "comparison.md").write_text("\n".join(md))
    with open(out / "per_run.csv", "w") as fh:
        fh.write("pair,method,dataset,K,seed,test_acc\n")
        for (pair, method, d, k, seed), v in sorted(acc.items()):
            fh.write(f"{pair},{method},{d},{k},{seed},{v:.6f}\n")
    print(f"wrote {out}/comparison.md and per_run.csv ({len(acc)} values)")


if __name__ == "__main__":
    main()
