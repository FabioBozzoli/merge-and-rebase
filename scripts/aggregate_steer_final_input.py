"""Aggregate steer block_ridge with block_ridge_input="final" on vision8 (alpha=1 fixed).

Test accuracy (%), mean ± std over seeds 33/54/89, per dataset x few-shot K.
Sources (under .log/<exp>/):
  summaries/<exp>_<T>_fs<K>_seed<S>.json  live test accuracy (rebased, target zero-shot)
  <T>/<K>_shots/seed_<S>_job_*.log        cached-space diagnostics printed by prepare():
                                           stage0 (B zero-shot), stage2, independent-equivalent global ridge
Reference: .log/<pair>_trace (same recipe, per-block input; SVHN excluded there, pre-96px live eval).

A run is flagged when cached-space and live B zero-shot differ by > 0.5 pt: the correction is then
fit on features that do not come from the evaluated pipeline (e.g. SVHN without the 96px transform).

    python scripts/aggregate_steer_final_input.py [--pairs vitb vitl] [--out .log/steer_final_input_tables.md]
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
KS = [1, 2, 5, 10, 20]
SEEDS = [33, 54, 89]
PAIRS = {"vitb": "ViT-B/16 datacomp → ViT-B/16 laion2b", "vitl": "ViT-B/16 datacomp → ViT-L/14 datacomp"}
MISMATCH_TOL = 0.005

_LOG_PATTERNS = {
    "cached_zs": r"cached-space B zero-shot test acc = ([0-9.]+)",
    "cached_stage2": r"stage2 \(block_ridge\) cached test acc = ([0-9.]+)",
    "cached_equiv": r"independent-equivalent global ridge cached test acc = ([0-9.]+)",
}


def _summaries(exp: str, pattern: str) -> dict[tuple[str, int, int], tuple[float, float]]:
    out = {}
    for f in glob.glob(f".log/{exp}/summaries/*.json"):
        m = re.search(pattern, f)
        if not m:
            continue
        tr = json.load(open(f))["test_results"]
        (task, rebased), = tr["per_task_absolute_accuracy"].items()
        out[(m.group(1), int(m.group(2)), int(m.group(3)))] = (rebased, tr["per_task_baseline_accuracy"][task])
    return out


def _log_diagnostics(exp: str) -> dict[tuple[str, int, int], dict[str, float]]:
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    for f in sorted(glob.glob(f".log/{exp}/*/*_shots/seed_*_job_*.log")):  # later jobs overwrite earlier ones
        m = re.search(r"/([A-Za-z0-9]+)/(\d+)_shots/seed_(\d+)_job_", f)
        text = open(f).read()
        diag = {k: float(v.group(1)) for k, p in _LOG_PATTERNS.items() if (v := re.search(p, text))}
        if len(diag) == len(_LOG_PATTERNS):
            out[(m.group(1), int(m.group(2)), int(m.group(3)))] = diag
    return out


def _cell(values: list[float], n_expected: int = len(SEEDS)) -> str:
    if not values:
        return "—"
    mean = 100 * statistics.fmean(values)
    std = 100 * statistics.stdev(values) if len(values) > 1 else 0.0
    flag = "" if len(values) == n_expected else f" (n={len(values)})"
    return f"{mean:.2f} ± {std:.2f}{flag}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def report(pair: str) -> list[str]:
    exp = f"vitb_{pair}_final_input"
    live = _summaries(exp, rf"{exp}_([A-Za-z0-9]+)_fs(\d+)_seed(\d+)\.json$")
    diag = _log_diagnostics(exp)
    trace = _summaries(f"vitb_{pair}_trace", r"trace_([A-Za-z0-9]+)_fs(\d+)_seed(\d+)\.json$")

    invalid = {
        key for key, (_, zs_live) in live.items() if key in diag and abs(diag[key]["cached_zs"] - zs_live) > MISMATCH_TOL
    }

    def seeds(src, t, k, fn):
        return [fn(src[(t, k, s)]) for s in SEEDS if (t, k, s) in src and (t, k, s) not in invalid]

    md = [f"## {PAIRS[pair]} (`{exp}`)", ""]
    md += [
        "Live test accuracy (%) of the rebased model, block_ridge_input=final, smoothed_residual ρ=0.9, "
        "λ₁=1, λ₂=0.1 (trace), residual, α=1. Mean ± std over seeds 33/54/89; ZS = target zero-shot.",
        "",
    ]
    rows = []
    for t in DATASETS:
        zs = sorted({round(v[1], 6) for (tt, _, _), v in live.items() if tt == t})
        row = [t, f"{100 * zs[0]:.2f}" if len(zs) == 1 else ("—" if not zs else "≠")]
        row += [_cell(seeds(live, t, k, lambda v: v[0])) for k in KS]
        rows.append(row)
    avg_row = ["**avg**", ""]
    for k in KS:
        per_ds = [statistics.fmean(v) for t in DATASETS if len(v := seeds(live, t, k, lambda x: x[0])) == len(SEEDS)]
        avg_row.append(f"**{100 * statistics.fmean(per_ds):.2f}**" + ("" if len(per_ds) == len(DATASETS) else f" ({len(per_ds)} ds)") if per_ds else "—")
    zs_all = [v[1] for v in live.values()]
    if zs_all:
        per_ds_zs = [statistics.fmean([v[1] for (tt, _, _), v in live.items() if tt == t]) for t in DATASETS if any(tt == t for tt, _, _ in live)]
        avg_row[1] = f"**{100 * statistics.fmean(per_ds_zs):.2f}**"
    rows.append(avg_row)
    md += _table(["dataset", "ZS"] + [f"K={k}" for k in KS], rows) + [""]

    md += [
        "Δ vs per-block input (`block_ridge` trace, same recipe): live acc(final) − live acc(per-block), pt, "
        "mean over seeds (SVHN not available for per-block). CAVEAT: the per-block runs used older code with "
        "the exact Stage-1 pinv (no 0.99 energy truncation), so this Δ also contains the Stage-1 change.",
        "",
    ]
    rows = []
    for t in DATASETS:
        row = [t]
        for k in KS:
            pairs_ = [
                live[(t, k, s)][0] - trace[(t, k, s)][0]
                for s in SEEDS
                if (t, k, s) in live and (t, k, s) in trace and (t, k, s) not in invalid and t != "SVHN"
            ]
            row.append(f"{100 * statistics.fmean(pairs_):+.2f}" if pairs_ else "—")
        rows.append(row)
    md += _table(["dataset"] + [f"K={k}" for k in KS], rows) + [""]

    md += [
        "Cached-space diagnostics (%, mean over seeds): stage2 = this method, equiv = independent-mode limit "
        "(one global ridge on Σ_b T_b). Δ = stage2 − equiv isolates the smoothed carry.",
        "",
    ]
    rows = []
    for t in DATASETS:
        row = [t]
        for k in KS:
            st = seeds(diag, t, k, lambda d: d["cached_stage2"])
            eq = seeds(diag, t, k, lambda d: d["cached_equiv"])
            row.append(
                f"{100 * statistics.fmean(st):.2f} / {100 * statistics.fmean(eq):.2f} ({100 * (statistics.fmean(st) - statistics.fmean(eq)):+.2f})"
                if st and eq
                else "—"
            )
        rows.append(row)
    md += _table(["dataset"] + [f"K={k} stage2 / equiv (Δ)" for k in KS], rows) + [""]

    missing = [(t, k, s) for t in DATASETS for k in KS for s in SEEDS if (t, k, s) not in live]
    md.append(f"Runs: {len(live)}/120 with summary; missing {len(missing)}; excluded (cached vs live ZS > {100 * MISMATCH_TOL:.1f} pt): {len(invalid)}")
    if invalid:
        md.append("Excluded: " + ", ".join(f"{t} K={k} s={s}" for t, k, s in sorted(invalid)))
    return md + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["vitb", "vitl"], choices=sorted(PAIRS))
    parser.add_argument("--out", type=Path, default=Path(".log/steer_final_input_tables.md"))
    args = parser.parse_args()
    md = ["# steer block_ridge, input = final target feature (vision8)", ""]
    for pair in args.pairs:
        md += report(pair)
    args.out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
