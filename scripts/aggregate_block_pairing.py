"""Aggregate the block-pairing shuffle ablation (scripts/slurm/steer_pairing_vision.sh, scripts/slurm/ntk_pairing.sh).

Metric: live test accuracy of the rebased target (``test_results.per_task_rebased``), in %.
Per (pair, dataset, few_shot):
  identity / reversed : mean +- std over the support seeds (33, 54, 89);
  random              : per permutation seed the mean over support seeds, then mean over the 5 permutations;
                        reported with std across permutations (perm) and the mean within-permutation std over
                        support seeds (supp).
  delta               : reversed - identity and random - identity.
Cells with missing runs are flagged with the number of runs actually averaged (n=...).

    python scripts/aggregate_block_pairing.py [--root DIR] [--out FILE.md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/work/intesasanpaolo_phd/merge-and-rebase/steer_pairing")
VISION_DATASETS = ("Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN")
VISION_SHOTS = (1, 2, 5, 10, 20)
TEXT_TASKS = ("mnli", "qnli", "rte", "scitail", "sick", "snli")
TEXT_SHOTS = (200, 300, 500)
SEEDS = (33, 54, 89)
PERM_SEEDS = (0, 1, 2, 3, 4)
CELL = "lin_block_ridge_attn_seg"


def _read(path: Path, task: str) -> tuple[float, float] | None:
    try:
        tr = json.loads(path.read_text())["test_results"]
        return 100 * tr["per_task_rebased"][task], 100 * tr["per_task_baseline_accuracy"][task]
    except (OSError, KeyError, ValueError):
        return None


def collect(root: Path) -> dict:
    """data[pair][dataset][shot][pairing_tag][seed] = (rebased, zeroshot)."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    tags = ["identity", "reversed"] + [f"random{s}" for s in PERM_SEEDS]
    for pair in ("vitb", "vitl"):
        exp = f"vitb_{pair}"
        for tag in tags:
            for ds in VISION_DATASETS:
                for k in VISION_SHOTS:
                    for s in SEEDS:
                        r = _read(root / exp / tag / "summaries" / f"{exp}_{tag}_{ds}_fs{k}_seed{s}.json", ds)
                        if r is not None:
                            data[f"ViT-B/16 -> {'ViT-B/16' if pair == 'vitb' else 'ViT-L/14'}"][ds][k][tag][s] = r
    for target in ("t5-large", "roberta-base"):
        for tag in tags:
            for task in TEXT_TASKS:
                for k in TEXT_SHOTS:
                    for s in SEEDS:
                        r = _read(root / "text" / target / task / f"{CELL}_{tag}_fs{k}_seed{s}" / "summary.json", task)
                        if r is not None:
                            data[f"t5-base -> {target}"][task][k][tag][s] = r
    return data


def _ms(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def summarize(cell: dict) -> dict:
    out: dict = {}
    zs = [v[1] for runs in cell.values() for v in runs.values()]
    out["zeroshot"] = st.mean(zs) if zs else None
    for tag in ("identity", "reversed"):
        vals = [v[0] for v in cell.get(tag, {}).values()]
        out[tag] = (*_ms(vals), len(vals)) if vals else None
    perm_means, supp_stds, n = [], [], 0
    for p in PERM_SEEDS:
        vals = [v[0] for v in cell.get(f"random{p}", {}).values()]
        if vals:
            m, sd = _ms(vals)
            perm_means.append(m)
            supp_stds.append(sd)
            n += len(vals)
    if perm_means:
        m, sd_perm = _ms(perm_means)
        out["random"] = (m, sd_perm, st.mean(supp_stds), n)
    else:
        out["random"] = None
    return out


def fmt(x, expected: int) -> str:
    if x is None:
        return "—"
    flag = "" if x[-1] == expected else f" (n={x[-1]})"
    if len(x) == 4:
        return f"{x[0]:.2f} ± {x[1]:.2f}p / {x[2]:.2f}s{flag}"
    return f"{x[0]:.2f} ± {x[1]:.2f}{flag}"


def delta(a, b) -> str:
    return "—" if a is None or b is None else f"{a[0] - b[0]:+.2f}"


def render(data: dict) -> str:
    lines = [
        "# Block-pairing shuffle ablation (steer block_ridge, smoothed_residual, rho=0.9)",
        "",
        "Live test accuracy (%) of the rebased target. identity/reversed: mean ± std over support seeds 33/54/89. "
        "random: mean over 5 permutations (seeds 0-4) of the per-permutation mean over support seeds; "
        "`±Xp` = std across permutations, `/Ys` = mean std across support seeds. "
        "`(n=..)` = runs averaged when fewer than expected (3 for identity/reversed, 15 for random).",
        "",
    ]
    for pair, per_ds in data.items():
        datasets = [d for d in (VISION_DATASETS if pair.startswith("ViT") else TEXT_TASKS) if d in per_ds]
        shots = VISION_SHOTS if pair.startswith("ViT") else TEXT_SHOTS
        lines += [f"## {pair}", ""]
        lines += ["| dataset | K | zero-shot | identity | reversed | random | Δ rev | Δ rand |", "|---|---|---|---|---|---|---|---|"]
        avg: dict = defaultdict(lambda: defaultdict(list))
        for ds in datasets:
            for k in shots:
                if k not in per_ds[ds]:
                    continue
                s = summarize(per_ds[ds][k])
                zs = "—" if s["zeroshot"] is None else f"{s['zeroshot']:.2f}"
                lines.append(
                    f"| {ds} | {k} | {zs} | {fmt(s['identity'], 3)} | {fmt(s['reversed'], 3)} | {fmt(s['random'], 15)} "
                    f"| {delta(s['reversed'], s['identity'])} | {delta(s['random'], s['identity'])} |"
                )
                for tag in ("identity", "reversed", "random"):
                    if s[tag] is not None:
                        avg[k][tag].append((ds, s[tag][0]))
        lines += ["", f"**Average over datasets ({pair})** — only datasets present for all three pairings at that K.", ""]
        lines += ["| K | #datasets | identity | reversed | random | Δ rev | Δ rand |", "|---|---|---|---|---|---|---|"]
        for k in shots:
            common = set.intersection(*[{d for d, _ in avg[k][t]} for t in ("identity", "reversed", "random")]) if all(avg[k][t] for t in ("identity", "reversed", "random")) else set()
            if not common:
                continue
            m = {t: st.mean(v for d, v in avg[k][t] if d in common) for t in ("identity", "reversed", "random")}
            lines.append(
                f"| {k} | {len(common)} | {m['identity']:.2f} | {m['reversed']:.2f} | {m['random']:.2f} "
                f"| {m['reversed'] - m['identity']:+.2f} | {m['random'] - m['identity']:+.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_compact(data: dict, no_identity: bool = False) -> str:
    """One row per (dataset, K): accuracy (%) averaged over support seeds; random also over the 5 permutations."""
    lines = []
    for pair, per_ds in data.items():
        vision = pair.startswith("ViT")
        shots = VISION_SHOTS if vision else TEXT_SHOTS
        if no_identity:
            lines += [f"### {pair}", "", "| dataset | K | zero-shot | reversed | random | random - reversed |", "|---|---|---|---|---|---|"]
        else:
            lines += [f"### {pair}", "", "| dataset | K | zero-shot | identity | reversed | random | Δ rev | Δ rand |", "|---|---|---|---|---|---|---|---|"]
        for ds in (VISION_DATASETS if vision else TEXT_TASKS):
            for k in shots:
                if ds not in per_ds or k not in per_ds[ds]:
                    continue
                r = summarize(per_ds[ds][k])
                if r["reversed"] is None or r["random"] is None or (r["identity"] is None and not no_identity):
                    continue
                v, n = r["reversed"][0], r["random"][0]
                if no_identity:
                    lines.append(f"| {ds} | {k} | {r['zeroshot']:.2f} | {v:.2f} | {n:.2f} | {n - v:+.2f} |")
                else:
                    i = r["identity"][0]
                    lines.append(f"| {ds} | {k} | {r['zeroshot']:.2f} | {i:.2f} | {v:.2f} | {n:.2f} | {v - i:+.2f} | {n - i:+.2f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-identity", action="store_true", help="--compact for roots without identity runs (drops identity and deltas).")
    ap.add_argument("--compact", action="store_true", help="one row per (dataset, K), seed-averaged, no std columns")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    data = collect(args.root)
    text = render_compact(data, args.no_identity) if args.compact else render(data)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
