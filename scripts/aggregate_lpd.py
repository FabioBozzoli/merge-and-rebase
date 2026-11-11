"""Aggregate the linear probe distillation (LPD, stage1_mode=logit_distill, BASELINE not steer) runs of lambda=1.

Reads test_results.per_task_rebased (LPD) and per_task_baseline_accuracy (zero-shot of B) from the summaries, averages over seeds
(and datasets/tasks), and puts next to them the reference steer (block_ridge, smoothed_residual, rho=0.9, identity pairing)
from the block-pairing ablation, when present. Missing cells are excluded and reported as (n=...).

    python scripts/aggregate_lpd.py [--lpd-root DIR ...] [--steer-root DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

VISION = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
V_K, V_SEEDS = [1, 2, 5, 10, 20], [33, 54, 89]
TEXT = ["snli", "mnli", "qnli", "scitail", "rte", "sick"]
T_K = [200, 300, 500]
TARGETS = {"t5-large": "t5 -> t5-large", "roberta-base": "t5 -> roberta-base"}


def _read(paths: list[Path], task: str):
    for p in paths:
        if p.exists():
            tr = json.loads(p.read_text())["test_results"]
            return float(tr["per_task_rebased"][task]) * 100, float(tr["per_task_baseline_accuracy"][task]) * 100
    return None


def _ms(xs):
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), float("nan"))


def _fmt(xs):
    m, s = _ms(xs)
    return "n/a" if m != m else f"{m:.2f}±{s:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lpd-root", type=Path, nargs="+", default=[Path("/work/fbozzoli_phd/steer_lpd"),
                    Path("/work/intesasanpaolo_phd/merge-and-rebase/steer_lpd")])
    ap.add_argument("--steer-root", type=Path, default=Path("/work/intesasanpaolo_phd/merge-and-rebase/steer_pairing"))
    ap.add_argument("--lam", default="1")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out: list[str] = []

    def table(title, pair_rows, ks, per_seed_fn):
        out.append(f"### {title}\n")
        out.append("| K | zero-shot B | LPD (λ=" + a.lam + ") | Δ LPD − zero-shot | steer ref (identity ρ=0.9) | Δ steer − zero-shot | n LPD |")
        out.append("|---|---|---|---|---|---|---|")
        for k in ks:
            zs, lp, sr, n = per_seed_fn(k)
            d = [l - z for l, z in zip(lp, zs)] if len(lp) == len(zs) else []
            ds = [s - z for s, z in zip(sr, zs[: len(sr)])] if sr and len(sr) == len(zs) else []
            out.append(f"| {k} | {_fmt(zs)} | {_fmt(lp)} | {_fmt(d)} | {_fmt(sr) if sr else 'n/a'} | {_fmt(ds) if ds else 'n/a'} | {n} |")
        out.append("")

    def vision_pair(pair):
        def per_seed(k):
            zs, lp, sr, n = [], [], [], 0
            for s in V_SEEDS:
                zl, ll, sl = [], [], []
                for d in VISION:
                    name = f"{pair}_lpd_lam{a.lam}_{d}_fs{k}_seed{s}"
                    r = _read([root / pair / f"lam{a.lam}" / "summaries" / f"{name}.json" for root in a.lpd_root], d)
                    if r:
                        ll.append(r[0]); zl.append(r[1]); n += 1
                    sr_ = _read([a.steer_root / pair / "identity" / "summaries" / f"{pair}_identity_{d}_fs{k}_seed{s}.json"], d)
                    if sr_: sl.append(sr_[0])
                if len(ll) == len(VISION):          # per-seed mean only over complete seeds
                    zs.append(st.mean(zl)); lp.append(st.mean(ll))
                    if len(sl) == len(VISION): sr.append(st.mean(sl))
            return zs, lp, sr, f"{n}/{len(VISION) * len(V_SEEDS)}"
        return per_seed

    def text_pair(target):
        def per_seed(k):
            zs, lp, sr, n = [], [], [], 0
            for s in V_SEEDS:
                zl, ll, sl = [], [], []
                for t in TEXT:
                    r = _read([root / "text" / target / t / f"lpd_lam{a.lam}_fs{k}_seed{s}" / "summary.json" for root in a.lpd_root], t)
                    if r:
                        ll.append(r[0]); zl.append(r[1]); n += 1
                    sr_ = _read([a.steer_root / "text" / target / t / f"lin_block_ridge_attn_seg_identity_fs{k}_seed{s}" / "summary.json"], t)
                    if sr_: sl.append(sr_[0])
                if len(ll) == len(TEXT):
                    zs.append(st.mean(zl)); lp.append(st.mean(ll))
                    if len(sl) == len(TEXT): sr.append(st.mean(sl))
            return zs, lp, sr, f"{n}/{len(TEXT) * len(V_SEEDS)}"
        return per_seed

    out.append(f"# LPD (stage1_mode=logit_distill, λ={a.lam}) — baseline, non steer\n")
    out.append("Accuratezza di test (%), media sui dataset/task, poi media±std (pop.) tra i 3 seed del support set (33, 54, 89).\n")
    table("ViT-B/16 → ViT-B/16 (8 dataset)", None, V_K, vision_pair("vitb_vitb"))
    for tgt, label in TARGETS.items():
        table(f"{label} (6 task NLI)", None, T_K, text_pair(tgt))

    out.append("## Per dataset / task (media sui seed, tra parentesi la std)\n")
    def detail(title, items, ks, getter):
        out.append(f"### {title}\n")
        out.append("| " + " | ".join(["K"] + [f"{i}" for i in items]) + " |")
        out.append("|" + "---|" * (len(items) + 1))
        for k in ks:
            row = [str(k)]
            for it in items:
                zl, ll = getter(it, k)
                if not ll: row.append("n/a"); continue
                m, s = _ms(ll)
                row.append(f"{m:.1f} ({s:.1f}) [zs {st.mean(zl):.1f}]")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    def vget(pair):
        def g(d, k):
            zl, ll = [], []
            for s in V_SEEDS:
                r = _read([root / pair / f"lam{a.lam}" / "summaries" / f"{pair}_lpd_lam{a.lam}_{d}_fs{k}_seed{s}.json" for root in a.lpd_root], d)
                if r: ll.append(r[0]); zl.append(r[1])
            return zl, ll
        return g
    def tget(target):
        def g(t, k):
            zl, ll = [], []
            for s in V_SEEDS:
                r = _read([root / "text" / target / t / f"lpd_lam{a.lam}_fs{k}_seed{s}" / "summary.json" for root in a.lpd_root], t)
                if r: ll.append(r[0]); zl.append(r[1])
            return zl, ll
        return g
    detail("ViT-B → ViT-B: LPD [zero-shot]", VISION, V_K, vget("vitb_vitb"))
    for tgt, label in TARGETS.items():
        detail(f"{label}: LPD [zero-shot]", TEXT, T_K, tget(tgt))

    text = "\n".join(out)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
