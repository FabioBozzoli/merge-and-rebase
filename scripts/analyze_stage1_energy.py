"""Ablation of Stage 1's energy-truncated pinv for steer concat_ridge, in cached-feature space.

For one (pair, regime, dataset), sweeps stage1_energy_threshold over --thresholds for every few-shot K and
seed, with everything else fixed as in the concat_ridge grid (stage1_lambda=1, ridge_lambda=1, residual
blocks + final feature, per-block trace scaling, cached heads). Reports, per threshold:
  kept/rank of the support-set pinv, Stage 1 oracle test acc (uses A's delta at test time) and the
  concat_ridge test acc predicted from B's cached features (the "stage2 cached test acc" of prepare(),
  which matched the live alpha=1 eval within ~0.3 points in the grid).
The concat ridge is solved in dual form, so its kernels are built once per (K, seed) and only the
Stage 1 target changes across thresholds. Threshold 0.99 must reproduce the grid logs (checked by
--check-log-root).

    python scripts/analyze_stage1_energy.py --pair vitb --regime nonlinear --task DTD --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from merge_and_rebase.rebase.methods import steer

FEATURES = Path("/work/intesasanpaolo_phd/merge-and-rebase/features")
PAIRS = {
    "vitb": ("ViT-B-16_datacomp_xl_s13b_b90k_to_ViT-B-16_laion2b_s34b_b88k", "vitb16-datacomp_to_vitb16-laion2b"),
    "vitl": ("ViT-B-16_datacomp_xl_s13b_b90k_to_ViT-L-14_datacomp_xl_s13b_b90k", "vitb16-datacomp_to_vitl14-datacomp"),
}
TRAIN_SPLIT = {"SVHN": "train_50perclass", "MNIST": "train_50perclass"}


def load(path: Path, name: str) -> torch.Tensor | dict:
    return torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", choices=PAIRS, required=True)
    p.add_argument("--regime", choices=["linear", "nonlinear"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.95, 0.99, 0.999, 1.0])
    p.add_argument("--few-shots", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 54, 89])
    p.add_argument("--stage1-lambda", type=float, default=1.0)
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--check-log-root", default=".log")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    pair_dir, log_pair = PAIRS[a.pair]
    regime_dir = "linear" if a.regime == "linear" else "standard"
    base = FEATURES / ("linear_feature" if a.regime == "linear" else "nonlinear_feature") / pair_dir / a.task / regime_dir
    blocks_base = FEATURES / "linear_feature" / pair_dir / a.task / "linear"
    train_split = TRAIN_SPLIT.get(a.task, "train")

    tr, te = base / train_split, base / "test"
    f_a, delta_a, f_b = (load(tr, n).double() for n in ("features_A", "delta_A", "features_B"))
    y = load(tr, "y_A").long()
    delta_a_test, f_b_test = load(te, "delta_A").double(), load(te, "features_B").double()
    y_test = load(te, "y_A").long()
    w_a = steer._load_cached_head(base / "train" / "head_A.pt")
    w_b = steer._load_cached_head(base / "train" / "head_B.pt")
    p_b = torch.linalg.pinv(w_b)
    # Same consistency check prepare() applies before reading blocks from the linear cache.
    for split, data in ((train_split, {"features_B": f_b, "y_A": y}), ("test", {"features_B": f_b_test, "y_A": y_test})):
        steer._attach_external_target_blocks(dict(data), blocks_base / split, verbose=False)
    blocks_train = load(blocks_base / train_split, "features_B_blocks")
    blocks_test = load(blocks_base / "test", "features_B_blocks")
    zero_shot = steer._accuracy(f_b_test, w_b, y_test)
    print(f"{a.pair} {a.regime} {a.task}: loaded in {time.time() - t0:.0f}s, zero-shot {zero_shot:.4f}", flush=True)

    rows = []
    for k in a.few_shots:
        for seed in a.seeds:
            sel = steer._few_shot(y, k, seed)
            n = sel.numel()
            gram = torch.zeros(n, n, dtype=torch.float64, device=dev)
            cross = torch.zeros(f_b_test.shape[0], n, dtype=torch.float64, device=dev)
            for b in sorted(blocks_train):
                x_sel = blocks_train[b][sel].double().to(dev)
                scale2 = float(x_sel.square().sum()) / n
                gram += x_sel @ x_sel.T / scale2
                for start in range(0, blocks_test[b].shape[0], 8192):
                    chunk = blocks_test[b][start : start + 8192].to(dev).double()
                    cross[start : start + chunk.shape[0]] += chunk @ x_sel.T / scale2
            system = gram + a.ridge_lambda * torch.eye(n, dtype=torch.float64, device=dev)
            for thr in a.thresholds:
                logit_map, kept, rank = steer._stage1_projection(
                    f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=sel,
                    regularization=a.stage1_lambda, energy_threshold=thr,
                )
                target = (delta_a[sel] @ logit_map.T @ p_b.T).to(dev)
                pred = (cross @ torch.linalg.solve(system, target)).cpu()
                stage1 = steer._accuracy(f_b_test + delta_a_test @ logit_map.T @ p_b.T, w_b, y_test)
                stage2 = steer._accuracy(f_b_test + pred, w_b, y_test)
                rows.append(dict(pair=a.pair, regime=a.regime, task=a.task, K=k, seed=seed, threshold=thr,
                                 kept=kept, rank=rank, n=n, stage1_oracle=stage1, concat_cached=stage2,
                                 zero_shot=zero_shot))
            print(f"  K={k} seed={seed}: " + " ".join(
                f"{r['threshold']}:{r['kept']}/{r['rank']}={100 * r['concat_cached']:.2f}" for r in rows[-len(a.thresholds):]),
                flush=True)

    # Reproduction check against the grid logs at the grid's threshold (0.99).
    mismatches = []
    for r in rows:
        if r["threshold"] != 0.99:
            continue
        log = Path(a.check_log_root) / f"steer_concatridge_{a.regime}" / log_pair / a.task / f"K{r['K']}" / f"seed{r['seed']}.log"
        m = re.search(r"stage2 \(concat_ridge\) cached test acc = ([0-9.]+)", log.read_text()) if log.exists() else None
        if m is None or abs(float(m.group(1)) - r["concat_cached"]) > 1e-3:
            mismatches.append((r["K"], r["seed"], m.group(1) if m else None, round(r["concat_cached"], 4)))
    print(f"reproduction check at 0.99: {'OK' if not mismatches else f'MISMATCH {mismatches}'}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.pair}_{a.regime}_{a.task}.json").write_text(json.dumps({"rows": rows, "mismatches": mismatches}, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
