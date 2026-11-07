"""Goodness of fit of the steer Stage-2 ridges across feature granularities.

All granularities regress the SAME Stage-2 target (A's feature correction mapped into B's feature
space, ``delta_A @ logit_map.T @ pinv(W_B).T``); they differ only in the predictors taken from B:
the final feature (global), the 13 residual-block outputs (block_granularity="residual") or the ~50
linear-operation sites (block_granularity="linear"). Since sum_b delta_A_blocks == delta_A, the
block methods predict the same total target, so R^2 on that total is directly comparable.

Fits reuse steer.py (_few_shot, _stage1_projection, _ridge, _fit_block_ridge, _predict_block_ridge)
exactly as SteerRebase.prepare() does, on the cached features. Reported, per few_shot x seed:
  - r2_test: out-of-sample R^2 on the full test split (the one that matters: the support has
    at most 200 points, far fewer than the predictors, so in-sample R^2 is ~1 by construction);
  - r2_train: in-sample R^2 on the support set;
  - acc_test: cached-space test accuracy of f_B + prediction (= the run's "stage2 cached test acc");
  - cumulative test R^2 of the block methods as blocks are added, from input to output.
R^2 = 1 - sum ||y - y_hat||^2 / sum ||y - mean_test(y)||^2 (pooled over the 512 output dims).

Both caches must hold the same samples in the same order (checked). Example:
  python scripts/analyze_steer_ridge_r2.py --task MNIST \
      --residual-cache <.../MNIST/linear> --linear-cache <.../MNIST/linear_lineargranularity> \
      --out logs/steer_ridge_r2/MNIST_vitb_vitb.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge_and_rebase.rebase.methods import steer  # noqa: E402

FEW_SHOTS = (1, 2, 5, 10, 20)
SEEDS = (33, 54, 89)


def load_split(root: Path, split: str) -> dict:
    return {name: torch.load(root / split / f"{name}.pt", map_location="cpu")
            for name in ("features_A", "delta_A", "features_B", "y_A", "delta_A_blocks", "features_B_blocks")}


def load_head(path: Path) -> torch.Tensor:
    head = torch.load(path, map_location="cpu")
    return (head["weight"] if isinstance(head, dict) else head).double()


def r2(y: torch.Tensor, y_hat: torch.Tensor, y_mean: torch.Tensor) -> float:
    return float(1.0 - (y - y_hat).square().sum() / (y - y_mean).square().sum())


def check_same(a: dict, b: dict, split: str) -> None:
    if not torch.equal(a["y_A"], b["y_A"]):
        raise SystemExit(f"{split}: residual and linear caches hold different samples (labels differ).")
    err = float((a["features_B"].double() - b["features_B"].double()).abs().max())
    if err > 1e-4:
        raise SystemExit(f"{split}: features_B differ between caches (max abs {err:.2e}): different samples.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--residual-cache", type=Path, required=True, help=".../<task>/linear")
    p.add_argument("--linear-cache", type=Path, required=True, help=".../<task>/linear_lineargranularity")
    p.add_argument("--train-split", default="train_50perclass")
    p.add_argument("--heads", type=Path, default=None, help="dir with head_A.pt/head_B.pt (default: <linear-cache>/train)")
    p.add_argument("--stage1-lambda", type=float, default=1.0)
    # 1.0 = exact pinv, as in the MNIST analysis reported before the energy-truncated Stage 1.
    p.add_argument("--stage1-energy-threshold", type=float, default=1.0)
    p.add_argument("--global-lambda", type=float, default=1.0)
    p.add_argument("--block-lambda", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    heads = a.heads or a.linear_cache / "train"
    w_a, w_b = load_head(heads / "head_A.pt"), load_head(heads / "head_B.pt")
    data = {}
    for gran, root in (("residual", a.residual_cache), ("linear", a.linear_cache)):
        data[gran] = {"train": load_split(root, a.train_split), "test": load_split(root, "test")}
    for split in ("train", "test"):
        check_same(data["residual"][split], data["linear"][split], split)

    base_tr, base_te = data["linear"]["train"], data["linear"]["test"]
    f_a, delta_a, f_b = (base_tr[k].double() for k in ("features_A", "delta_A", "features_B"))
    labels = base_tr["y_A"].long()
    delta_a_te, f_b_te, labels_te = base_te["delta_A"].double(), base_te["features_B"].double(), base_te["y_A"].long()
    blocks = {}
    for gran in ("residual", "linear"):
        tr, te = data[gran]["train"], data[gran]["test"]
        n_blocks = tr["delta_A_blocks"].shape[1]
        if len(tr["features_B_blocks"]) != n_blocks:
            # Equal-depth pair: one predictor group per source block (identity grouping).
            raise SystemExit(f"{gran}: {len(tr['features_B_blocks'])} target blocks vs {n_blocks} source blocks; "
                             "only equal-depth pairs are supported here.")
        blocks[gran] = dict(
            delta_tr=tr["delta_A_blocks"].double(),
            x_tr={b: tr["features_B_blocks"][b].double() for b in range(n_blocks)},
            x_te={b: te["features_B_blocks"][b].double() for b in range(n_blocks)},
            n=n_blocks,
        )
        del data[gran]
    p_b = torch.linalg.pinv(w_b)

    results = []
    for few_shot in FEW_SHOTS:
        for seed in SEEDS:
            sel = steer._few_shot(labels, few_shot, seed)
            logit_map, kept_rank, rank = steer._stage1_projection(
                f_a=f_a, delta_a=delta_a, w_a=w_a, f_b=f_b, w_b=w_b, selected=sel,
                regularization=a.stage1_lambda, energy_threshold=a.stage1_energy_threshold)
            proj = logit_map.T @ p_b.T
            y_tr, y_te = delta_a[sel] @ proj, delta_a_te @ proj
            mean_tr, mean_te = y_tr.mean(0, keepdim=True), y_te.mean(0, keepdim=True)
            row = dict(few_shot=few_shot, seed=seed, n_support=int(sel.numel()), stage1_kept_rank=kept_rank, stage1_rank=rank,
                       acc_zeroshot=steer._accuracy(f_b_te, w_b, labels_te),
                       acc_stage1_oracle=steer._accuracy(f_b_te + y_te, w_b, labels_te), methods={})

            def record(name, pred_tr, pred_te, n_params, extra=None):
                row["methods"][name] = dict(r2_test=r2(y_te, pred_te, mean_te), r2_train=r2(y_tr, pred_tr, mean_tr),
                                            acc_test=steer._accuracy(f_b_te + pred_te, w_b, labels_te),
                                            n_coefficients=n_params, **(extra or {}))

            # Global ridge on B's final feature, as stage_2_strategy="global_ridge" (plain lambda)...
            coef = steer._ridge(f_b[sel], y_tr, a.global_lambda)
            record(f"global_lambda{a.global_lambda:g}", f_b[sel] @ coef, f_b_te @ coef, coef.numel())
            # ...and with the blocks' trace scaling, to separate granularity from regularization.
            x = f_b[sel]
            coef = steer._ridge(x, y_tr, a.block_lambda * float(x.square().sum()) / x.shape[0])
            record(f"global_trace{a.block_lambda:g}", x @ coef, f_b_te @ coef, coef.numel())

            for gran in ("residual", "linear"):
                blk = blocks[gran]
                targets = blk["delta_tr"][sel] @ proj  # [n_sel, n_blocks, d_B]
                x_sel = {b: v[sel] for b, v in blk["x_tr"].items()}
                local = torch.arange(int(sel.numel()))
                for mode in ("smoothed_residual", "independent"):
                    coefs = steer._fit_block_ridge(x_sel, targets, selected=local, regularization=a.block_lambda,
                                                   mode=mode, rho=a.rho, regularization_scaling="trace")
                    contrib_te = [blk["x_te"][b] @ c for b, c in enumerate(coefs)]
                    pred_te = torch.stack(contrib_te).sum(0)
                    pred_tr = steer._predict_block_ridge(coefs, x_sel)
                    cum, run = [], torch.zeros_like(y_te)
                    for c in contrib_te:
                        run = run + c
                        cum.append(r2(y_te, run, mean_te))
                    record(f"{gran}{blk['n']}_{mode}", pred_tr, pred_te, sum(c.numel() for c in coefs),
                           dict(cumulative_r2_test=cum))
            results.append(row)
            print(f"fs={few_shot:2d} seed={seed}: " + "  ".join(
                f"{k}: R2={v['r2_test']:.3f} acc={v['acc_test']*100:.2f}" for k, v in row["methods"].items()), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(task=a.task, residual_cache=str(a.residual_cache), linear_cache=str(a.linear_cache),
                train_split=a.train_split, stage1_lambda=a.stage1_lambda, stage1_energy_threshold=a.stage1_energy_threshold, global_lambda=a.global_lambda,
                block_lambda=a.block_lambda, rho=a.rho, n_test=int(labels_te.numel()),
                n_blocks={g: blocks[g]["n"] for g in blocks})
    a.out.write_text(json.dumps(dict(meta=meta, results=results), indent=1) + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
