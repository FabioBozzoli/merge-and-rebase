"""Drive the real text_rebase entrypoint, but fan steer_text.prepare out over every
(block_ridge_target_strategy, block_residuals_weighting_strategy) combo and check
each one against an independent reference implementation of the spec.

Usage: python harness.py <config.json> <log-dir>
"""
import runpy
import sys

import torch

import merge_and_rebase.rebase.text.steer_text as st

orig_prepare = st.SteerTextRebase.prepare
orig_fit = st._fit_block_ridge
orig_load = st._load_or_compute_split

rec: dict = {}


def spy_fit(blocks, targets, **kw):
    rec["fit_blocks"], rec["fit_targets"], rec["fit_kw"] = blocks, targets.clone(), kw
    out = orig_fit(blocks, targets, **kw)
    rec["raw_coefs"] = [c.clone() for c in out]
    return out


def spy_load(*, split, **kw):
    data = orig_load(split=split, **kw)
    rec[f"data_{split}"] = data
    return data


st._fit_block_ridge = spy_fit
st._load_or_compute_split = spy_load

COMBOS = [(t, w) for t in ("reuse_logitmap", "blockwise_logitmap", "last_only") for w in ("identity", "mean")]
RETURN = ("blockwise_logitmap", "mean")  # the prepared object handed back to the live eval
BASELINE_STAGE2 = 0.5863  # rte fs200 seed33 grid run (reuse_logitmap semantics, identity weights)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def wrapped(self, **kwargs):
    results = {}
    blk_lambda = float(kwargs.get("block_ridge_blockwise_stage1_lambda", 1.0))
    for combo in COMBOS:
        t, w = combo
        rec.clear()
        kw = dict(kwargs, block_ridge_target_strategy=t, block_residuals_weighting_strategy=w, verbose=True)
        print(f"\n===== combo target={t} weighting={w} =====", flush=True)
        prep = orig_prepare(self, **kw)
        results[combo] = prep
        art = prep["artifacts"]

        # ---- independent reference ----------------------------------------
        tr = rec["data_train"]
        f_a, dA, f_b = tr["features_A"].double(), tr["delta_A"].double(), tr["features_B"].double()
        dAb = tr["delta_A_blocks"].double()
        sel = art["selected"]
        w_a, w_b = art["w_a"], art["w_b"]
        p_b = torch.linalg.pinv(w_b)
        lam1 = float(art["stage1_lambda"])

        def stage1_map(delta):  # the spec, written out: pinv(delta[S]) @ residual / (1+lam), transposed
            resid = (f_a[sel] + delta[sel]) @ w_a.T - f_b[sel] @ w_b.T
            return (torch.linalg.pinv(delta[sel]) @ resid).T

        M = stage1_map(dA) / (1 + lam1)
        train_target = dA[sel] @ M.T @ p_b.T
        L = dAb.shape[1]
        if t == "reuse_logitmap":
            ref_T = torch.stack([dAb[sel, b] @ M.T @ p_b.T for b in range(L)], dim=1)
        elif t == "blockwise_logitmap":
            ref_T = torch.stack(
                [dAb[sel, b] @ (stage1_map(dAb[:, b]) / (1 + blk_lambda)).T @ p_b.T for b in range(L)], dim=1
            )
        else:
            ref_T = train_target.unsqueeze(1).repeat(1, L, 1)

        got_T = rec["fit_targets"]
        print(f"[check] targets shape got={tuple(got_T.shape)} ref={tuple(ref_T.shape)}  rel.err={rel(got_T, ref_T):.2e}")
        print(f"[check] sum_b targets vs stage-1 train_target: rel.err={rel(got_T.sum(1), train_target):.3e}")
        print(f"[check] per-block target norms / stage-1 target norm: "
              f"{[round(float(got_T[:, b].norm() / train_target.norm()), 3) for b in range(L)]}")
        wt = 1.0 if w == "identity" else 1.0 / L
        coefs = art["stage2_state"]["coefficients"]
        cerr = max(rel(c.cpu(), r * wt) for c, r in zip(coefs, rec["raw_coefs"]))
        print(f"[check] weighting: expected w={wt:.4f}, n_coefs={len(coefs)} (L={L}), max rel.err={cerr:.2e}")
        lm_err = rel(art["stage1_logit_map"].cpu(), M / 1.0)
        print(f"[check] artifacts['stage1_logit_map'] == stage-1 full-delta map? rel.err={lm_err:.2e}")
        print(f"[result] stage1={prep['diagnostics']['stage1_test_acc']:.4f} "
              f"stage2={prep['diagnostics']['stage2_test_acc']:.4f}")
        if combo == ("reuse_logitmap", "identity"):
            print(f"[check] regression vs grid run stage2={BASELINE_STAGE2}: "
                  f"got {prep['diagnostics']['stage2_test_acc']:.4f}")

    # Default (no strategy kwargs) must be the original reuse_logitmap/identity fit.
    rec.clear(); d = orig_prepare(self, **dict(kwargs, verbose=False))
    ref = results[("reuse_logitmap", "identity")]
    print("[check] default == reuse/identity:", d["diagnostics"]["stage2_test_acc"] == ref["diagnostics"]["stage2_test_acc"],
          all(torch.equal(a, b) for a, b in zip(d["artifacts"]["stage2_state"]["coefficients"], ref["artifacts"]["stage2_state"]["coefficients"])))
    for bad in ({"block_ridge_target_strategy": "blockwise"}, {"block_residuals_weighting_strategy": "avg"}):
        try:
            orig_prepare(self, **dict(kwargs, verbose=False, **bad)); print("[check] invalid accepted!", bad)
        except ValueError as e:
            print("[check] invalid rejected:", e)
    print("[check] stage2_state keys:", {k: v for k, v in results[RETURN]["artifacts"]["stage2_state"].items() if k != "coefficients"})

    # Rank of the per-block support deltas: when full row rank, delta_b[S] @ pinv(delta_b[S]) = I,
    # so the blockwise train target collapses to the block's own residual (no delta_b structure left).
    tr = rec["data_train"]
    dAb = tr["delta_A_blocks"].double()
    sel = results[COMBOS[0]]["artifacts"]["selected"]
    ranks = [int(torch.linalg.matrix_rank(dAb[sel, b])) for b in range(dAb.shape[1])]
    print(f"\n[info] n_support={sel.numel()}  d_A={dAb.shape[-1]}  rank(delta_b[S]) per block={ranks}")

    print("\n===== summary (stage2 cached test acc) =====")
    for combo, prep in results.items():
        print(f"  {combo[0]:>20s} / {combo[1]:<8s}  stage2={prep['diagnostics']['stage2_test_acc']:.4f}")
    print(f"\nReturning combo {RETURN} to the live eval; its cached stage2 = "
          f"{results[RETURN]['diagnostics']['stage2_test_acc']:.4f}", flush=True)
    import os
    if os.environ.get("STOP_AFTER_PREPARE"):
        raise SystemExit(0)
    return results[RETURN]


st.SteerTextRebase.prepare = wrapped

config, logdir = sys.argv[1], sys.argv[2]
sys.argv = ["text_rebase", "--config", config, "--local-log-dir", logdir, "--run-name", "summary"]
runpy.run_module("merge_and_rebase.eval.text_rebase", run_name="__main__")
