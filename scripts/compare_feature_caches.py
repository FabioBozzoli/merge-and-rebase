"""Compare two steer_text feature caches split by split, tensor by tensor.

Used to check that the current code reproduces a cache computed elsewhere before reusing it:

    python scripts/compare_feature_caches.py <reference_root> <recomputed_root> [--atol 1e-4] [--rtol 1e-3]

Walks every ``*.pt`` under ``<recomputed_root>`` and loads the file at the same relative path under
``<reference_root>``. For each tensor it reports the shape, the max absolute difference and the max
difference relative to the reference's max magnitude. Exit status 1 if any tensor is missing, has a
different shape, or fails ``torch.allclose``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _flatten(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}[{k}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("recomputed", type=Path)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()

    ok = True
    files = sorted(args.recomputed.rglob("*.pt"))
    if not files:
        print(f"no .pt files under {args.recomputed}")
        return 1
    for f in files:
        rel = f.relative_to(args.recomputed)
        ref_path = args.reference / rel
        if not ref_path.exists():
            print(f"MISSING  {rel}")
            ok = False
            continue
        new = dict(_flatten(torch.load(f, map_location="cpu", weights_only=True)))
        ref = dict(_flatten(torch.load(ref_path, map_location="cpu", weights_only=True)))
        if new.keys() != ref.keys():
            print(f"KEYS     {rel}: {sorted(ref)} vs {sorted(new)}")
            ok = False
            continue
        for k in new:
            a, b = ref[k], new[k]
            if a.shape != b.shape:
                print(f"SHAPE    {rel}{k}: {tuple(a.shape)} vs {tuple(b.shape)}")
                ok = False
                continue
            if not a.is_floating_point():
                same = torch.equal(a, b)
                print(f"{'OK' if same else 'DIFF':8s} {rel}{k} {tuple(a.shape)} exact={same}")
                ok &= same
                continue
            a64, b64 = a.double(), b.double()
            abs_diff = (a64 - b64).abs().max().item()
            rel_diff = abs_diff / max(a64.abs().max().item(), 1e-12)
            close = torch.allclose(a64, b64, atol=args.atol, rtol=args.rtol)
            print(f"{'OK' if close else 'DIFF':8s} {rel}{k} {tuple(a.shape)} max_abs={abs_diff:.3e} max_rel={rel_diff:.3e}")
            ok &= close
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
