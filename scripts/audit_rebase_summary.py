#!/usr/bin/env python3
"""Validate raw and normalized rebase metrics before table generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merge_and_rebase.eval.rebase_metrics import audit_rebase_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to a vision_rebase JSON summary.")
    args = parser.parse_args()

    with args.summary.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Summary root must be a JSON object.")
    for message in audit_rebase_summary(payload):
        print(message)


if __name__ == "__main__":
    main()
