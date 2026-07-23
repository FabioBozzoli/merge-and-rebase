#!/usr/bin/env python3
"""List or execute a stage from experiments/camera_ready_plan.json."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/camera_ready_plan.json"))
    parser.add_argument("--stage", type=str, default=None, help="Run or display one named stage.")
    parser.add_argument("--execute", action="store_true", help="Execute only concrete commands without placeholders.")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    stages = [item for item in plan.get("stages", []) if isinstance(item, dict)]
    selected = [item for item in stages if args.stage is None or item.get("name") == args.stage]
    if not selected:
        raise ValueError(f"No stage named '{args.stage}'.")
    for stage in selected:
        command = str(stage["command"])
        print(f"{stage['name']}: {command}")
        if args.execute:
            if "<" in command or stage["name"] in {"seed-replicates", "efficiency-ablation", "hpo-sensitivity"}:
                raise ValueError(f"Stage '{stage['name']}' requires concrete checkpoint paths/configs before execution.")
            subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
