#!/usr/bin/env python3
"""Add locally verified checkpoint files to an artifact manifest bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merge_and_rebase.artifacts import sha256_file


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact {path} is outside artifact root {root}.") from error
    if not path.is_file():
        raise ValueError(f"Artifact is not a file: {path}")
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Checkpoint or summary files to include.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--bundle", required=True, help="Bundle to create or update.")
    parser.add_argument("--root", type=Path, default=Path("src/checkpoints"), help="Local artifact root.")
    parser.add_argument("--url-prefix", required=True, help="Immutable URL prefix, e.g. hf-hub:anonymous-neurips-2199/benchmark-artifacts")
    parser.add_argument("--release", action="store_true", help="Mark the bundle released after all URLs are public.")
    parser.add_argument(
        "--release-notes",
        help="Required with --release; provenance, license, and compatibility note stored with every artifact.",
    )
    args = parser.parse_args()
    if args.release and not args.release_notes:
        parser.error("--release-notes is required when --release is set.")

    manifest: dict[str, Any]
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "bundles": []}
    if int(manifest.get("schema_version", 0)) != 1 or not isinstance(manifest.get("bundles"), list):
        raise ValueError("Manifest must use schema_version=1 with a bundles list.")

    bundle = next((item for item in manifest["bundles"] if isinstance(item, dict) and item.get("name") == args.bundle), None)
    if bundle is None:
        bundle = {"name": args.bundle, "status": "pending_release", "artifacts": []}
        manifest["bundles"].append(bundle)
    if not isinstance(bundle.get("artifacts"), list):
        raise ValueError(f"Bundle '{args.bundle}' has an invalid artifacts field.")

    entries = []
    prefix = args.url_prefix.rstrip("/")
    for path in args.files:
        relative = _safe_relative(path, args.root)
        entries.append(
            {
                "name": relative.name,
                "path": relative.as_posix(),
                "url": f"{prefix}/{relative.as_posix()}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "metadata": {
                    "source_path": relative.as_posix(),
                    "release_notes": args.release_notes
                    or "Fill base_model, task, strategy, seed, and license before publication."
                },
            }
        )
    existing = {str(item.get("path")): item for item in bundle["artifacts"] if isinstance(item, dict)}
    existing.update({str(item["path"]): item for item in entries})
    bundle["artifacts"] = [existing[key] for key in sorted(existing)]
    if args.release:
        bundle["status"] = "released"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated {args.manifest}: bundle={args.bundle}, artifacts={len(bundle['artifacts'])}, status={bundle['status']}")


if __name__ == "__main__":
    main()
