#!/usr/bin/env python3
"""Fetch a released, SHA-256-verified benchmark artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from merge_and_rebase.artifacts import fetch_artifact, load_manifest, resolve_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Released bundle name in the manifest.")
    parser.add_argument("--manifest", default="artifacts/manifest.json", help="Artifact manifest path.")
    parser.add_argument("--destination", default="src/checkpoints", help="Destination directory for artifact paths.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files with verified downloads.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    entries = resolve_bundle(manifest, args.bundle)
    for entry in entries:
        path = fetch_artifact(entry, destination_root=args.destination, overwrite=args.overwrite)
        print(f"verified {entry.name}: {path}")


if __name__ == "__main__":
    main()
