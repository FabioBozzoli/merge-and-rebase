from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merge_and_rebase.artifacts import fetch_artifact, load_manifest, resolve_bundle


def _manifest(*, url: str, checksum: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bundles": [
            {
                "name": "smoke",
                "status": "released",
                "artifacts": [
                    {
                        "name": "checkpoint",
                        "path": "vision8/Cars.pt",
                        "url": url,
                        "sha256": checksum,
                    }
                ],
            }
        ],
    }


def test_fetch_artifact_verifies_local_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(url=source.as_uri(), checksum=checksum)), encoding="utf-8")

    entry = resolve_bundle(load_manifest(manifest_path), "smoke")[0]
    destination = fetch_artifact(entry, destination_root=tmp_path / "download")

    assert destination.read_bytes() == b"checkpoint"


def test_manifest_rejects_unreleased_or_unsafe_entries() -> None:
    pending = {"schema_version": 1, "bundles": [{"name": "x", "status": "pending_release", "artifacts": []}]}
    with pytest.raises(ValueError, match="not released"):
        resolve_bundle(pending, "x")

    unsafe = _manifest(url="file:///tmp/example", checksum="0" * 64)
    unsafe["bundles"][0]["artifacts"][0]["path"] = "../escape.pt"  # type: ignore[index]
    with pytest.raises(ValueError, match="safe relative"):
        resolve_bundle(unsafe, "smoke")
