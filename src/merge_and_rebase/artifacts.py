"""Verified release artifacts for benchmark checkpoint bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from merge_and_rebase.io.ckpt import resolve_ckpt_path


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    path: str
    url: str
    sha256: str
    size_bytes: int | None = None
    metadata: dict[str, Any] | None = None


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Artifact manifest must be a JSON object with schema_version=1.")
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        raise ValueError("Artifact manifest must contain a bundles list.")
    return manifest


def resolve_bundle(manifest: dict[str, Any], name: str) -> list[ArtifactEntry]:
    bundle = next((item for item in manifest["bundles"] if isinstance(item, dict) and item.get("name") == name), None)
    if bundle is None:
        available = ", ".join(str(item.get("name")) for item in manifest["bundles"] if isinstance(item, dict))
        raise ValueError(f"Unknown artifact bundle '{name}'. Available: {available or '(none)'}.")
    if str(bundle.get("status", "released")) != "released":
        raise ValueError(f"Artifact bundle '{name}' is not released yet.")
    entries = bundle.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Artifact bundle '{name}' has no artifacts.")

    out: list[ArtifactEntry] = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError(f"Artifact bundle '{name}' contains a non-object entry.")
        required = ("name", "path", "url", "sha256")
        if any(not isinstance(raw.get(key), str) or not raw[key] for key in required):
            raise ValueError(f"Artifact entry in bundle '{name}' must define non-empty {required}.")
        checksum = str(raw["sha256"]).lower()
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError(f"Artifact '{raw['name']}' has an invalid SHA-256 digest.")
        relative = Path(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact '{raw['name']}' path must be a safe relative path.")
        out.append(
            ArtifactEntry(
                name=str(raw["name"]),
                path=str(relative),
                url=str(raw["url"]),
                sha256=checksum,
                size_bytes=(int(raw["size_bytes"]) if raw.get("size_bytes") is not None else None),
                metadata=(dict(raw["metadata"]) if isinstance(raw.get("metadata"), dict) else None),
            )
        )
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_artifact(entry: ArtifactEntry, *, destination_root: str | Path, overwrite: bool = False) -> Path:
    destination = Path(destination_root) / entry.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if sha256_file(destination) == entry.sha256:
            return destination
        raise ValueError(f"Existing artifact checksum mismatch: {destination}. Use --overwrite to replace it.")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if entry.url.startswith("hf-hub:"):
            shutil.copyfile(resolve_ckpt_path(entry.url), temporary)
        else:
            with urlopen(entry.url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if entry.size_bytes is not None and temporary.stat().st_size != entry.size_bytes:
            raise ValueError(f"Downloaded artifact size mismatch for '{entry.name}'.")
        actual = sha256_file(temporary)
        if actual != entry.sha256:
            raise ValueError(f"Downloaded artifact checksum mismatch for '{entry.name}'.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
