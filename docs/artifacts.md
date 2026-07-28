# Artifacts and Checkpoints

## Released Bundles

`artifacts/manifest.json` records released files, immutable URLs, byte sizes, and SHA-256 digests. Download a released bundle with validation:

```bash
python scripts/fetch_artifacts.py \
  --bundle vision8-table2 \
  --destination src/checkpoints
```

The downloader rejects unreleased bundles and files whose size or digest differs from the manifest.

## Hosted Checkpoints

Checkpoint references may be local paths, immutable HTTPS URLs, or Hugging Face Hub references:

```text
hf-hub:org/repository/path/to/checkpoint.pt
```

The evaluation loaders download Hub and URL references automatically. The released Vision8 configuration is [configs/vision8_task_arithmetic_hf_release.json](https://github.com/apanariello4/merge-and-rebase/blob/main/configs/vision8_task_arithmetic_hf_release.json).

## TSV Layout

The legacy TSV Vision8 configuration expects a directory tree under `src/checkpoints/tsv`.

```bash
python scripts/download_tsv_checkpoints.py --out src/checkpoints/tsv
python -m merge_and_rebase.eval.vision_merge \
  --config configs/vision8_task_arithmetic.json
```

## Your Own Checkpoints

Fine-tune a compatible base model, save its checkpoint, and provide its local path in `tuned_ckpts`. The importer supports common full/raw state-dict wrappers, normalizes common key prefixes, and loads compatible tensor keys and shapes. This validates structure, not a third-party checkpoint's claimed training provenance.

For a new release, generate manifest entries only from the exact files being published:

```bash
python scripts/build_artifact_manifest.py \
  --bundle my-bundle \
  --root checkpoints \
  --url-prefix hf-hub:org/repository \
  --release \
  checkpoints/path/to/checkpoint.pt
```
