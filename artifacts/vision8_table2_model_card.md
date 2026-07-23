---
library_name: pytorch
tags:
- checkpoint
- vision
- reproducibility
---

# Vision8 Table 2 checkpoints

This repository contains the minimal eight ViT-B/32 task checkpoints used by
`configs/vision8_task_arithmetic_hf_release.json` in the accompanying anonymous
code repository. The configuration downloads the checkpoints automatically via
the `hf-hub:` paths embedded in that file.

The bundle mirrors the public TSV checkpoint distribution used for the reported
Vision8 Task Arithmetic result. `scripts/download_tsv_checkpoints.py` in the
code repository remains an independent source fallback. Use of these files is
subject to the upstream checkpoint distribution's terms and the underlying
model and dataset licenses.

The anonymous code repository's `artifacts/manifest.json` records the SHA-256
digest and size of every released file. It can be fetched and verified with:

```bash
python scripts/fetch_artifacts.py --bundle vision8-table2 --destination checkpoints
```
