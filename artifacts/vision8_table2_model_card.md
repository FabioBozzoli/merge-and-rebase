---
library_name: pytorch
tags:
- checkpoint
- vision
- reproducibility
---

# Vision8 Table 2 checkpoints

This repository contains the eight full fine-tuned ViT-B/32 OpenAI task
checkpoints used by `configs/vision8_task_arithmetic_hf_release.json` in the
accompanying anonymous code repository. It is the exact checkpoint set
referenced by the submitted `configs/vision8_task_arithmetic_my_check.json`
configuration: Cars, DTD, MNIST, SVHN, EuroSAT, GTSRB, RESISC45, and SUN397.

The public TSV distribution remains available as an independent reproduction
route. In the anonymous code repository, run:

```bash
python scripts/download_tsv_checkpoints.py --out src/checkpoints/tsv
```

If the script cannot complete because Google Drive is throttled or requires
browser consent, manually download
https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e and
preserve its `models/checkpoints/...` hierarchy below `src/checkpoints/tsv`.
