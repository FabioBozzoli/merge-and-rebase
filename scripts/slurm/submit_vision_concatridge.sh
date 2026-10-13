#!/bin/bash
# Submit the vision8 steer concat_ridge grid: one array per (features, pair, dataset),
# array index = few-shot K in {1, 2, 5, 10, 20}; each task runs seeds 33/54/89.
# Job names: cr<l|nl>_<pair>_<Dataset>.
# Resources: measured global_ridge runs (3 seeds/task: vitb <= 3.8 GB, SUN397 10.8 GB; vitl <= 4.8 GB, SUN397
# 11.8 GB, <= 28 min) + the cached B block tensors concat_ridge also loads (fp32, largest split pair):
#   vitb ~0.8 + 0.7 GB, vitl up to ~2.6 GB train (GTSRB) / 2.4 GB test (SVHN) / 1.9 + 1.7 GB (SUN397).
# Requests keep ~1.5x headroom and short walltimes for backfill; any GPU >= 11 GB fits (ViT-L fp32 eval).
# Runs only read existing caches. Resubmitting skips seeds whose summary exists (e.g. after an OOM).
#
#   bash scripts/slurm/submit_vision_concatridge.sh [linear|nonlinear ...]   (env PAIRS="vitb vitl", DATASETS=...)

cd /homes/fbozzoli/merge-and-rebase || exit 1
FEATURE_SETS=("${@:-linear nonlinear}")
read -r -a FEATURE_SETS <<< "${FEATURE_SETS[*]}"
read -r -a PAIRS <<< "${PAIRS:-vitb vitl}"
read -r -a DATASETS <<< "${DATASETS:-EuroSAT SVHN Cars RESISC45 DTD GTSRB MNIST SUN397}"

for FEATURES in "${FEATURE_SETS[@]}"; do
  case "$FEATURES" in
    linear) TAG="crl" ;;
    nonlinear) TAG="crnl" ;;
    *) echo "unknown features $FEATURES" >&2; exit 1 ;;
  esac
  for PAIR in "${PAIRS[@]}"; do
    case "$PAIR" in
      vitb) PAIR_DIR="vitb16-datacomp_to_vitb16-laion2b" ;;
      vitl) PAIR_DIR="vitb16-datacomp_to_vitl14-datacomp" ;;
      *) echo "unknown pair $PAIR" >&2; exit 1 ;;
    esac
    for TASK in "${DATASETS[@]}"; do
      case "$PAIR:$TASK" in
        vitb:SUN397)            RES=(--mem=18G --time=1:00:00) ;;
        vitl:SUN397)            RES=(--mem=24G --time=1:30:00) ;;
        vitl:GTSRB|vitl:SVHN)   RES=(--mem=16G --time=1:00:00) ;;
        vitb:*)                 RES=(--mem=8G  --time=0:40:00) ;;
        vitl:*)                 RES=(--mem=12G --time=1:00:00) ;;
      esac
      DIR=".log/steer_concatridge_${FEATURES}/${PAIR_DIR}/${TASK}"
      for K in 1 2 5 10 20; do mkdir -p "${DIR}/K${K}"; done
      JOB=$(sbatch --parsable --job-name="${TAG}_${PAIR}_${TASK}" --array=1,2,5,10,20 "${RES[@]}" \
        --output="${DIR}/K%a/slurm_%A_%a.out" --error="${DIR}/K%a/slurm_%A_%a.err" \
        scripts/slurm/vision_concatridge.sh "$FEATURES" "$PAIR" "$TASK" 2>/dev/null)
      echo "${JOB}  ${TAG}_${PAIR}_${TASK}  -> ${DIR}/K<k>/"
    done
  done
done
