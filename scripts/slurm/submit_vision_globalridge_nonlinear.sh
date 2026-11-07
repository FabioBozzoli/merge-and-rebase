#!/bin/bash
# Submit the vision8 steer global_ridge (non-linear features) grid: one array per (pair, dataset),
# array index = few-shot K in {1, 2, 5, 10, 20}; each task runs seeds 33/54/89.
# Job names: grnl_<pair>_<Dataset>  (grnl = global ridge, non-linear features).
# Resources from measured runs (jobs 146386 vitb / 146387 vitl, 3 seeds per task):
#   vitb: <= 3.6 min, 3.8 GB per run; SUN397 12 min, 10.8 GB
#   vitl: <= 10 min, 4.8 GB per run;  SUN397 28 min, 11.8 GB
#
#   bash scripts/slurm/submit_vision_globalridge_nonlinear.sh [vitb|vitl ...]

cd /homes/fbozzoli/merge-and-rebase || exit 1
PAIRS=("${@:-vitb vitl}")
read -r -a PAIRS <<< "${PAIRS[*]}"
DATASETS=(EuroSAT SVHN Cars RESISC45 DTD GTSRB MNIST SUN397)

for PAIR in "${PAIRS[@]}"; do
  case "$PAIR" in
    vitb) PAIR_DIR="vitb16-datacomp_to_vitb16-laion2b" ;;
    vitl) PAIR_DIR="vitb16-datacomp_to_vitl14-datacomp" ;;
    *) echo "unknown pair $PAIR" >&2; exit 1 ;;
  esac
  for TASK in "${DATASETS[@]}"; do
    case "$PAIR:$TASK" in
      vitb:SUN397) RES=(--mem=14G --time=1:00:00) ;;
      vitl:SUN397) RES=(--mem=16G --time=2:00:00) ;;
      vitb:*)      RES=(--mem=6G  --time=0:40:00) ;;
      vitl:*)      RES=(--mem=8G  --time=1:30:00) ;;
    esac
    DIR=".log/steer_globalridge_nonlinear/${PAIR_DIR}/${TASK}"
    for K in 1 2 5 10 20; do mkdir -p "${DIR}/K${K}"; done
    JOB=$(sbatch --parsable --job-name="grnl_${PAIR}_${TASK}" --array=1,2,5,10,20 "${RES[@]}" \
      --output="${DIR}/K%a/slurm_%A_%a.out" --error="${DIR}/K%a/slurm_%A_%a.err" \
      scripts/slurm/vision_globalridge_nonlinear.sh "$PAIR" "$TASK" 2>/dev/null)
    echo "${JOB}  grnl_${PAIR}_${TASK}  -> ${DIR}/K<k>/"
  done
done
