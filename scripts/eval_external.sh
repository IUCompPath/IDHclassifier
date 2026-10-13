#!/bin/bash
# Evaluate the TCGA-trained fold models on an independent hold-out cohort (all slides of the cohort are test data).
# Usage: ./scripts/eval_external.sh <MAG> <BACKBONE> [COHORT] [SPLITS_USED_FOR_TRAINING]
#   COHORT: ebrains (default) | path-free custom cohort name; expects
#           features/<COHORT>/<MAG>/<BACKBONE>/pt_files/ and dataset_csv/<LABELS>  (LABELS env, default ebrain_df_label_idh.csv)
# The label csv must use the TCGA coding: 1 = IDH-mutant, 0 = IDH-wildtype.
# Models are read from results/idh_<MAG>_<BACKBONE>_s1/ (written by ./scripts/train.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

MAG=${1:-}
BACKBONE=${2:-}
COHORT=${3:-ebrains}
LABELS=${LABELS:-ebrain_df_label_idh.csv}
K=${K:-10}
RESULTS_DIR=${RESULTS_DIR:-results}
EXP_CODE=${EXP_CODE:-idh_${MAG}_${BACKBONE}}

if [ -z "$MAG" ] || [ -z "$BACKBONE" ]; then
    echo "Usage: ./scripts/eval_external.sh <mag> <backbone> [cohort]"
    exit 1
fi

EXP=$EXP_CODE
python eval.py \
    --k "$K" \
    --models_exp_code "${EXP}_s1" \
    --save_exp_code "${EXP}_${COHORT}" --eval_dir "${EVAL_DIR:-eval_results}" \
    --task task_idh_classifier \
    --model_type clam_sb \
    --results_dir "$RESULTS_DIR" \
    --split all \
    --features_dir "features/${COHORT}/${MAG}/${BACKBONE}" \
    --csv_path "dataset_csv/${LABELS}"
