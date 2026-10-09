#!/bin/bash
# Usage: ./scripts/eval.sh <MAG> <BACKBONE> [SPLITS]
set -euo pipefail
cd "$(dirname "$0")/.."

MAG=${1:-}
SELECTED_BACKBONE=${2:-}
SPLITS=${3:-tcga_idh_100}
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./scripts/eval.sh <mag> <backbone> [splits]"
    echo "Example: ./scripts/eval.sh 20x uni"
    exit 1
fi

if [[ ! " ${BACKBONES[*]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'. Choose from: ${BACKBONES[*]}"
    exit 1
fi

# main.py names the results folder <exp_code>_s<seed> (seed 1 by default).
EXP="idh_${MAG}_${SELECTED_BACKBONE}"
python eval.py \
    --k 10 \
    --models_exp_code "${EXP}_s1" \
    --save_exp_code "${EXP}" \
    --task task_idh_classifier \
    --model_type clam_sb \
    --results_dir results \
    --split test \
    --features_dir "features/${MAG}/${SELECTED_BACKBONE}" \
    --csv_path "dataset_csv/df_idh_label.csv" \
    --splits_dir "splits/${SPLITS}"
