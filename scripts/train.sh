#!/bin/bash
# Usage: ./scripts/train.sh <MAG> <BACKBONE> [SPLITS]
#   SPLITS: a folder name under splits/ (default: tcga_idh_100)
set -euo pipefail
cd "$(dirname "$0")/.."

MAG=${1:-}
SELECTED_BACKBONE=${2:-}
SPLITS=${3:-tcga_idh_100}
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./scripts/train.sh <mag> <backbone> [splits]"
    echo "Example: ./scripts/train.sh 20x uni"
    exit 1
fi

if [[ ! " ${BACKBONES[*]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'. Choose from: ${BACKBONES[*]}"
    exit 1
fi

FEATURES_DIR="features/${MAG}/${SELECTED_BACKBONE}"
echo "Magnification: $MAG | Backbone: $SELECTED_BACKBONE | Features: $FEATURES_DIR | Splits: $SPLITS"

# Pure attention-based MIL (CLAM-SB without instance-level clustering).
python main.py \
    --early_stopping \
    --lr 1e-4 \
    --k 10 \
    --label_frac 1.0 \
    --seed 1 \
    --exp_code "idh_${MAG}_${SELECTED_BACKBONE}" \
    --bag_loss ce \
    --task task_idh_classifier \
    --model_type clam_sb \
    --subtyping \
    --no_inst_cluster \
    --log_data \
    --weighted_sample \
    --csv_path "dataset_csv/df_idh_label.csv" \
    --features_dir "$FEATURES_DIR" \
    --split_dir "$SPLITS"
