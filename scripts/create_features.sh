#!/bin/bash
# Usage: ./scripts/create_features.sh <MAG> <BACKBONE> [BATCH_SIZE]
set -euo pipefail
cd "$(dirname "$0")/.."

MAG=${1:-}
SELECTED_BACKBONE=${2:-}
BATCH_SIZE=${3:-128}
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./scripts/create_features.sh <mag> <backbone> [batch_size]"
    echo "Example: ./scripts/create_features.sh 20x uni 128"
    exit 1
fi
if [[ ! " ${BACKBONES[*]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'. Choose from: ${BACKBONES[*]}"
    exit 1
fi

python extract_features.py \
    --data_h5_dir "data/slides_patches_$MAG" \
    --data_slide_dir "data/slides_$MAG" \
    --csv_path "dataset_csv/all_slides_gbm_lgg_${MAG}.csv" \
    --feat_dir "features/$MAG/$SELECTED_BACKBONE" \
    --batch_size "$BATCH_SIZE" \
    --slide_ext ".svs" \
    --target_patch_size 224 \
    --model "$SELECTED_BACKBONE"
