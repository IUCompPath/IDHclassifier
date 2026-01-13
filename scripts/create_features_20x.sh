#!/bin/bash
nvidia-smi
conda activate idh_classifier

# Positional Arguments
MAG=$1              # e.g., 20x
BATCH_SIZE=${2:-1}  # e.g., 64 (defaults to 1 if not provided)
SELECTED_BACKBONE=$3 # e.g., uni

# Valid Backbone List
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

# Validation: Check if all arguments are present
if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./extract_features.sh [mag] [batch_size] [backbone]"
    echo "Example: ./extract_features.sh 20x 128 uni"
    exit 1
fi

# Directory Mapping based on Magnification
DATA_DIR="data/slides_$MAG"
COORD_DIR="data/slides_patches_$MAG"

# Logic to verify the backbone and execute
if [[ " ${BACKBONES[@]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "-------------------------------------------------------"
    echo "Magnification:  $MAG"
    echo "Batch Size:     $BATCH_SIZE"
    echo "Model Backbone: $SELECTED_BACKBONE"
    echo "Input Dir:      $DATA_DIR"
    echo "-------------------------------------------------------"

    python extract_features_fp.py \
        --data_h5_dir "$COORD_DIR/patches" \
        --data_slide_dir "$DATA_DIR" \
        --csv_path "dataset_csv/all_slides_gbm_lgg_20x.csv" \
        --feat_dir "features/$SELECTED_BACKBONE" \
        --batch_size "$BATCH_SIZE" \
        --slide_ext ".svs" \
        --target_patch_size 224 \
        --model "$SELECTED_BACKBONE"
else
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'."
    echo "Please choose from: ${BACKBONES[*]}"
    exit 1
fi