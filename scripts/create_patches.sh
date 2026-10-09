#!/bin/bash
# Usage: ./scripts/create_patches.sh <20x|40x>
set -euo pipefail
cd "$(dirname "$0")/.."

MAG=${1:-}
case "$MAG" in
    20x) PATCH_SIZE=256 ;;
    40x) PATCH_SIZE=512 ;;
    *) echo "Usage: ./scripts/create_patches.sh <20x|40x>"; exit 1 ;;
esac

DATA_DIR="data/slides_$MAG"
COORD_DIR="data/slides_patches_$MAG"
echo "Input: $DATA_DIR | Output: $COORD_DIR | Patch size: $PATCH_SIZE"

# Step 1: tissue segmentation + patch coordinates
python step_1_patching.py \
    --source "$DATA_DIR" \
    --save_dir "$COORD_DIR" \
    --preset tcga.csv \
    --patch_size "$PATCH_SIZE" \
    --step_size "$PATCH_SIZE" \
    --seg --patch --stitch

# Step 2: remove blank / blurry / non-tissue patches (rewrites the h5 files in place)
python step_1a_patch_cleaning.py \
    --wsi_dir "$DATA_DIR" \
    --h5_dir "$COORD_DIR/patches" \
    --csv_path "$COORD_DIR/process_list_autogen.csv" \
    --patching "$MAG"
