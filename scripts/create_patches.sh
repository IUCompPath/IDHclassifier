#!/bin/bash

# Check if an argument was provided (20x or 40x)
if [ -z "$1" ]; then
    echo "Error: No magnification provided. Usage: ./create_patches.sh 20x"
    exit 1
fi

MAG=$1
source activate idh_classifier
nvidia-smi

# Logic for magnification-specific patch sizes and directories
if [[ "$MAG" == "20x" ]]; then
    echo "Workflow: 20x"
    PATCH_SIZE=256
elif [[ "$MAG" == "40x" ]]; then
    echo "Workflow: 40x"
    PATCH_SIZE=512
else
    echo "Invalid magnification: $MAG. Please use '20x' or '40x'."
    exit 1
fi

# Dynamically update directories based on magnification
DATA_DIR="data/slides_$MAG"
COORD_DIR="data/slides_patches_$MAG"

echo "Input Directory: $DATA_DIR"
echo "Output Directory: $COORD_DIR"
echo "Patch Size: $PATCH_SIZE"

# Execute Python script
python step_1_patching.py \
    --data_dir "$DATA_DIR" \
    --coordinates_dir "$COORD_DIR" \
    --patch_size $PATCH_SIZE \
    --seg \
    --patch \
    --stitch


# STEP 2: Coordinate Cleanup (Filtering white/dark/junk)
echo "Running Cleanup Step..."
python step_2_cleanup.py \
    --wsi_dir "$DATA_DIR" \
    --h5_dir "$COORD_DIR/patches" \
    --csv_path "$COORD_DIR/process_list_autogen.csv" \
    --patching "$MAG"