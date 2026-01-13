#!/bin/bash
#$ -S /bin/bash
nvidia-smi
source activate idh_classifier

# Positional Arguments
MAG=$1               # e.g., 20x
SELECTED_BACKBONE=$2 # e.g., uni

# Valid Backbone List
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

# Validation: Check if arguments are present
if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./train.sh [mag] [backbone]"
    echo "Example: ./train.sh 20x uni"
    exit 1
fi

# Verify Backbone and Execute
if [[ " ${BACKBONES[@]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "-------------------------------------------------------"
    echo "Starting Training Workflow"
    echo "Magnification:  $MAG"
    echo "Model Backbone: $SELECTED_BACKBONE"
    echo "Features Dir:   features/$SELECTED_BACKBONE"
    echo "-------------------------------------------------------"

    # Execute main.py with dynamic experiment code and feature directory
    python main.py \
        --early_stopping \
        --lr 1e-4 \
        --k 10 \
        --label_frac 1.0 \
        --exp_code "tcga_gbm_idh_${MAG}_${SELECTED_BACKBONE}" \
        --bag_loss ce \
        --inst_loss svm \
        --task _idh \
        --model_type clam_sb \
        --log_data \
        --data_root_dir /cbica/home/innanis/clam_idh \
        --weighted_sample \
        --features_dir "features/$SELECTED_BACKBONE" \
        --no_inst_cluster
else
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'."
    echo "Please choose from: ${BACKBONES[*]}"
    exit 1
fi