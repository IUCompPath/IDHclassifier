#!/bin/bash
#$ -S /bin/bash
nvidia-smi
source activate idh_classifier

# Positional Arguments
MAG=$1               # e.g., 20x (though mostly used for path mapping here)
SELECTED_BACKBONE=$2 # e.g., uni

# Valid Backbone List
BACKBONES=("ctranspath" "hipt4k" "imagenet" "lunit" "retccl" "simclr" "uni")

# Validation: Check if arguments are present
if [ -z "$MAG" ] || [ -z "$SELECTED_BACKBONE" ]; then
    echo "Usage: ./eval.sh [mag] [backbone]"
    echo "Example: ./eval.sh 20x uni"
    exit 1
fi

# Verify Backbone and Execute
if [[ " ${BACKBONES[@]} " =~ " ${SELECTED_BACKBONE} " ]]; then
    echo "-------------------------------------------------------"
    echo "Starting Evaluation Workflow"
    echo "Magnification:  $MAG"
    echo "Model Backbone: $SELECTED_BACKBONE"
    echo "Loading Models: results/idh_${SELECTED_BACKBONE}_s1"
    echo "Features Dir:   features/$SELECTED_BACKBONE"
    echo "-------------------------------------------------------"

    # Note: main.py appends '_s1' (or whatever the seed is) to the exp_code 
    # for the folder name. We account for that in --models_exp_code.
    
    python eval.py \
        --k 10 \
        --models_exp_code "idh_${SELECTED_BACKBONE}_s1" \
        --save_exp_code "idh_eval_results_${SELECTED_BACKBONE}" \
        --task task_idh_classifier \
        --model_type clam_sb \
        --results_dir results \
        --split test \
        --features_dir "features/$SELECTED_BACKBONE" \
        --csv_path "dataset_csv/df_idh_label.csv" \
        --splits_dir "splits/tcga_idh_100"
else
    echo "Error: Invalid backbone '$SELECTED_BACKBONE'."
    echo "Please choose from: ${BACKBONES[*]}"
    exit 1
fi