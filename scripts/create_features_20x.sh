#!/bin/bash
nvidia-smi
source activate idh_classifier

python extract_features_fp_vit.py --data_h5_dir patches/tcga_gbm_lgg_20x_$1 --data_slide_dir tcga_gbm_lgg_20x_wsi_all \
--csv_path dataset_csv/all_slides_gbm_lgg_20x.csv --feat_dir features/retccl_gbm_lgg_$1 \
--batch_size $2 --slide_ext .svs --target_patch_size 224
