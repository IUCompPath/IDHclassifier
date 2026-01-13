#$ -S /bin/bash
conda activate idh_classifier
python extract_features_fp_vit_orig.py --data_h5_dir patches/tcga_gbm_lgg_40x_$1 \
--data_slide_dir tcga_gbm_lgg_40x_wsi_all --csv_path dataset_csv/all_slides_gbm_lgg_40x.csv \
--feat_dir features/retccl_gbm_lgg_$1_orig --batch_size $2 --slide_ext .svs --custom_downsample 2 --target_patch_size 224
