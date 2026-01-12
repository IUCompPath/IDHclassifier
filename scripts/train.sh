cd /home/shubham/ssl/CLAM_lunit
python main_ms.py --early_stopping --lr 1e-4 --k 10 --label_frac 1.0 --exp_code tcga_2021_$1 --bag_loss ce --inst_loss svm --task $2 --model_type clam_sb --log_data --data_root_dir /home/shubham/ssl/CLAM_lunit --weighted_sample --features_dir features/features_$3_256 --split_dir task_who_2021_100 --subtyping --no_inst_cluster --csv_path dataset_csv/$4.csv
