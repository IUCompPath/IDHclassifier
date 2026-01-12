cd /home/shubham/ssl/CLAM_lunit/
for j in train val test
    do 
        python eval_miccai.py --k 10 --models_exp_code tcga_2021_$1_s1 --save_exp_code tcga_2021_$1_"$j" --task $2 --model_type clam_sb --results_dir results --data_root_dir /home/shubham/ssl/CLAM_lunit/ --split "$j" --features_dir features/features_$3_256 --csv_path dataset_csv/$4.csv
done
 
 
for j in 2.5x 5x 10x 20x; do  python eval_miccai.py --k 10 --models_exp_code tcga_2021_grade_"$j"_s1 --save_exp_code tcga_2021_grade_"$j"_test --task tcga_3_class --model_type clam_sb --results_dir results --data_root_dir /home/shubham/ssl/CLAM_lunit/ --split test --features_dir features_ebrains/features_"$j"_256 --csv_path dataset_csv/ebrain_df_label_grade.csv --splits_dir splits/ebrain_who_2021_100; done
 
 
for j in 2.5x 5x 10x 20x; do  python eval_miccai.py --k 10 --models_exp_code tcga_2021_who_"$j"_s1 --save_exp_code tcga_2021_who_"$j"_test --task tcga_who_2021 --model_type clam_sb --results_dir results --data_root_dir /home/shubham/ssl/CLAM_lunit/ --split test --features_dir features_ebrains/features_"$j"_256 --csv_path dataset_csv/ebrain_df_who.csv --splits_dir splits/ebrain_who_2021_100; done

for j in 2.5x 5x 10x 20x; do  python eval_miccai.py --k 10 --models_exp_code tcga_2021_idh_"$j"_s1 --save_exp_code tcga_2021_idh_"$j"_test --task tcga_2_class --model_type clam_sb --results_dir results --data_root_dir /home/shubham/ssl/CLAM_lunit/ --split test --features_dir features_ebrains/features_"$j"_256 --csv_path dataset_csv/ebrain_df_label_idh.csv --splits_dir splits/ebrain_who_2021_100; done

for j in 2.5x 5x 10x 20x; do  python eval_miccai.py --k 10 --models_exp_code tcga_2021_histology_"$j"_s1 --save_exp_code tcga_2021_histology_"$j"_test --task tcga_3_class --model_type clam_sb --results_dir results --data_root_dir /home/shubham/ssl/CLAM_lunit/ --split test --features_dir features_ebrains/features_"$j"_256 --csv_path dataset_csv/ebrain_df_label_histology.csv --splits_dir splits/ebrain_who_2021_100; done

