#!/bin/bash
# End-to-end smoke test (training + evaluation) on 24 TCGA slides with UNI 20x features
# (12 IDH-mutant / 12 wildtype, 24 patches each, 3 folds). Outputs go to smoke_test/out/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."

ST=smoke_test
OUT=$ST/out
PY=${PYTHON:-python}
rm -rf "$OUT"; mkdir -p "$OUT"

$PY main.py --early_stopping --max_epochs 15 --k 3 --seed 1 --lr 1e-4 \
    --exp_code smoke --task task_idh_classifier --model_type clam_sb --subtyping --no_inst_cluster --weighted_sample \
    --csv_path $ST/labels.csv --features_dir $ST/features/uni_20x --split_dir $ST/splits --results_dir $OUT/results

$PY eval.py --k 3 --task task_idh_classifier --model_type clam_sb \
    --models_exp_code smoke_s1 --save_exp_code smoke --results_dir $OUT/results --eval_dir $OUT/eval \
    --split test --csv_path $ST/labels.csv --features_dir $ST/features/uni_20x --splits_dir $ST/splits

echo; echo "Smoke test finished: outputs in $OUT"; cat $OUT/eval/EVAL_smoke/summary.csv
