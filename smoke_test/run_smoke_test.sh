#!/bin/bash
# End-to-end smoke test on 24 TCGA slides (12 IDH-mutant / 12 wildtype, 24 patches each, 3 folds):
#   1. single-scale UNI 20x            : train + evaluate
#   2. multi-scale ImageNet 5x/10x/20x : train + evaluate each scale, then ensemble
# Runs in about 3 minutes on CPU. Outputs go to smoke_test/out/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."

ST=smoke_test
OUT=$ST/out
PY=${PYTHON:-python}
rm -rf "$OUT"

run_scale () {   # $1 = feature set, e.g. uni_20x
    local name=$1
    $PY main.py --early_stopping --max_epochs 15 --k 3 --seed 1 --lr 1e-4 \
        --exp_code "$name" --task task_idh_classifier --model_type clam_sb --subtyping --no_inst_cluster --weighted_sample \
        --csv_path $ST/labels.csv --features_dir $ST/features/$name --split_dir $ST/splits --results_dir $OUT/results > "$OUT/$name.train.log" 2>&1 \
        || { echo "training failed, see $OUT/$name.train.log"; exit 1; }
    $PY eval.py --k 3 --task task_idh_classifier --model_type clam_sb \
        --models_exp_code "${name}_s1" --save_exp_code "$name" --results_dir $OUT/results --eval_dir $OUT/eval \
        --split test --csv_path $ST/labels.csv --features_dir $ST/features/$name --splits_dir $ST/splits > "$OUT/$name.eval.log" 2>&1 \
        || { echo "evaluation failed, see $OUT/$name.eval.log"; exit 1; }
    echo "== $name"; cat $OUT/eval/EVAL_$name/summary.csv
}

mkdir -p "$OUT"
echo "### 1. single scale (UNI, 20x)"
run_scale uni_20x

echo; echo "### 2. multi-scale ensemble (ImageNet, 5x + 10x + 20x)"
for m in 5x 10x 20x; do run_scale imagenet_$m; done
$PY ensemble.py --eval_dirs $OUT/eval/EVAL_imagenet_5x $OUT/eval/EVAL_imagenet_10x $OUT/eval/EVAL_imagenet_20x \
    --names 5x 10x 20x --out_dir $OUT/eval/ENSEMBLE_imagenet

echo; echo "Smoke test finished: outputs in $OUT"
