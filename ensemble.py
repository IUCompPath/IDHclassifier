"""Multi-scale (late-fusion) ensemble.

Averages the slide-level class probabilities written by eval.py for several magnifications / feature sets and
reports per-fold AUC and accuracy of the fused prediction next to each single-scale model.

    python ensemble.py --eval_dirs eval_results/EVAL_idh_5x_uni eval_results/EVAL_idh_10x_uni eval_results/EVAL_idh_20x_uni \
        --names 5x 10x 20x --out_dir eval_results/ENSEMBLE_idh_uni

Every eval dir must contain fold_<k>.csv (slide_id, Y, Y_hat, p_0, p_1). Slides missing at a scale are fused over the
scales that have them.
"""
import argparse
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def read_folds(eval_dir):
    folds = {}
    for f in os.listdir(eval_dir):
        m = re.fullmatch(r'fold_(\d+)\.csv', f)
        if m:
            folds[int(m.group(1))] = pd.read_csv(os.path.join(eval_dir, f))
    if not folds:
        raise FileNotFoundError('no fold_<k>.csv in {}'.format(eval_dir))
    return folds


def fuse(dfs):
    df = pd.concat(dfs, ignore_index=True)
    out = df.groupby('slide_id', as_index=False).agg({'Y': 'first', 'p_0': 'mean', 'p_1': 'mean'})
    out['Y_hat'] = (out['p_1'] > out['p_0']).astype(int)
    return out[['slide_id', 'Y', 'Y_hat', 'p_0', 'p_1']]


def metrics(df):
    auc = roc_auc_score(df['Y'], df['p_1']) if df['Y'].nunique() == 2 else float('nan')
    return auc, accuracy_score(df['Y'], df['Y_hat'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval_dirs', nargs='+', required=True, help='eval.py output folders, one per scale (>= 2)')
    ap.add_argument('--names', nargs='+', default=None, help='label for each eval dir (default: folder names)')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    if len(args.eval_dirs) < 2:
        ap.error('need at least two eval dirs to ensemble')
    names = args.names or [os.path.basename(os.path.normpath(d)) for d in args.eval_dirs]
    assert len(names) == len(args.eval_dirs), '--names must match --eval_dirs'

    per_scale = [read_folds(d) for d in args.eval_dirs]
    folds = sorted(set.intersection(*[set(p) for p in per_scale]))
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for k in folds:
        for n, p in zip(names, per_scale):
            auc, acc = metrics(p[k])
            rows.append({'fold': k, 'model': n, 'auc': auc, 'acc': acc})
        fused = fuse([p[k] for p in per_scale])
        fused.to_csv(os.path.join(args.out_dir, 'fold_{}.csv'.format(k)), index=False)
        auc, acc = metrics(fused)
        rows.append({'fold': k, 'model': 'ensemble(' + '+'.join(names) + ')', 'auc': auc, 'acc': acc})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, 'fold_metrics.csv'), index=False)
    mean = summary.groupby('model', sort=False)[['auc', 'acc']].agg(['mean', 'std']).round(4)
    mean.to_csv(os.path.join(args.out_dir, 'summary.csv'))
    print(mean.to_string())


if __name__ == '__main__':
    main()
