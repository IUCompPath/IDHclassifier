"""Build the tiny smoke-test dataset from full TCGA feature folders (maintainer tool, not needed to run the smoke test).

Picks N_PER_CLASS slides per class (one slide per patient, fixed seed), keeps N_PATCHES random patches
of every bag (stored as float16), and writes labels + 3-fold patient-disjoint splits.

    python smoke_test/make_smoke_data.py \
        --uni_dir  /path/features_uni/tcga_gbm_lgg_20x \
        --imagenet_fmt '/path/features_all/imagenet_gbm_lgg_{mag}'
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

N_PER_CLASS, N_PATCHES, K, SEED = 12, 24, 3, 0
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uni_dir', required=True, help='folder containing pt_files/ (UNI, 20x)')
    ap.add_argument('--imagenet_fmt', required=True, help="folder pattern with '{mag}', e.g. .../imagenet_gbm_lgg_{mag}")
    ap.add_argument('--mags', nargs='+', default=['5x', '10x', '20x'])
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(HERE, '..', 'dataset_csv', 'df_idh_label.csv'))
    sources = {'uni_20x': args.uni_dir, **{f'imagenet_{m}': args.imagenet_fmt.format(mag=m) for m in args.mags}}
    have = set.intersection(*[{f[:-3] for f in os.listdir(os.path.join(d, 'pt_files'))} for d in sources.values()])
    labels = labels[labels.slide_id.isin(have)].drop_duplicates('case_id')  # one slide per patient

    rng = np.random.RandomState(SEED)
    picked = pd.concat([labels[labels.label == c].sample(N_PER_CLASS, random_state=SEED) for c in (0, 1)])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked.to_csv(os.path.join(HERE, 'labels.csv'), index=False)

    for name, d in sources.items():
        out = os.path.join(HERE, 'features', name, 'pt_files')
        os.makedirs(out, exist_ok=True)
        for sid in picked.slide_id:
            x = torch.load(os.path.join(d, 'pt_files', sid + '.pt'), map_location='cpu')
            idx = np.sort(rng.choice(len(x), min(N_PATCHES, len(x)), replace=False))
            torch.save(x[idx].half().contiguous(), os.path.join(out, sid + '.pt'))

    # patient-disjoint, class-balanced folds: test = one third, val = 2 slides/class from the rest
    sp = os.path.join(HERE, 'splits')
    os.makedirs(sp, exist_ok=True)
    parts = {c: np.array_split(picked[picked.label == c].slide_id.values, K) for c in (0, 1)}
    for k in range(K):
        test = [s for c in (0, 1) for s in parts[c][k]]
        rest = {c: [s for j in range(K) if j != k for s in parts[c][j]] for c in (0, 1)}
        val = [s for c in (0, 1) for s in rest[c][:2]]
        train = [s for c in (0, 1) for s in rest[c][2:]]
        pd.DataFrame({'train': pd.Series(train), 'val': pd.Series(val), 'test': pd.Series(test)}).to_csv(
            os.path.join(sp, f'splits_{k}.csv'))
    print('picked', len(picked), 'slides;', picked.label.value_counts().to_dict())


if __name__ == '__main__':
    main()
