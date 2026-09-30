# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Bootstrap uncertainty for the CAP (Claude Opus 4.6) prevalence errors.

Each replicate resamples, with replacement, the calibration set (within each
calibration sub-population; duplicated rows are collapsed into one row with a
count weight, see below), the in-distribution test pool (within each
sub-population), and the two out-of-distribution targets; redraws the
within-support scenarios from the resampled test pool; refits every
estimator; and records the signed prevalence error (estimate minus the true
prevalence of that replicate's target). The LLM outputs are held fixed, so
intervals reflect sampling variability of the labeled calibration data and of
the targets, conditional on the measurement device.

Calibration duplicates are passed as weights rather than repeated rows because
MCGrad selects its number of boosting rounds by cross-validation and IPW
cross-fits its propensity model: repeated rows would fall into both training
and validation folds, making extra rounds look better than they are.

Replicate 0 uses the original samples and seeds and reproduces the point
estimates of run_cap_opus.py.

Usage:
    python cap_analysis/opus/bootstrap_cap_opus.py [--n-boot 500] [--workers 14]
"""

import os

# One thread per worker process; replicates run in parallel instead.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mcgrad import methods as mcgrad_methods
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
IMG_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'paper', 'images')

CAL_SUBPOPS = [
    'Denmark_parliamentary_question',
    'Spain_parliamentary_question',
    'United States_bill',
    'Belgium_newspaper',
]
CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']
N = 5000
SCENARIOS = ['Baseline', 'Country shift', 'Doc-type shift', 'Spain media', 'Belgium TV']
METHODS = ['CC', 'RG', 'SLD', 'IPW', 'Iso', 'MC (binary)', 'MC (scores)']


def load_data():
    sample = pd.read_csv(os.path.join(DATA_DIR, 'opus_30k_sample.csv'))
    sample['party'] = sample['party'].fillna('unknown')
    sample['text_len'] = sample['text'].str.len()
    binary = pd.read_csv(
        os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-binary/merged.csv')
    ).drop_duplicates(subset='id', keep='first')
    binary['llm_yes'] = (binary['answer'].str.lower() == 'yes').astype(int)
    pyn = pd.read_csv(os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-pyn/merged.csv'))
    data = sample.merge(binary[['id', 'llm_yes']], on='id', how='left')
    data = data.merge(pyn[['id', 'score']], on='id', how='left')
    data.rename(columns={'score': 'pyn_score'}, inplace=True)
    data = data.dropna(subset=['llm_yes'])
    data['llm_yes'] = data['llm_yes'].astype(int)
    data['llm_label'] = data['llm_yes'].map({1: 'yes', 0: 'no'})
    return data


def split(data):
    cal_parts, test_parts = [], []
    for key in CAL_SUBPOPS:
        sub = data[data['subpop'] == key].copy()
        cal, test = train_test_split(sub, test_size=0.33, random_state=42, stratify=sub['law_crime'])
        cal_parts.append(cal)
        test_parts.append(test)
    return (pd.concat(cal_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True),
            data[data['subpop'] == 'Spain_media'].copy(),
            data[data['subpop'] == 'Belgium_tv_news'].copy())


def resample_within(df, rng, by='subpop'):
    parts = [g.sample(n=len(g), replace=True, random_state=rng.integers(2**31))
             for _, g in df.groupby(by)]
    return pd.concat(parts, ignore_index=True)


def collapse(df):
    """One row per distinct calibration row ('_rid'), with count weight 'w'."""
    counts = df['_rid'].value_counts()
    out = df.drop_duplicates('_rid').set_index('_rid')
    out['w'] = counts.reindex(out.index).astype(float)
    return out.reset_index()


def ipw_estimate(cal, target):
    features = CAT_PYN + NUM
    X = pd.concat([cal[features], target[features]], ignore_index=True)
    for c in CAT_PYN:
        X[c] = X[c].astype(str).astype('category')
    z = np.r_[np.zeros(len(cal)), np.ones(len(target))]
    sw = np.r_[cal['w'].values, np.ones(len(target))]
    p = np.zeros(len(X))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in folds.split(X, z):
        clf = LGBMClassifier(random_state=42, verbose=-1)
        clf.fit(X.iloc[train_idx], z[train_idx], sample_weight=sw[train_idx])
        p[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
    p = p[:len(cal)]
    w = cal['w'].values * p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal['law_crime'].values, weights=w), 0, 1)


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    p_hat = source_prevalence
    for _ in range(max_iter):
        rp = p_hat / source_prevalence
        rn = (1 - p_hat) / (1 - source_prevalence)
        p_new = ((rp * scores) / (rp * scores + rn * (1 - scores))).mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


def resample_weighted(df, col, target, rs, factor=5.0):
    w = np.where(df[col] == target, factor, 1.0)
    w = w / w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)


def estimate(cal_df, test_df, ood_spain, ood_belgium, rs):
    """Fit all estimators on cal_df (count weights in 'w'); return signed errors (pp)."""
    cal_df = cal_df.copy()
    w = cal_df['w'].values
    # Unit weights (replicate 0) fit unweighted, reproducing the published point estimates.
    wcol = None if (w == 1).all() else 'w'
    base_rate = np.average(cal_df['law_crime'], weights=w)
    cal_df['binary_init'] = base_rate

    mc_bin = mcgrad_methods.MCGrad().fit(
        cal_df, 'binary_init', 'law_crime',
        categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM,
        weight_column_name=wcol)
    mc_pyn = mcgrad_methods.MCGrad().fit(
        cal_df, 'pyn_score', 'law_crime',
        categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM,
        weight_column_name=wcol)
    iso = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime',
                                                  weight_column_name=wcol)

    y, yhat = cal_df['law_crime'].astype(int).values, cal_df['llm_yes'].astype(int).values
    tpr = (w * yhat * y).sum() / (w * y).sum()
    fpr = (w * yhat * (1 - y)).sum() / (w * (1 - y)).sum()

    targets = {
        'Baseline': test_df.sample(n=min(N, len(test_df)), replace=True, random_state=rs),
        'Country shift': resample_weighted(test_df, 'country', 'Belgium', rs),
        'Doc-type shift': resample_weighted(test_df, 'doc_type', 'bill', rs),
        'Spain media': ood_spain,
        'Belgium TV': ood_belgium,
    }

    out = {}
    for name, t in targets.items():
        t = t.copy()
        t['binary_init'] = base_rate
        mcb = mc_bin.predict(t, 'binary_init', categorical_feature_column_names=CAT_BIN,
                             numerical_feature_column_names=NUM)
        mcp = mc_pyn.predict(t, 'pyn_score', categorical_feature_column_names=CAT_PYN,
                             numerical_feature_column_names=NUM)
        iso_pred = iso.predict(t, 'pyn_score')
        tp = t['law_crime'].mean()
        cc = t['llm_yes'].mean()
        d = tpr - fpr
        rg = np.clip((cc - fpr) / d, 0, 1) if abs(d) > 1e-10 else cc
        est = {
            'CC': cc, 'RG': rg,
            'SLD': sld_estimate(np.asarray(iso_pred), base_rate),
            'IPW': ipw_estimate(cal_df, t),
            'Iso': np.mean(iso_pred),
            'MC (binary)': np.mean(mcb), 'MC (scores)': np.mean(mcp),
        }
        out[name] = {'true': tp * 100, **{m: (v - tp) * 100 for m, v in est.items()}}
    return out


_FRAMES = None


def _init_worker(*frames):
    global _FRAMES
    _FRAMES = frames
    warnings.filterwarnings('ignore')


def _replicate(seed):
    rng = np.random.default_rng(seed)
    cal_df, test_df, ood_spain, ood_belgium = _FRAMES
    return estimate(
        collapse(resample_within(cal_df, rng)), resample_within(test_df, rng),
        resample_within(ood_spain, rng), resample_within(ood_belgium, rng),
        rs=int(rng.integers(2**31)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()

    data = load_data()
    cal_df, test_df, ood_spain, ood_belgium = split(data)
    cal_df['_rid'] = np.arange(len(cal_df))
    cal_df['w'] = 1.0
    print(f"Cal: {len(cal_df):,} | Test: {len(test_df):,} | "
          f"OOD Spain: {len(ood_spain):,} | OOD Belgium: {len(ood_belgium):,}")

    t0 = time.time()
    point = estimate(cal_df, test_df, ood_spain, ood_belgium, rs=42)
    print(f"Replicate 0 (point estimates) in {time.time() - t0:.1f}s")
    for s in SCENARIOS:
        print(f"  {s:15s} " + ' '.join(f"{m}={point[s][m]:+.2f}" for m in METHODS))

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(2**31, size=args.n_boot)
    path = os.path.join(IMG_DIR, 'cap_bootstrap.json')
    reps = []
    save(path, args, point, reps)
    with Pool(args.workers, initializer=_init_worker,
              initargs=(cal_df, test_df, ood_spain, ood_belgium)) as pool:
        for rep in pool.imap(_replicate, seeds):
            reps.append(rep)
            # Checkpoint after every replicate so a partial run is usable.
            save(path, args, point, reps)
            if len(reps) % 25 == 0:
                print(f"  {len(reps)}/{args.n_boot} replicates ({time.time() - t0:.0f}s)", flush=True)

    summary = save(path, args, point, reps)
    print(f"Saved {path}")
    for s in SCENARIOS:
        print(f"  {s:15s} " + ' '.join(
            f"{m}={summary[s][m]['point']:+.1f}[{summary[s][m]['ci_low']:+.1f},{summary[s][m]['ci_high']:+.1f}]"
            for m in METHODS))


def save(path, args, point, reps):
    summary = {}
    for s in SCENARIOS:
        summary[s] = {}
        for m in METHODS:
            draws = np.array([r[s][m] for r in reps])
            summary[s][m] = {
                'point': point[s][m],
                'se': float(draws.std(ddof=1)) if len(reps) > 1 else None,
                'ci_low': float(np.percentile(draws, 2.5)) if len(reps) else None,
                'ci_high': float(np.percentile(draws, 97.5)) if len(reps) else None,
            }
        summary[s]['true'] = point[s]['true']
    with open(path, 'w') as f:
        json.dump({'n_boot': args.n_boot, 'n_done': len(reps), 'seed': args.seed,
                   'summary': summary, 'replicates': reps}, f, indent=1)
    return summary

if __name__ == '__main__':
    main()
