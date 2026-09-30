# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Bootstrap uncertainty for the ACS employment prevalence errors.

Each replicate resamples, with replacement, the calibration set and the
in-distribution and held-out target pools; redraws the age-shifted targets
from the resampled pools; refits every estimator; and records the signed
prevalence error (estimate minus the true prevalence of that replicate's
target). The logistic-regression classifier is held fixed, so intervals
reflect sampling variability of the labeled calibration data and of the
targets, conditional on the classifier.

Duplicated calibration rows are collapsed into one row with a count weight
rather than repeated, because MCGrad selects its number of boosting rounds by
cross-validation and IPW cross-fits its propensity model: repeated rows would
fall into both training and validation folds, making extra rounds look better
than they are.

Replicate 0 uses the original samples and seeds and reproduces the point
estimates of run_acs.py.

Usage:
    python acs_analysis/bootstrap_acs.py [--n-boot 200] [--workers 4]
    (threads per worker: OMP_NUM_THREADS, default 4)
"""

import os

# Threads per process, set before LightGBM loads its OpenMP runtime; spawned
# workers re-run this line, so each worker is limited to the same count.
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mcgrad import methods as mcgrad_methods
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from helpers import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN, NUMERICAL_COLUMNS,
    create_logistic_pipeline, load_acs_employment_data,
    resample_with_age_shift,
    calibrate_threshold_prevalence_matching, estimate_classifier_error_rates,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'images')

TRAIN_STATES = ["TX", "MI", "PA", "OH", "IL", "GA", "NC", "VA"]
OOD_STATES = ["CA", "NY", "FL", "WA", "AZ", "CO"]
SURVEY_YEARS = ["2016", "2017", "2018"]
BASE = 'base_model_prediction'
ALL_FEATURE_COLS = NUMERICAL_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
CAT_SEG = CATEGORICAL_COLUMNS + BINARY_COLUMNS
NUM_SEG = NUMERICAL_COLUMNS
SHIFTS = {'Original': 'original', 'Young-skewed': 'young',
          'Old-skewed': 'old', 'Bimodal': 'bimodal'}
METHODS = ['Raw', 'CC', 'RG', 'PACC', 'SLD', 'IPW', 'Iso', 'MCGrad']


def prepare():
    """Load data, split and score with the fixed classifier, as in run_acs.py."""
    train_source_df = load_acs_employment_data(
        states=TRAIN_STATES, survey_years=SURVEY_YEARS,
        include_state=True, include_year=True)
    ood_df = load_acs_employment_data(
        states=OOD_STATES, survey_years=SURVEY_YEARS,
        include_state=True, include_year=True)
    train_df, test_df = train_test_split(
        train_source_df, test_size=0.30, random_state=42,
        stratify=train_source_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1))
    model_train_df, calibration_df = train_test_split(
        train_df, test_size=0.30, random_state=42,
        stratify=train_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1))
    test_df, calibration_df, ood_df = test_df.copy(), calibration_df.copy(), ood_df.copy()

    feature_cols = [c for c in model_train_df.columns if c not in [LABEL_COLUMN, "STATE", "YEAR"]]
    pipeline = create_logistic_pipeline()
    pipeline.fit(model_train_df[feature_cols], model_train_df[LABEL_COLUMN].astype(int).to_numpy())
    for df in [calibration_df, test_df, ood_df]:
        df[BASE] = pipeline.predict_proba(df[feature_cols])[:, 1]
    keep = ALL_FEATURE_COLS + [LABEL_COLUMN, BASE]
    calibration_df = calibration_df[keep].reset_index(drop=True)
    calibration_df['_rid'] = np.arange(len(calibration_df))
    calibration_df['w'] = 1.0
    return calibration_df, test_df[keep], ood_df[keep]


def collapse(df):
    """One row per distinct calibration row ('_rid'), with count weight 'w'."""
    counts = df['_rid'].value_counts()
    out = df.drop_duplicates('_rid').set_index('_rid')
    out['w'] = counts.reindex(out.index).astype(float)
    return out.reset_index()


def ipw_estimate(cal_df, target_df, n_jobs):
    X = pd.concat([cal_df[ALL_FEATURE_COLS], target_df[ALL_FEATURE_COLS]], ignore_index=True)
    for c in CAT_SEG:
        X[c] = X[c].astype(str).astype('category')
    z = np.r_[np.zeros(len(cal_df)), np.ones(len(target_df))]
    sw = np.r_[cal_df['w'].values, np.ones(len(target_df))]
    p = np.zeros(len(X))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in folds.split(X, z):
        clf = LGBMClassifier(random_state=42, verbose=-1, n_jobs=n_jobs)
        clf.fit(X.iloc[train_idx], z[train_idx], sample_weight=sw[train_idx])
        p[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
    p = p[:len(cal_df)]
    w = cal_df['w'].values * p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal_df[LABEL_COLUMN].values, weights=w), 0, 1)


def sld_estimate(scores, source_prevalence, max_iter=1000, tol=1e-8):
    p_hat = source_prevalence
    for _ in range(max_iter):
        rp = p_hat / source_prevalence
        rn = (1 - p_hat) / (1 - source_prevalence)
        p_new = ((rp * scores) / (rp * scores + rn * (1 - scores))).mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


def estimate(cal_df, test_df, ood_df, rs, n_jobs):
    """Fit all estimators on cal_df (count weights in 'w'); return signed errors (pp)."""
    w = cal_df['w'].to_numpy()
    # Unit weights (replicate 0) fit unweighted, reproducing the published point estimates.
    wcol = None if (w == 1).all() else 'w'
    iso = mcgrad_methods.IsotonicRegression().fit(cal_df, BASE, LABEL_COLUMN,
                                                  weight_column_name=wcol)
    mc = mcgrad_methods.MCGrad().fit(
        cal_df, BASE, LABEL_COLUMN,
        categorical_feature_column_names=CAT_SEG, numerical_feature_column_names=NUM_SEG,
        weight_column_name=wcol)
    # Threshold and error rates on the expanded sample (exact bootstrap equivalents).
    reps = w.astype(int)
    y_exp = pd.Series(np.repeat(cal_df[LABEL_COLUMN].to_numpy(), reps))
    s_exp = pd.Series(np.repeat(cal_df[BASE].to_numpy(), reps))
    thr = calibrate_threshold_prevalence_matching(y_exp, s_exp)
    tpr, fpr = estimate_classifier_error_rates(y_exp, s_exp, thr)
    src_prev = np.average(cal_df[LABEL_COLUMN], weights=w)
    y_cal = cal_df[LABEL_COLUMN].astype(bool).to_numpy()
    s_cal = cal_df[BASE].to_numpy()
    pos_mean = np.average(s_cal[y_cal], weights=w[y_cal])
    neg_mean = np.average(s_cal[~y_cal], weights=w[~y_cal])

    out = {}
    for setting, pool in [('In-Dist', test_df), ('OOD', ood_df)]:
        for name, shift in SHIFTS.items():
            t = resample_with_age_shift(pool, shift=shift, random_state=rs)
            tp = t[LABEL_COLUMN].mean()
            iso_pred = np.asarray(iso.predict(t, BASE))
            mc_pred = mc.predict(t, BASE, categorical_feature_column_names=CAT_SEG,
                                 numerical_feature_column_names=NUM_SEG)
            ap = (t[BASE] >= thr).mean()
            d = tpr - fpr
            rg = np.clip((ap - fpr) / d, 0, 1) if abs(d) > 1e-10 else ap
            raw = t[BASE].mean()
            dp = pos_mean - neg_mean
            pacc = np.clip((raw - neg_mean) / dp, 0, 1) if abs(dp) > 1e-10 else raw
            est = {
                'Raw': raw, 'CC': ap, 'RG': rg, 'PACC': pacc,
                'SLD': sld_estimate(iso_pred, src_prev),
                'IPW': ipw_estimate(cal_df, t, n_jobs),
                'Iso': iso_pred.mean(), 'MCGrad': np.mean(mc_pred),
            }
            out[f'{setting}|{name}'] = {'true': tp * 100,
                                        **{m: (v - tp) * 100 for m, v in est.items()}}
    return out


_FRAMES, _N_JOBS = None, None


def _init_worker(frames, n_jobs):
    global _FRAMES, _N_JOBS
    warnings.filterwarnings('ignore')
    _FRAMES, _N_JOBS = frames, n_jobs


def _boot(df, rng):
    return df.sample(n=len(df), replace=True, random_state=rng.integers(2**31)).reset_index(drop=True)


def _replicate(seed):
    rng = np.random.default_rng(seed)
    cal_df, test_df, ood_df = _FRAMES
    return estimate(collapse(_boot(cal_df, rng)), _boot(test_df, rng), _boot(ood_df, rng),
                    rs=int(rng.integers(2**31)), n_jobs=_N_JOBS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=int(os.environ['OMP_NUM_THREADS']))
    args = parser.parse_args()

    t0 = time.time()
    frames = prepare()
    print(f"Cal: {len(frames[0]):,} | Test: {len(frames[1]):,} | OOD: {len(frames[2]):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    point = estimate(*frames, rs=42, n_jobs=args.workers * args.threads)
    print(f"Replicate 0 (point estimates) in {time.time() - t1:.0f}s", flush=True)
    for k, v in point.items():
        print(f"  {k:22s} " + ' '.join(f"{m}={v[m]:+.2f}" for m in METHODS), flush=True)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(2**31, size=args.n_boot)
    path = os.path.join(IMG_DIR, 'acs_bootstrap.json')
    reps = []
    save(path, args, point, reps)
    if args.n_boot:
        with Pool(args.workers, initializer=_init_worker, initargs=(frames, args.threads)) as pool:
            for rep in pool.imap(_replicate, seeds):
                reps.append(rep)
                # Checkpoint after every replicate so a partial run is usable.
                save(path, args, point, reps)
                print(f"  {len(reps)}/{args.n_boot} replicates ({time.time() - t0:.0f}s)", flush=True)
    print(f"Saved {path}")


def save(path, args, point, reps):
    summary = {}
    for k in point:
        summary[k] = {'true': point[k]['true']}
        for m in METHODS:
            draws = np.array([r[k][m] for r in reps])
            summary[k][m] = {
                'point': point[k][m],
                'se': float(np.std(draws, ddof=1)) if len(reps) > 1 else None,
                'ci_low': float(np.percentile(draws, 2.5)) if len(reps) else None,
                'ci_high': float(np.percentile(draws, 97.5)) if len(reps) else None,
            }
    with open(path, 'w') as f:
        json.dump({'n_boot': args.n_boot, 'n_done': len(reps), 'seed': args.seed,
                   'summary': summary, 'replicates': reps}, f, indent=1)


if __name__ == '__main__':
    main()
