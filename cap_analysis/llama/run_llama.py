"""
Llama 3.3 70B replication script (SI Appendix S2).

Reproduces the CAP Law & Crime prevalence estimation analysis using
Llama 3.3 70B Instruct verbalized confidence scores (2-stage with nudge).
Generates SI Appendix Table S3.

Usage:
    conda run -n mcgrad_tutorials python3 cap_analysis/llama/run_llama.py

Requires:
    - cap_analysis/data/full_sample.csv (105K documents)
    - cap_analysis/data/inference_output/llama-70b-verbalized-2stage/full_codebook.csv
"""
import logging
import os
import warnings

# Silence FutureWarnings from sklearn/pandas/mcgrad — they don't affect numerical
# correctness here.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

np.random.seed(42)

# Resolve paths relative to this file so the script runs from any cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

LABEL = 'law_crime'
SUBPOP_MAP = {
    ('Denmark', 'parliamentary_question'): 'denmark_questions',
    ('Spain', 'parliamentary_question'): 'spain_questions',
    ('Spain', 'media'): 'spain_media',
    ('United States', 'bill'): 'us_bills',
    ('Belgium', 'tv_news'): 'belgium_tv',
    ('Belgium', 'newspaper'): 'belgium_newspaper',
}
CAL_SUBPOPS = ['denmark_questions', 'spain_questions', 'us_bills', 'belgium_newspaper']
CAT_FEATS = ['doc_type', 'country', 'party']
NUM_FEATS = ['decade']
IPW_FEATS = ['country', 'doc_type', 'decade']
N_TARGET = 20_000

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
data_df = pd.read_csv(os.path.join(DATA_DIR, 'full_sample.csv'))
scores_df = pd.read_csv(os.path.join(
    DATA_DIR, 'inference_output/llama-70b-verbalized-2stage/full_codebook.csv'
))
data_df['llm_score'] = scores_df['score'].astype(float)

subpop_keys = pd.Series(list(zip(data_df['country'], data_df['doc_type'])))
data_df['subpop'] = subpop_keys.map(SUBPOP_MAP).fillna('unknown').values
data_df['decade'] = (data_df['year'] // 10) * 10
data_df['party'] = data_df['party'].fillna('unknown')

subpops = {key: data_df[data_df['subpop'] == key].copy() for key in SUBPOP_MAP.values()}

print(f"  Loaded {len(data_df):,} documents")
print(f"  Score range: [{data_df['llm_score'].min():.2f}, {data_df['llm_score'].max():.2f}]")
print(f"  Unique scores: {data_df['llm_score'].nunique()}")

# ============================================================
# 2. AUC by sub-population
# ============================================================
print("\nAUC by sub-population:")
for key, df in subpops.items():
    auc = roc_auc_score(df[LABEL], df['llm_score'])
    print(f"  {key:<25} AUC={auc:.3f}  N={len(df):,}  prev={df[LABEL].mean():.1%}")

# ============================================================
# 3. Calibration / test split
# ============================================================
cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = subpops[key]
    cal_frac = min(10_000 / len(sub), 0.67)
    cal, test = train_test_split(
        sub, test_size=1 - cal_frac, random_state=42, stratify=sub[LABEL],
    )
    cal_parts.append(cal)
    test_parts.append(test)
cal_df = pd.concat(cal_parts, ignore_index=True).copy()
test_df = pd.concat(test_parts, ignore_index=True).copy()

print(f"\nCalibration: {len(cal_df):,}, Test: {len(test_df):,}")
print(f"Cal prevalence: {cal_df[LABEL].mean():.3f}")

# ============================================================
# 4. Fit calibration methods
# ============================================================
print("\nFitting models...")
isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'llm_score', LABEL)
mcgrad = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad = mcgrad.fit(
    cal_df, 'llm_score', LABEL,
    categorical_feature_column_names=CAT_FEATS,
    numerical_feature_column_names=NUM_FEATS,
)

ood_spain = subpops['spain_media']
ood_belgium = subpops['belgium_tv']

for df in [test_df, ood_spain, ood_belgium]:
    df['iso_pred'] = isotonic.predict(df, 'llm_score')
    df['mc_pred'] = mcgrad.predict(
        df, 'llm_score',
        categorical_feature_column_names=CAT_FEATS,
        numerical_feature_column_names=NUM_FEATS,
    )

# ============================================================
# 5. Calibration parameters for CC, RG, SLD, IPW
# ============================================================
fpr_arr, tpr_arr, thresholds = roc_curve(cal_df[LABEL], cal_df['llm_score'])
THRESHOLD = float(thresholds[np.argmax(tpr_arr - fpr_arr)])
binary_cal = (cal_df['llm_score'] >= THRESHOLD).astype(int)
labels_cal = cal_df[LABEL].astype(int)
n_pos = (labels_cal == 1).sum()
n_neg = (labels_cal == 0).sum()
cal_tpr = ((binary_cal == 1) & (labels_cal == 1)).sum() / n_pos
cal_fpr = ((binary_cal == 1) & (labels_cal == 0)).sum() / n_neg
src_prev = cal_df[LABEL].mean()


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (EMQ) prevalence estimator."""
    p_hat = source_prevalence
    for _ in range(max_iter):
        ratio_pos = p_hat / source_prevalence
        ratio_neg = (1 - p_hat) / (1 - source_prevalence)
        adjusted = (ratio_pos * scores) / (
            ratio_pos * scores + ratio_neg * (1 - scores)
        )
        p_new = adjusted.mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


def ipw_estimate(cal, target):
    """Inverse-probability-weighted prevalence estimate."""
    combined = pd.concat([
        cal[IPW_FEATS].assign(_t=0),
        target[IPW_FEATS].assign(_t=1),
    ], ignore_index=True)
    X = pd.get_dummies(combined[IPW_FEATS], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    weights = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal[LABEL].values, weights=weights), 0, 1)


# ============================================================
# 6. Compute prevalence bias across scenarios
# ============================================================
def resample_country(df, random_state=42):
    weights = np.where(df['country'] == 'Belgium', 5.0, 1.0)
    weights = weights / weights.sum()
    return df.sample(
        n=min(N_TARGET, len(df)), weights=weights, replace=True,
        random_state=random_state,
    )


def resample_doctype(df, random_state=42):
    weights = np.where(df['doc_type'] == 'bill', 5.0, 1.0)
    weights = weights / weights.sum()
    return df.sample(
        n=min(N_TARGET, len(df)), weights=weights, replace=True,
        random_state=random_state,
    )


scenarios = {
    'Baseline':       test_df.sample(n=min(N_TARGET, len(test_df)), replace=True, random_state=42),
    'Country shift':  resample_country(test_df),
    'Doc-type shift': resample_doctype(test_df),
    'Spain media':    ood_spain.sample(n=min(N_TARGET, len(ood_spain)), replace=True, random_state=42),
    'Belgium TV':     ood_belgium.sample(n=min(N_TARGET, len(ood_belgium)), replace=True, random_state=42),
}

print(
    f"\n{'Scenario':<18} {'True':>6}  "
    f"{'CC':>7} {'RG':>7} {'SLD':>7} {'IPW':>7} {'Iso':>7} {'MCGrad':>7}"
)
print("-" * 75)

for name, target in scenarios.items():
    true_prev = target[LABEL].mean()
    cc = (target['llm_score'] >= THRESHOLD).mean()
    denom = cal_tpr - cal_fpr
    if abs(denom) > 1e-10:
        rg = float(np.clip((cc - cal_fpr) / denom, 0, 1))
    else:
        rg = float(cc)
    sld = sld_estimate(target['llm_score'].values, src_prev)
    ipw = ipw_estimate(cal_df, target)
    iso = target['iso_pred'].mean()
    mc = target['mc_pred'].mean()

    def fmt_bias(estimate):
        return f"{(estimate - true_prev) * 100:+.1f}"

    print(
        f"{name:<18} {true_prev:>5.1%}  "
        f"{fmt_bias(cc):>7} {fmt_bias(rg):>7} {fmt_bias(sld):>7} "
        f"{fmt_bias(ipw):>7} {fmt_bias(iso):>7} {fmt_bias(mc):>7}"
    )

print("\nCC = Classify & Count, RG = Rogan-Gladen, SLD = Saerens-Latinne-Decaestecker")
print("IPW = Importance-weighted, Iso = Isotonic regression")
print(f"\nThreshold (Youden's J): {THRESHOLD:.4f}")
print("Done.")
