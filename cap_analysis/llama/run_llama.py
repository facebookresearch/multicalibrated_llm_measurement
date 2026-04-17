"""
Llama 3.3 70B replication script (SI Appendix S2).

Reproduces the CAP Law & Crime prevalence estimation analysis using
Llama 3.3 70B Instruct verbalized confidence scores (2-stage with nudge).
Generates SI Appendix Table S3.

Usage:
    cd cap_analysis && conda run -n mcgrad_tutorials python3 llama/run_llama.py

Requires:
    - data/full_sample.csv (105K documents)
    - data/inference_output/llama-70b-verbalized-2stage/full_codebook.csv
"""
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
data_df = pd.read_csv('data/full_sample.csv')
scores_df = pd.read_csv('data/inference_output/llama-70b-verbalized-2stage/full_codebook.csv')
data_df['llm_score'] = scores_df['score'].astype(float)

LABEL = 'law_crime'

SUBPOP_MAP = {
    ('Denmark', 'parliamentary_question'): 'denmark_questions',
    ('Spain', 'parliamentary_question'): 'spain_questions',
    ('Spain', 'media'): 'spain_media',
    ('United States', 'bill'): 'us_bills',
    ('Belgium', 'tv_news'): 'belgium_tv',
    ('Belgium', 'newspaper'): 'belgium_newspaper',
}
data_df['subpop'] = data_df.apply(
    lambda r: SUBPOP_MAP.get((r['country'], r['doc_type']), 'unknown'), axis=1
)
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
# 3. Calibration split
# ============================================================
CAL_SUBPOPS = ['denmark_questions', 'spain_questions', 'us_bills', 'belgium_newspaper']
cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = subpops[key]
    cal_frac = min(10_000 / len(sub), 0.67)
    cal, test = train_test_split(sub, test_size=1-cal_frac, random_state=42, stratify=sub[LABEL])
    cal_parts.append(cal)
    test_parts.append(test)
cal_df = pd.concat(cal_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

print(f"\nCalibration: {len(cal_df):,}, Test: {len(test_df):,}")
print(f"Cal prevalence: {cal_df[LABEL].mean():.3f}")

# ============================================================
# 4. Fit calibration methods
# ============================================================
print("\nFitting models...")
CAT_FEATS = ['doc_type', 'country', 'party']
NUM_FEATS = ['decade']

isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'llm_score', LABEL)
mcgrad = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad = mcgrad.fit(cal_df, 'llm_score', LABEL,
    categorical_feature_column_names=CAT_FEATS,
    numerical_feature_column_names=NUM_FEATS)

ood_spain = subpops['spain_media']
ood_belgium = subpops['belgium_tv']

for df in [test_df, ood_spain, ood_belgium]:
    df['iso_pred'] = isotonic.predict(df, 'llm_score')
    df['mc_pred'] = mcgrad.predict(df, 'llm_score',
        categorical_feature_column_names=CAT_FEATS,
        numerical_feature_column_names=NUM_FEATS)

# ============================================================
# 5. Calibration parameters for CC, RG, SLD, IPW
# ============================================================
fpr_arr, tpr_arr, thresholds = roc_curve(cal_df[LABEL], cal_df['llm_score'])
THRESHOLD = float(thresholds[np.argmax(tpr_arr - fpr_arr)])
bc = (cal_df['llm_score'] >= THRESHOLD).astype(int)
cl = cal_df[LABEL].astype(int)
cal_tpr = ((bc == 1) & (cl == 1)).sum() / cl.sum()
cal_fpr = ((bc == 1) & (cl == 0)).sum() / (1 - cl).sum()
src_prev = cal_df[LABEL].mean()

def sld_estimate(scores, sp=src_prev):
    p = sp
    for _ in range(100):
        rp = p / sp; rn = (1 - p) / (1 - sp)
        adj = (rp * scores) / (rp * scores + rn * (1 - scores))
        pn = adj.mean()
        if abs(pn - p) < 1e-6: break
        p = pn
    return p

IPW_FEATS = ['country', 'doc_type', 'decade']
def ipw_estimate(cal, target):
    combined = pd.concat([cal[IPW_FEATS].assign(_t=0), target[IPW_FEATS].assign(_t=1)], ignore_index=True)
    X = pd.get_dummies(combined[IPW_FEATS], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal[LABEL].values, weights=w), 0, 1)

# ============================================================
# 6. Compute prevalence bias across scenarios
# ============================================================
N = 20_000

def rs_country(df, rs=42):
    w = np.where(df['country'] == 'Belgium', 5.0, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)
def rs_doctype(df, rs=42):
    w = np.where(df['doc_type'] == 'bill', 5.0, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)

scenarios = {
    'Baseline':       test_df.sample(n=min(N, len(test_df)), replace=True, random_state=42),
    'Country shift':  rs_country(test_df),
    'Doc-type shift': rs_doctype(test_df),
    'Spain media':    ood_spain.sample(n=min(N, len(ood_spain)), replace=True, random_state=42),
    'Belgium TV':     ood_belgium.sample(n=min(N, len(ood_belgium)), replace=True, random_state=42),
}

print(f"\n{'Scenario':<18} {'True':>6}  {'CC':>7} {'RG':>7} {'SLD':>7} {'IPW':>7} {'Iso':>7} {'MCGrad':>7}")
print("-" * 75)

for name, t in scenarios.items():
    tp = t[LABEL].mean()
    cc = (t['llm_score'] >= THRESHOLD).mean()
    d = cal_tpr - cal_fpr
    rg = np.clip((cc - cal_fpr) / d, 0, 1) if abs(d) > 1e-10 else cc
    sld = sld_estimate(t['llm_score'].values)
    ipw = ipw_estimate(cal_df, t)
    iso = t['iso_pred'].mean()
    mc = t['mc_pred'].mean()

    def fb(e): return f"{(e-tp)*100:+.1f}"
    print(f"{name:<18} {tp:>5.1%}  {fb(cc):>7} {fb(rg):>7} {fb(sld):>7} {fb(ipw):>7} {fb(iso):>7} {fb(mc):>7}")

print("\nCC = Classify & Count, RG = Rogan-Gladen, SLD = Saerens-Latinne-Decaestecker")
print("IPW = Importance-weighted, Iso = Isotonic regression")
print(f"\nThreshold (Youden's J): {THRESHOLD:.4f}")
print("Done.")
