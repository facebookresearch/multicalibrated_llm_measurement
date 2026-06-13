# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ReadMe quantification baseline for the CAP application.

Adds the canonical political-science quantifier ReadMe (Hopkins & King 2010) as
a covariate-shift baseline alongside the methods in run_cap_opus.py, on the SAME
calibration split and the SAME shift scenarios, so the numbers are directly
comparable to SI Table S2.

Why ReadMe (Hopkins-King 2010) and not the official ReadMe2 package:
  - ReadMe2 (Jerzak, King & Strezhnev 2023) refines the *feature summary* (word
    vectors + a matched-subspace selection) but estimates category proportions
    under the SAME label-shift assumption: it requires P(text-features | class)
    to be stable between the labeled and target sets. The question here is
    whether that assumption survives covariate shift, where P(X) changes while
    P(Y|X) is stable. Classic ReadMe is the cleanest, fully-specified test of
    that assumption and needs no TensorFlow/embedding backend.
  - The official IQSS `readme` package builds multilingual word-vector summaries
    via a Keras/TensorFlow backend; it is not installed here and is fragile to
    build for a 4-language corpus. We implement the Hopkins-King estimator
    directly and note the official ReadMe2 run as a possible confirmation.

ReadMe estimator (Hopkins & King 2010):
  Documents are summarized by binary word-presence features S. For a random
  subset of features, the identity P(S) = sum_d P(S | D=d) P(D) is solved for
  the class proportions P(D): P(S|D) is estimated on the labeled calibration
  set, P(S) on the unlabeled target. Averaging over many random feature subsets
  gives the prevalence estimate. The method is classifier-free.

Usage:
    conda run -n mcgrad_tutorials python3 cap_analysis/readme_baseline.py
"""
import logging
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'data')

# ============================================================
# 1. Load + merge (identical to run_cap_opus.py)
# ============================================================
print("Loading data...")
sample = pd.read_csv(os.path.join(DATA_DIR, 'opus_30k_sample.csv'))
sample['party'] = sample['party'].fillna('unknown')
sample['text'] = sample['text'].fillna('')
sample['text_len'] = sample['text'].str.len()

binary = pd.read_csv(
    os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-binary/merged.csv')
).drop_duplicates(subset='id', keep='first')
binary['llm_yes'] = (binary['answer'].str.lower() == 'yes').astype(int)

pyn = pd.read_csv(
    os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-pyn/merged.csv')
)

data = sample.merge(binary[['id', 'llm_yes']], on='id', how='left')
data = data.merge(pyn[['id', 'score']], on='id', how='left')
data.rename(columns={'score': 'pyn_score'}, inplace=True)
data = data.dropna(subset=['llm_yes'])
data['llm_yes'] = data['llm_yes'].astype(int)

# ============================================================
# 2. Split + scenarios (identical to run_cap_opus.py)
# ============================================================
CAL_SUBPOPS = [
    'Denmark_parliamentary_question',
    'Spain_parliamentary_question',
    'United States_bill',
    'Belgium_newspaper',
]

cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = data[data['subpop'] == key].copy()
    cal, test = train_test_split(sub, test_size=0.33, random_state=42,
                                 stratify=sub['law_crime'])
    cal_parts.append(cal)
    test_parts.append(test)

cal_df = pd.concat(cal_parts, ignore_index=True).copy()
test_df = pd.concat(test_parts, ignore_index=True).copy()
ood_spain = data[data['subpop'] == 'Spain_media'].copy()
ood_belgium = data[data['subpop'] == 'Belgium_tv_news'].copy()

BASE_RATE = cal_df['law_crime'].mean()
CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']
for df in [cal_df, test_df, ood_spain, ood_belgium]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

print(f"  Cal: {len(cal_df):,} | Test: {len(test_df):,} | "
      f"OOD Spain: {len(ood_spain):,} | OOD Belgium: {len(ood_belgium):,}")

# ============================================================
# 3. Existing comparison methods (copied from run_cap_opus.py)
# ============================================================
print("Fitting MCGrad / isotonic / baselines...")
mcgrad_bin = mcgrad_methods.MCGrad().fit(
    cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)
mcgrad_pyn = mcgrad_methods.MCGrad().fit(
    cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)
isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime')

for df in [test_df, ood_spain, ood_belgium]:
    df['mc_bin'] = mcgrad_bin.predict(
        df, 'binary_init', categorical_feature_column_names=CAT_BIN,
        numerical_feature_column_names=NUM)
    df['mc_pyn'] = mcgrad_pyn.predict(
        df, 'pyn_score', categorical_feature_column_names=CAT_PYN,
        numerical_feature_column_names=NUM)
    df['iso_pred'] = isotonic.predict(df, 'pyn_score')

# Scenarios are built AFTER predictions so the sampled frames carry the
# mc_bin / mc_pyn / iso_pred columns (identical construction to run_cap_opus.py).
N = 5000


def resample_country(df, target='Belgium', factor=5.0, rs=42):
    w = np.where(df['country'] == target, factor, 1.0)
    return df.sample(n=min(N, len(df)), weights=w / w.sum(), replace=True, random_state=rs)


def resample_doctype(df, target='bill', factor=5.0, rs=42):
    w = np.where(df['doc_type'] == target, factor, 1.0)
    return df.sample(n=min(N, len(df)), weights=w / w.sum(), replace=True, random_state=rs)


scenarios = {
    'Baseline (balanced test)': test_df.sample(n=min(N, len(test_df)), replace=True, random_state=42),
    'Country shift (overweight Belgium)': resample_country(test_df),
    'Doc-type shift (overweight bills)': resample_doctype(test_df),
    'Spain media (OOD doc type)': ood_spain,
    'Belgium TV (OOD doc type)': ood_belgium,
}
WITHIN = list(scenarios.keys())[:3]
OOD = list(scenarios.keys())[3:]

cal_labels = cal_df['law_crime'].astype(int)
cal_preds = cal_df['llm_yes'].astype(int)
cal_tpr = ((cal_preds == 1) & (cal_labels == 1)).sum() / cal_labels.sum()
cal_fpr = ((cal_preds == 1) & (cal_labels == 0)).sum() / (1 - cal_labels).sum()
src_prev = cal_df['law_crime'].mean()


def ipw_estimate(cal, target, features=('country', 'doc_type', 'decade', 'text_len')):
    features = list(features)
    combined = pd.concat([cal[features].assign(_t=0), target[features].assign(_t=1)],
                         ignore_index=True)
    X = pd.get_dummies(combined[features], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal['law_crime'].values, weights=w), 0, 1)


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    p_hat = source_prevalence
    for _ in range(max_iter):
        rp = p_hat / source_prevalence
        rn = (1 - p_hat) / (1 - source_prevalence)
        adj = (rp * scores) / (rp * scores + rn * (1 - scores))
        p_new = adj.mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


# ============================================================
# 4. ReadMe (Hopkins & King 2010)
# ============================================================
print("Building document-term matrix for ReadMe...")
# Binary word-presence features. Vocabulary built on the labeled calibration set,
# mid-frequency band (avoid the rarest and the near-ubiquitous), capped by count.
vectorizer = CountVectorizer(
    binary=True, lowercase=True, strip_accents=None,
    min_df=10, max_df=0.9, max_features=3000,
)
cal_dtm = vectorizer.fit_transform(cal_df['text'].values)
V = cal_dtm.shape[1]
print(f"  Vocabulary size: {V}")
cal_codes_mat = np.asarray(cal_dtm.todense(), dtype=np.int8)
cal_y = cal_df['law_crime'].values.astype(int)
n1, n0 = (cal_y == 1).sum(), (cal_y == 0).sum()


def _props(codes, n):
    u, c = np.unique(codes, return_counts=True)
    return dict(zip(u.tolist(), (c / n).tolist()))


def readme_estimate(target_texts, n_feat=15, n_runs=300, seed=0):
    """ReadMe (Hopkins-King 2010) positive-class prevalence estimate.

    Solves P(S) = P(S|D=1) p + P(S|D=0) (1-p) by least squares for p over many
    random feature subsets and averages. Closed-form 1D solve per subset
    (two classes). Returns mean and std of per-run estimates.
    """
    tgt_dtm = vectorizer.transform(pd.Series(target_texts).fillna('').values)
    tgt_mat = np.asarray(tgt_dtm.todense(), dtype=np.int8)
    n_t = tgt_mat.shape[0]
    rng = np.random.default_rng(seed)
    pw = (1 << np.arange(n_feat, dtype=np.int64))  # bit weights
    ests = []
    for _ in range(n_runs):
        idx = rng.choice(V, size=n_feat, replace=False)
        cal_codes = cal_codes_mat[:, idx].astype(np.int64) @ pw
        tgt_codes = tgt_mat[:, idx].astype(np.int64) @ pw
        m1 = _props(cal_codes[cal_y == 1], n1)
        m0 = _props(cal_codes[cal_y == 0], n0)
        bt = _props(tgt_codes, n_t)
        keys = set(m1) | set(m0) | set(bt)
        a = np.array([m1.get(k, 0.0) - m0.get(k, 0.0) for k in keys])
        c = np.array([bt.get(k, 0.0) - m0.get(k, 0.0) for k in keys])
        denom = float(a @ a)
        if denom < 1e-12:
            continue
        p = float(np.clip((a @ c) / denom, 0.0, 1.0))
        ests.append(p)
    ests = np.array(ests)
    return ests.mean(), ests.std()


# ============================================================
# 5. Compute all estimates per scenario
# ============================================================
print("Computing prevalence estimates (incl. ReadMe)...")
rows = []
readme_cache = {}
for name, target in scenarios.items():
    tp = target['law_crime'].mean()
    cc = target['llm_yes'].mean()
    d = cal_tpr - cal_fpr
    rg = np.clip((cc - cal_fpr) / d, 0, 1) if abs(d) > 1e-10 else cc
    ipw = ipw_estimate(cal_df, target)
    iso = target['iso_pred'].mean()
    mcb = target['mc_bin'].mean()
    mcp = target['mc_pyn'].mean()
    sld = sld_estimate(target['pyn_score'].values, src_prev)
    rm_mean, rm_std = readme_estimate(target['text'].values, n_feat=15, n_runs=300, seed=0)
    readme_cache[name] = (rm_mean, rm_std)
    rows.append({
        'Scenario': name, 'Region': 'within' if name in WITHIN else 'OOD',
        'True': tp, 'CC': cc, 'RG': rg, 'IPW': ipw, 'Iso': iso,
        'MC_binary': mcb, 'MC_scores': mcp, 'SLD': sld, 'ReadMe': rm_mean,
        'ReadMe_runstd': rm_std,
    })

res = pd.DataFrame(rows)

# Bias table (percentage points)
bias = res.copy()
for col in ['CC', 'RG', 'IPW', 'Iso', 'MC_binary', 'MC_scores', 'SLD', 'ReadMe']:
    bias[col] = (res[col] - res['True']) * 100
bias['True'] = res['True'] * 100

# ============================================================
# 6. ReadMe sensitivity to subset size
# ============================================================
print("ReadMe sensitivity sweep over subset size...")
sens_rows = []
for nf in [8, 12, 15, 20, 25]:
    row = {'n_feat': nf}
    for name, target in scenarios.items():
        m, _ = readme_estimate(target['text'].values, n_feat=nf, n_runs=200, seed=1)
        row[name] = (m - target['law_crime'].mean()) * 100
    sens_rows.append(row)
sens = pd.DataFrame(sens_rows)

# ============================================================
# 6b. Diagnostic: feature density per scenario (document-length mechanism)
# ============================================================
# ReadMe's binary word-presence summary is sensitive to how many features each
# document activates; large differences in feature density between calibration
# and target violate the stable-P(S|D) assumption via document length.
cal_density = float(np.asarray(cal_dtm.sum(axis=1)).mean())
dens_rows = [{'set': 'CALIBRATION', 'mean_active_features': cal_density,
             'median_text_len': float(cal_df['text_len'].median())}]
for name, target in scenarios.items():
    t_dtm = vectorizer.transform(pd.Series(target['text'].values).fillna('').values)
    dens_rows.append({
        'set': name,
        'mean_active_features': float(np.asarray(t_dtm.sum(axis=1)).mean()),
        'median_text_len': float(target['text_len'].median()),
    })
density = pd.DataFrame(dens_rows)

# ============================================================
# 7. Save + print
# ============================================================
res.to_csv(os.path.join(OUT_DIR, 'readme_baseline_estimates.csv'), index=False)
bias.to_csv(os.path.join(OUT_DIR, 'readme_baseline_bias.csv'), index=False)
sens.to_csv(os.path.join(OUT_DIR, 'readme_baseline_sensitivity.csv'), index=False)

pd.set_option('display.width', 160, 'display.max_columns', 20)
print("\n" + "=" * 110)
print("PREVALENCE ESTIMATION BIAS (percentage points): ReadMe vs existing methods")
print("=" * 110)
show = bias[['Scenario', 'Region', 'True', 'CC', 'RG', 'IPW', 'Iso',
             'MC_binary', 'MC_scores', 'SLD', 'ReadMe']].copy()
for c in ['True', 'CC', 'RG', 'IPW', 'Iso', 'MC_binary', 'MC_scores', 'SLD', 'ReadMe']:
    show[c] = show[c].map(lambda v: f"{v:+.1f}")
show['True'] = [f"{v*100:.1f}%" for v in res['True']]
print(show.to_string(index=False))
print("\nReadMe per-run std (Monte Carlo over feature subsets):")
for name in scenarios:
    m, s = readme_cache[name]
    print(f"  {name:<38s} est={m*100:5.1f}%  run-std={s*100:.1f}pp")
print("\nReadMe sensitivity to subset size (bias pp):")
print(sens.round(1).to_string(index=False))
print("\nFeature-density diagnostic (mean active binary features per document):")
print(density.round(1).to_string(index=False))
density.to_csv(os.path.join(OUT_DIR, 'readme_baseline_density.csv'), index=False)
print("\nSaved: readme_baseline_estimates.csv, readme_baseline_bias.csv, "
      "readme_baseline_sensitivity.csv, readme_baseline_density.csv in", OUT_DIR)
print("Done.")
