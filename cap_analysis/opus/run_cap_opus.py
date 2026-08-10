# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
CAP Opus reproduction script.

Reproduces the CAP (Comparative Agendas Project) analysis using Claude Opus 4.6
inference results on 30K documents. Generates Figure 3 (bias dot plot) and
SI Figure S4 (score distribution), plus a console results summary.

Usage:
    conda run -n mcgrad_tutorials python3 cap_analysis/opus/run_cap_opus.py
"""
import json
import logging
import os
import warnings

# Silence FutureWarnings from sklearn/pandas/mcgrad — they don't affect numerical
# correctness here. SettingWithCopyWarning is avoided below by .copy()-ing splits.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

# Resolve paths relative to this file so the script runs from any cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
IMG_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'paper', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
sample = pd.read_csv(os.path.join(DATA_DIR, 'opus_30k_sample.csv'))
sample['party'] = sample['party'].fillna('unknown')
sample['text_len'] = sample['text'].str.len()

# Binary campaign (has ~100 duplicate IDs)
binary = pd.read_csv(
    os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-binary/merged.csv')
).drop_duplicates(subset='id', keep='first')
binary['llm_yes'] = (binary['answer'].str.lower() == 'yes').astype(int)

# P(Y/N) campaign
pyn = pd.read_csv(
    os.path.join(DATA_DIR, 'inference_output/claude-opus-30k-pyn/merged.csv')
)

# Merge
data = sample.merge(binary[['id', 'llm_yes']], on='id', how='left')
data = data.merge(pyn[['id', 'score']], on='id', how='left')
data.rename(columns={'score': 'pyn_score'}, inplace=True)
data = data.dropna(subset=['llm_yes'])
data['llm_yes'] = data['llm_yes'].astype(int)

print(f"  Sample: {len(sample):,} | Binary: {len(binary):,} | "
      f"P(Y/N): {len(pyn):,} | Merged: {len(data):,}")

# ============================================================
# 2. Train/test split (67/33 stratified per subpop)
# ============================================================
CAL_SUBPOPS = [
    'Denmark_parliamentary_question',
    'Spain_parliamentary_question',
    'United States_bill',
    'Belgium_newspaper',
]
OOD_SUBPOPS = ['Spain_media', 'Belgium_tv_news']

cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = data[data['subpop'] == key].copy()
    cal, test = train_test_split(sub, test_size=0.33, random_state=42, stratify=sub['law_crime'])
    cal_parts.append(cal)
    test_parts.append(test)

cal_df = pd.concat(cal_parts, ignore_index=True).copy()
test_df = pd.concat(test_parts, ignore_index=True).copy()
ood_spain = data[data['subpop'] == 'Spain_media'].copy()
ood_belgium = data[data['subpop'] == 'Belgium_tv_news'].copy()

print(f"  Cal: {len(cal_df):,} | Test: {len(test_df):,} | "
      f"OOD Spain: {len(ood_spain):,} | OOD Belgium: {len(ood_belgium):,}")

# ============================================================
# 3. Fit calibration models
# ============================================================
print("Fitting models...")
BASE_RATE = cal_df['law_crime'].mean()
CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']

for df in [cal_df, test_df, ood_spain, ood_belgium]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

# MCGrad with binary labels (base rate init + LLM label as feature)
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(
    cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN,
    numerical_feature_column_names=NUM,
)

# MCGrad with P(Y/N) scores
mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(
    cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN,
    numerical_feature_column_names=NUM,
)

# Isotonic regression on P(Y/N)
isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime')

# Predict on all target sets
for df in [test_df, ood_spain, ood_belgium]:
    df['mc_bin'] = mcgrad_bin.predict(
        df, 'binary_init',
        categorical_feature_column_names=CAT_BIN,
        numerical_feature_column_names=NUM,
    )
    df['mc_pyn'] = mcgrad_pyn.predict(
        df, 'pyn_score',
        categorical_feature_column_names=CAT_PYN,
        numerical_feature_column_names=NUM,
    )
    df['iso_pred'] = isotonic.predict(df, 'pyn_score')

# ============================================================
# 4. Baseline estimators (CC, RG, IPW, SLD)
# ============================================================
# Binary error rates for Rogan-Gladen
cal_labels = cal_df['law_crime'].astype(int)
cal_preds = cal_df['llm_yes'].astype(int)
cal_tpr = ((cal_preds == 1) & (cal_labels == 1)).sum() / cal_labels.sum()
cal_fpr = ((cal_preds == 1) & (cal_labels == 0)).sum() / (1 - cal_labels).sum()
src_prev = cal_df['law_crime'].mean()


def ipw_estimate(cal, target, features=('country', 'doc_type', 'decade', 'text_len')):
    """Inverse probability weighting prevalence estimate."""
    features = list(features)
    combined = pd.concat([
        cal[features].assign(_t=0),
        target[features].assign(_t=1),
    ], ignore_index=True)
    X = pd.get_dummies(combined[features], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal['law_crime'].values, weights=w), 0, 1)


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (EMQ) prevalence estimator."""
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
# 5. Define scenarios and compute estimates
# ============================================================
N = 5000


def resample_country(df, target='Belgium', factor=5.0, rs=42):
    w = np.where(df['country'] == target, factor, 1.0)
    w = w / w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)


def resample_doctype(df, target='bill', factor=5.0, rs=42):
    w = np.where(df['doc_type'] == target, factor, 1.0)
    w = w / w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)


scenarios = {
    'Baseline\n(balanced test)': test_df.sample(n=min(N, len(test_df)), replace=True, random_state=42),
    'Country shift\n(overweight Belgium)': resample_country(test_df),
    'Doc-type shift\n(overweight bills)': resample_doctype(test_df),
    'Spain media\n(OOD doc type)': ood_spain,
    'Belgium TV\n(OOD doc type)': ood_belgium,
}

WITHIN_CAL_SCENARIOS = list(scenarios.keys())[:3]
OOD_SCENARIOS = list(scenarios.keys())[3:]

# Compute all prevalence estimates
all_results = {}
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

    all_results[name] = {
        'True Prevalence': tp,
        'Classify &\nCount': cc,
        'Rogan-\nGladen': rg,
        'IPW': ipw,
        'Isotonic\nRegression': iso,
        'MC\n(binary)': mcb,
        'MC\n(scores)': mcp,
        'SLD (EMQ)': sld,
    }

methods = ['Classify &\nCount', 'Rogan-\nGladen', 'IPW',
           'Isotonic\nRegression', 'MC\n(binary)', 'MC\n(scores)']

# ============================================================
# 6. Figure 3: Bias dot plot (two panels)
# ============================================================
print("Generating Figure 3 (bias dot plot)...")

DOT_COLOR = '#333333'
MEAN_COLOR = '#333333'
n_methods = len(methods)
x_methods = np.arange(n_methods)

within_markers = ['o', 's', 'D']
ood_markers = ['o', 's']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                gridspec_kw={'width_ratios': [1, 1]})

# --- Panel 1: Within calibration ---
n_within = len(WITHIN_CAL_SCENARIOS)
for i, method in enumerate(methods):
    biases = [abs((all_results[s][method] - all_results[s]['True Prevalence']) * 100)
              for s in WITHIN_CAL_SCENARIOS]
    for j, sc_name in enumerate(WITHIN_CAL_SCENARIOS):
        # Short label for legend
        short = sc_name.split('\n')[0]
        offset = (j - (n_within - 1) / 2) * 0.12
        ax1.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=within_markers[j], s=35, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=short if i == 0 else None)
    mean_val = np.mean(biases)
    ax1.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax1.set_xticks(x_methods)
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_ylabel('|Bias| (percentage points)')
ax1.set_title('Within calibration', fontsize=10)
ax1.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax1.legend(fontsize=7, loc='upper left', title='Scenario', title_fontsize=7,
           framealpha=0.95)

# --- Panel 2: OOD ---
n_ood = len(OOD_SCENARIOS)
for i, method in enumerate(methods):
    biases = [abs((all_results[s][method] - all_results[s]['True Prevalence']) * 100)
              for s in OOD_SCENARIOS]
    for j, sc_name in enumerate(OOD_SCENARIOS):
        short = sc_name.split('\n')[0]
        offset = (j - (n_ood - 1) / 2) * 0.12
        ax2.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=ood_markers[j], s=35, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=short if i == 0 else None)
    mean_val = np.mean(biases)
    ax2.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax2.set_xticks(x_methods)
ax2.set_xticklabels(methods, fontsize=8)
ax2.set_title('Out-of-distribution', fontsize=10)
ax2.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax2.legend(fontsize=7, loc='upper left', title='Scenario', title_fontsize=7,
           framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure_cap_v5.png'), dpi=300, bbox_inches='tight')
print(f"  Saved {os.path.join(IMG_DIR, 'figure_cap_v5.png')}")

# Dump absolute biases (pp) so the combined main-text figure can be rebuilt
# without re-running the full analysis.
_bias_dump = {
    'methods': methods,
    'within_scenarios': [s.split('\n')[0] for s in WITHIN_CAL_SCENARIOS],
    'ood_scenarios': [s.split('\n')[0] for s in OOD_SCENARIOS],
    'within': {m: [abs((all_results[s][m] - all_results[s]['True Prevalence']) * 100)
                   for s in WITHIN_CAL_SCENARIOS] for m in methods},
    'ood': {m: [abs((all_results[s][m] - all_results[s]['True Prevalence']) * 100)
                for s in OOD_SCENARIOS] for m in methods},
}
_dump_path = os.path.join(IMG_DIR, 'cap_biases.json')
with open(_dump_path, 'w') as _f:
    json.dump(_bias_dump, _f, indent=2)
print(f"  Saved {_dump_path}")

# ============================================================
# 7. SI Figure S4: Score distribution by label and sub-population
# ============================================================
print("Generating SI Figure S4 (score distribution)...")

subpop_order = [
    'Denmark_parliamentary_question', 'Spain_parliamentary_question',
    'United States_bill', 'Belgium_newspaper',
    'Spain_media', 'Belgium_tv_news',
]
subpop_titles = [
    'Denmark Questions\n(calibration)', 'Spain Questions\n(calibration)',
    'US Bills\n(calibration)', 'Belgium Newspaper\n(calibration)',
    'Spain Media\n(OOD)', 'Belgium TV\n(OOD)',
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, key, title in zip(axes.flat, subpop_order, subpop_titles):
    df_sub = data[data['subpop'] == key]
    neg_scores = df_sub.loc[df_sub['law_crime'] == 0, 'pyn_score']
    pos_scores = df_sub.loc[df_sub['law_crime'] == 1, 'pyn_score']

    ax.hist(neg_scores, bins=50, alpha=0.6, color='steelblue',
            label=f'Y=0 (n={len(neg_scores):,})', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, color='coral',
            label=f'Y=1 (n={len(pos_scores):,})', density=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('P(Yes)')
    ax.legend(fontsize=6)

    ax.text(0.5, 0.95,
            f'Pos mean: {pos_scores.mean():.2f}\nNeg mean: {neg_scores.mean():.2f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Claude Opus 4.6: P(Yes) Score Distribution by Label and Sub-population', fontsize=13)
fig.tight_layout()
fig.savefig(
    os.path.join(IMG_DIR, 'figure_cap_score_distribution.png'),
    dpi=300, bbox_inches='tight',
)
print(f"  Saved {os.path.join(IMG_DIR, 'figure_cap_score_distribution.png')}")

# ============================================================
# 8. Results summary table
# ============================================================
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

# Print header
header_methods = ['CC', 'RG', 'IPW', 'Isotonic', 'MC(bin)', 'MC(score)', 'SLD']
result_keys = ['Classify &\nCount', 'Rogan-\nGladen', 'IPW',
               'Isotonic\nRegression', 'MC\n(binary)', 'MC\n(scores)', 'SLD (EMQ)']

print(f"\n{'Scenario':<30s} {'True':>6s}", end='')
for h in header_methods:
    print(f" {h:>9s}", end='')
print()
print("-" * 90)

for sc_name in list(scenarios.keys()):
    r = all_results[sc_name]
    short_name = sc_name.replace('\n', ' ')
    tp = r['True Prevalence']
    print(f"{short_name:<30s} {tp:>6.1%}", end='')
    for key in result_keys:
        bias = (r[key] - tp) * 100
        print(f" {bias:>+8.1f}p", end='')
    print()

print("-" * 90)
print("Values are bias in percentage points (estimate - true prevalence).")
print("SLD results printed for reference only (not in Figure 3, reported in SI table).")
print()

# Calibration quality
print(f"Calibration base rate: {BASE_RATE:.3f}")
print(f"Binary TPR: {cal_tpr:.3f}, FPR: {cal_fpr:.3f}")
print(f"Unique P(Y/N) values: {data['pyn_score'].nunique()}")

print("\nDone.")
