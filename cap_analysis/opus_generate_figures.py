"""Generate paper figures from Claude Opus 30K results."""
import pandas as pd
import numpy as np
import glob
import warnings
import logging
import os
import sys

warnings.filterwarnings('ignore')
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from mcgrad import methods as mcgrad_methods

sys.path.insert(0, '..')
from plot_config import METHOD_COLORS

os.makedirs('../paper/images', exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# ============================================================
# Load data
# ============================================================
print("Loading data...")
sample = pd.read_csv('data/opus_30k_sample.csv')
sample['party'] = sample['party'].fillna('unknown')
sample['text_len'] = sample['text'].str.len()

binary_files = sorted(glob.glob('data/inference_output/claude-opus-30k-binary/shard_*.csv'))
binary = pd.concat([pd.read_csv(f) for f in binary_files]).drop_duplicates(subset='id', keep='first')
binary['llm_yes'] = (binary['answer'].str.lower() == 'yes').astype(int)

pyn_files = sorted(glob.glob('data/inference_output/claude-opus-30k-pyn/shard_*.csv'))
pyn = pd.concat([pd.read_csv(f) for f in pyn_files])

data = sample.merge(binary[['id', 'llm_yes']], on='id', how='left')
data = data.merge(pyn[['id', 'score']], on='id', how='left')
data.rename(columns={'score': 'pyn_score'}, inplace=True)
data = data.dropna(subset=['llm_yes'])
data['llm_yes'] = data['llm_yes'].astype(int)

# ============================================================
# Split cal/test
# ============================================================
CAL_SUBPOPS = ['Denmark_parliamentary_question', 'Spain_parliamentary_question',
               'United States_bill', 'Belgium_newspaper']
OOD_SUBPOPS = ['Spain_media', 'Belgium_tv_news']

cal_parts, test_parts = [], []
for key in CAL_SUBPOPS:
    sub = data[data['subpop'] == key].copy()
    cal, test = train_test_split(sub, test_size=0.33, random_state=42, stratify=sub['law_crime'])
    cal_parts.append(cal)
    test_parts.append(test)
cal_df = pd.concat(cal_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

ood_spain = data[data['subpop'] == 'Spain_media'].copy()
ood_belgium = data[data['subpop'] == 'Belgium_tv_news'].copy()

# ============================================================
# Fit models
# ============================================================
print("Fitting models...")
BASE_RATE = cal_df['law_crime'].mean()
CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
CAT_PYN = ['country', 'doc_type', 'party']
NUM = ['decade', 'text_len']

for df in [cal_df, test_df, ood_spain, ood_belgium]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

# MCGrad binary
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)

# MCGrad P(Y/N)
mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)

# Isotonic
isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime')

for df in [test_df, ood_spain, ood_belgium]:
    df['mc_bin'] = mcgrad_bin.predict(df, 'binary_init',
        categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)
    df['mc_pyn'] = mcgrad_pyn.predict(df, 'pyn_score',
        categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)
    df['iso_pred'] = isotonic.predict(df, 'pyn_score')

# ============================================================
# Calibration params for CC, RG, IPW
# ============================================================
# Binary RG
cal_labels = cal_df['law_crime'].astype(int)
cal_preds = cal_df['llm_yes'].astype(int)
cal_tpr = ((cal_preds == 1) & (cal_labels == 1)).sum() / cal_labels.sum()
cal_fpr = ((cal_preds == 1) & (cal_labels == 0)).sum() / (1 - cal_labels).sum()
src_prev = cal_df['law_crime'].mean()

def ipw_estimate(cal, target, features=['country', 'doc_type', 'decade', 'text_len']):
    combined = pd.concat([cal[features].assign(_t=0), target[features].assign(_t=1)], ignore_index=True)
    X = pd.get_dummies(combined[features], drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal['law_crime'].values, weights=w), 0, 1)

# ============================================================
# Figure 3: Score distribution by label and sub-population
# ============================================================
print("Generating Figure 3 (score distribution)...")

subpop_order = ['Denmark_parliamentary_question', 'Spain_parliamentary_question',
                'United States_bill', 'Belgium_newspaper', 'Spain_media', 'Belgium_tv_news']
titles = ['Denmark Questions\n(calibration)', 'Spain Questions\n(calibration)',
          'US Bills\n(calibration)', 'Belgium Newspaper\n(calibration)',
          'Spain Media\n(OOD)', 'Belgium TV\n(OOD)']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, key, title in zip(axes.flat, subpop_order, titles):
    df = data[data['subpop'] == key]
    neg_scores = df.loc[df['law_crime'] == 0, 'pyn_score']
    pos_scores = df.loc[df['law_crime'] == 1, 'pyn_score']

    ax.hist(neg_scores, bins=50, alpha=0.6, color='steelblue',
            label=f'Y=0 (n={len(neg_scores):,})', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, color='coral',
            label=f'Y=1 (n={len(pos_scores):,})', density=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('P(Yes)')
    ax.legend(fontsize=6)

    pos_mean = pos_scores.mean()
    neg_mean = neg_scores.mean()
    ax.text(0.5, 0.95,
            f'Pos mean: {pos_mean:.2f}\nNeg mean: {neg_mean:.2f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Claude Opus 4.6: P(Yes) Score Distribution by Label and Sub-population', fontsize=13)
fig.tight_layout()
fig.savefig('../paper/images/figure_cap_score_distribution.png', dpi=300, bbox_inches='tight')
print("  Saved figure_cap_score_distribution.png")

# ============================================================
# Figure 4: Bias across shift gradient
# ============================================================
print("Generating Figure 4 (shift gradient)...")

N = 5000
def rs_country(df, t='Belgium', f=5.0, rs=42):
    w = np.where(df['country'] == t, f, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)
def rs_doctype(df, t='bill', f=5.0, rs=42):
    w = np.where(df['doc_type'] == t, f, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)

scenarios = {
    'Baseline\n(balanced test)': test_df.sample(n=min(N, len(test_df)), replace=True, random_state=42),
    'Country shift\n(overweight Belgium)': rs_country(test_df),
    'Doc-type shift\n(overweight bills)': rs_doctype(test_df),
    'Spain media\n(OOD doc type)': ood_spain,
    'Belgium TV\n(OOD doc type)': ood_belgium,
}

methods_to_plot = ['Classify & Count', 'Rogan-Gladen', 'IPW',
                   'Isotonic Regression', 'MC (binary)', 'MC (scores)']
colors_map = {
    'Classify & Count': METHOD_COLORS.get('Classify & Count', '#ff7f0e'),
    'Rogan-Gladen': METHOD_COLORS.get('Rogan-Gladen', '#2ca02c'),
    'IPW': '#9467bd',
    'Isotonic Regression': METHOD_COLORS.get('Isotonic Regression', '#d62728'),
    'MC (binary)': '#1f77b4',
    'MC (scores)': '#17becf',
}

# Compute all biases
all_results = {}
for name, target in scenarios.items():
    tp = target['law_crime'].mean()
    cc = target['llm_yes'].mean()
    rg_app = target['llm_yes'].mean()
    d = cal_tpr - cal_fpr
    rg = np.clip((rg_app - cal_fpr) / d, 0, 1) if abs(d) > 1e-10 else rg_app
    ipw = ipw_estimate(cal_df, target)
    iso = target['iso_pred'].mean()
    mcb = target['mc_bin'].mean()
    mcp = target['mc_pyn'].mean()
    all_results[name] = {
        'True Prevalence': tp,
        'Classify & Count': cc,
        'Rogan-Gladen': rg,
        'IPW': ipw,
        'Isotonic Regression': iso,
        'MC (binary)': mcb,
        'MC (scores)': mcp,
    }

scenario_labels = list(scenarios.keys())
fig, ax = plt.subplots(figsize=(14, 6.5))

x = np.arange(len(scenario_labels))
n_methods = len(methods_to_plot)
group_width = 0.82
bar_width = group_width / n_methods * 0.85

for i, method in enumerate(methods_to_plot):
    biases = [
        (all_results[s][method] - all_results[s]['True Prevalence']) * 100
        for s in scenario_labels
    ]
    offset = (i - n_methods / 2 + 0.5) * (group_width / n_methods)
    ax.bar(x + offset, biases, bar_width,
           label=method, color=colors_map[method],
           edgecolor='white', linewidth=0.5)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Bias (percentage points)')
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
ax.legend(fontsize=7, loc='best')

ax.annotate(
    '', xy=(len(scenario_labels) - 0.5, ax.get_ylim()[0]),
    xytext=(-0.5, ax.get_ylim()[0]),
    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5),
)
ax.text(
    len(scenario_labels) / 2 - 0.5, ax.get_ylim()[0] * 0.95,
    'increasing distributional shift $\\longrightarrow$',
    ha='center', va='top', fontsize=9, color='#888888', style='italic',
)

fig.suptitle('Law & Crime Prevalence Estimation Bias Across Shift Gradient (Claude Opus 4.6)', fontsize=13)
fig.tight_layout()
fig.savefig('../paper/images/figure_cap_shift_gradient.png', dpi=300, bbox_inches='tight')
print("  Saved figure_cap_shift_gradient.png")

print("Done.")
