"""v5: 5 methods (with IPW), single color, scenario by marker shape only."""
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
from mcgrad import methods as mcgrad_methods

os.makedirs('../paper/images', exist_ok=True)

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
# Data loading + model fitting
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

CAL_SUBPOPS = ['Denmark_parliamentary_question', 'Spain_parliamentary_question',
               'United States_bill', 'Belgium_newspaper']
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

BASE_RATE = cal_df['law_crime'].mean()
CAT_BIN = ['country', 'doc_type', 'party', 'llm_label']
NUM = ['decade', 'text_len']

for df in [cal_df, test_df, ood_spain, ood_belgium]:
    df['binary_init'] = BASE_RATE
    df['llm_label'] = df['llm_yes'].map({1: 'yes', 0: 'no'})

CAT_PYN = ['country', 'doc_type', 'party']

print("Fitting models...")
mcgrad_bin = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_bin = mcgrad_bin.fit(cal_df, 'binary_init', 'law_crime',
    categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)

mcgrad_pyn = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad_pyn = mcgrad_pyn.fit(cal_df, 'pyn_score', 'law_crime',
    categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)

isotonic = mcgrad_methods.IsotonicRegression().fit(cal_df, 'pyn_score', 'law_crime')

for df in [test_df, ood_spain, ood_belgium]:
    df['mc_bin'] = mcgrad_bin.predict(df, 'binary_init',
        categorical_feature_column_names=CAT_BIN, numerical_feature_column_names=NUM)
    df['mc_pyn'] = mcgrad_pyn.predict(df, 'pyn_score',
        categorical_feature_column_names=CAT_PYN, numerical_feature_column_names=NUM)
    df['iso_pred'] = isotonic.predict(df, 'pyn_score')

cal_labels = cal_df['law_crime'].astype(int)
cal_preds = cal_df['llm_yes'].astype(int)
cal_tpr = ((cal_preds == 1) & (cal_labels == 1)).sum() / cal_labels.sum()
cal_fpr = ((cal_preds == 1) & (cal_labels == 0)).sum() / (1 - cal_labels).sum()

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
# Scenarios
# ============================================================
N = 5000
def rs_country(df, rs=42):
    w = np.where(df['country'] == 'Belgium', 5.0, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)
def rs_doctype(df, rs=42):
    w = np.where(df['doc_type'] == 'bill', 5.0, 1.0); w /= w.sum()
    return df.sample(n=min(N, len(df)), weights=w, replace=True, random_state=rs)

def sld_estimate(scores, src_prev):
    p = src_prev
    for _ in range(100):
        rp = p / src_prev; rn = (1 - p) / (1 - src_prev)
        adj = (rp * scores) / (rp * scores + rn * (1 - scores))
        pn = adj.mean()
        if abs(pn - p) < 1e-6: break
        p = pn
    return p

SRC_PREV = cal_df['law_crime'].mean()

methods = ['Classify &\nCount', 'Rogan-\nGladen', 'IPW', 'Isotonic\nRegression', 'MCGrad\n(binary)', 'MCGrad\n(scores)']

within_cal_scenarios = [
    ('Baseline', lambda rs: test_df.sample(n=min(N, len(test_df)), replace=True, random_state=rs), 'o'),
    ('Country shift', lambda rs: rs_country(test_df, rs=rs), 's'),
    ('Doc-type shift', lambda rs: rs_doctype(test_df, rs=rs), 'D'),
]
ood_scenarios = [
    ('Spain media', lambda rs: ood_spain.sample(n=min(N, len(ood_spain)), replace=True, random_state=rs), 'o'),
    ('Belgium TV', lambda rs: ood_belgium.sample(n=min(N, len(ood_belgium)), replace=True, random_state=rs), 's'),
]

def compute_bias(target, method_name):
    tp = target['law_crime'].mean()
    d = cal_tpr - cal_fpr
    if method_name.startswith('Classify'):
        return (target['llm_yes'].mean() - tp) * 100
    elif method_name.startswith('Rogan'):
        ap = target['llm_yes'].mean()
        return (np.clip((ap - cal_fpr) / d, 0, 1) - tp) * 100 if abs(d) > 1e-10 else 0
    elif method_name == 'SLD':
        return (sld_estimate(target['pyn_score'].values, SRC_PREV) - tp) * 100
    elif method_name == 'IPW':
        return (ipw_estimate(cal_df, target) - tp) * 100
    elif method_name.startswith('Isotonic'):
        return (target['iso_pred'].mean() - tp) * 100
    elif method_name == 'MCGrad' or method_name.startswith('MCGrad') and 'binary' in method_name:
        return (target['mc_bin'].mean() - tp) * 100
    elif method_name.startswith('MCGrad') and 'scores' in method_name:
        return (target['mc_pyn'].mean() - tp) * 100

# Bootstrap
print("Running bootstrap...")
N_BOOT = 200

def get_biases(scenario_list):
    out = {}
    for method in methods:
        out[method] = []
        for sc_name, sample_fn, _ in scenario_list:
            biases = [compute_bias(sample_fn(rs=b), method) for b in range(N_BOOT)]
            out[method].append(np.abs(np.mean(biases)))
    return out

within_biases = get_biases(within_cal_scenarios)
print("  Within-cal done")
ood_biases = get_biases(ood_scenarios)
print("  OOD done")

# ============================================================
# Figure
# ============================================================
print("Generating figure...")

DOT_COLOR = '#333333'
MEAN_COLOR = '#333333'
n_methods = len(methods)
x_methods = np.arange(n_methods)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                gridspec_kw={'width_ratios': [3, 2]})

# --- Panel 1: Within calibration ---
for i, method in enumerate(methods):
    biases = within_biases[method]
    for j, (sc_name, _, marker) in enumerate(within_cal_scenarios):
        offset = (j - 1) * 0.15
        ax1.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=marker, s=40, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=sc_name if i == 0 else None)
    # Mean line
    mean_val = np.mean(biases)
    ax1.plot([i - 0.25, i + 0.25], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax1.set_xticks(x_methods)
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_ylabel('|Bias| (percentage points)')
ax1.set_title('Within calibration', fontsize=10)
ax1.set_ylim(-0.15, 5.5)
ax1.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax1.legend(fontsize=7, loc='upper left', title='Scenario', title_fontsize=7,
           framealpha=0.95)

# --- Panel 2: OOD ---
for i, method in enumerate(methods):
    biases = ood_biases[method]
    for j, (sc_name, _, marker) in enumerate(ood_scenarios):
        offset = (j - 0.5) * 0.2
        ax2.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=marker, s=40, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=sc_name if i == 0 else None)
    mean_val = np.mean(biases)
    ax2.plot([i - 0.2, i + 0.2], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax2.set_xticks(x_methods)
ax2.set_xticklabels(methods, fontsize=8)
ax2.set_title('Out-of-distribution', fontsize=10)
ax2.set_ylim(-0.3, 14)
ax2.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax2.legend(fontsize=7, loc='upper left', title='Scenario', title_fontsize=7,
           framealpha=0.95)

fig.tight_layout()
fig.savefig('../paper/images/figure_cap_v5.png', dpi=300, bbox_inches='tight')
print("  Saved figure_cap_v5.png")
print("Done.")
