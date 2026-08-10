# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce ACS employment analysis: prevalence estimation under age distribution shift.

Runs all 5 methods (CC, Rogan-Gladen, IPW, Isotonic, MCGrad) across 4 age shifts
in both in-distribution and OOD settings. Generates Figure 2 (two-panel dot plot).

Usage:
    conda run -n mcgrad_tutorials python3 acs_analysis/run_acs.py
"""

import warnings
import json
import logging
import os
import sys

# Silence FutureWarnings from sklearn/pandas/mcgrad — they don't affect numerical
# correctness here. SettingWithCopyWarning is avoided below by .copy()-ing splits.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('mcgrad').setLevel(logging.WARNING)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mcgrad import methods as mcgrad_methods

# Ensure helpers is importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))
from helpers import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN, NUMERICAL_COLUMNS,
    create_logistic_pipeline, load_acs_employment_data,
    resample_with_age_shift,
    calibrate_threshold_prevalence_matching, estimate_classifier_error_rates,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'images')
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
# Configuration
# ============================================================
TRAIN_STATES = ["TX", "MI", "PA", "OH", "IL", "GA", "NC", "VA"]
OOD_STATES = ["CA", "NY", "FL", "WA", "AZ", "CO"]
SURVEY_YEARS = ["2016", "2017", "2018"]
BASE_MODEL_COL = 'base_model_prediction'
IR_COL = 'isotonic_prediction'
MCGRAD_COL = 'mcgrad_prediction'
ALL_FEATURE_COLS = NUMERICAL_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS

# ============================================================
# Load data
# ============================================================
print("Loading ACS data...")
train_source_df = load_acs_employment_data(
    states=TRAIN_STATES, survey_years=SURVEY_YEARS,
    include_state=True, include_year=True,
)
ood_df = load_acs_employment_data(
    states=OOD_STATES, survey_years=SURVEY_YEARS,
    include_state=True, include_year=True,
)
print(f"Train: {len(train_source_df):,}, OOD: {len(ood_df):,}")

# ============================================================
# Train / calibration / test split
# ============================================================
print("Splitting data...")
train_df, test_df = train_test_split(
    train_source_df, test_size=0.30, random_state=42,
    stratify=train_source_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1),
)
model_train_df, calibration_df = train_test_split(
    train_df, test_size=0.30, random_state=42,
    stratify=train_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1),
)
# Detach from parent frames so column assignments below don't trigger
# SettingWithCopyWarning and so writes are guaranteed to land in the splits.
test_df = test_df.copy()
calibration_df = calibration_df.copy()
ood_df = ood_df.copy()

# ============================================================
# Train logistic regression
# ============================================================
print("Training logistic regression...")
feature_cols = [c for c in model_train_df.columns if c not in [LABEL_COLUMN, "STATE", "YEAR"]]
X_train = model_train_df[feature_cols]
y_train = model_train_df[LABEL_COLUMN].astype(int).to_numpy()
pipeline = create_logistic_pipeline()
pipeline.fit(X_train, y_train)

# Score all splits
for df in [calibration_df, test_df, ood_df]:
    X = df[[c for c in df.columns if c not in [LABEL_COLUMN, "STATE", "YEAR"]]]
    df[BASE_MODEL_COL] = pipeline.predict_proba(X)[:, 1]

# ============================================================
# Fit calibration methods (Isotonic + MCGrad)
# ============================================================
print("Fitting calibration methods...")
categorical_segment_features = CATEGORICAL_COLUMNS + BINARY_COLUMNS
numerical_segment_features = NUMERICAL_COLUMNS

isotonic = mcgrad_methods.IsotonicRegression().fit(
    calibration_df, BASE_MODEL_COL, LABEL_COLUMN)
mcgrad = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad = mcgrad.fit(
    calibration_df, BASE_MODEL_COL, LABEL_COLUMN,
    categorical_feature_column_names=categorical_segment_features,
    numerical_feature_column_names=numerical_segment_features,
)

for df in [test_df, ood_df]:
    df[IR_COL] = isotonic.predict(df, BASE_MODEL_COL)
    df[MCGRAD_COL] = mcgrad.predict(
        df, BASE_MODEL_COL,
        categorical_feature_column_names=categorical_segment_features,
        numerical_feature_column_names=numerical_segment_features,
    )

# ============================================================
# CC / RG / PACC calibration parameters
# ============================================================
THRESHOLD = calibrate_threshold_prevalence_matching(
    calibration_df[LABEL_COLUMN], calibration_df[BASE_MODEL_COL])
cal_tpr, cal_fpr = estimate_classifier_error_rates(
    calibration_df[LABEL_COLUMN], calibration_df[BASE_MODEL_COL], THRESHOLD)
src_prev = calibration_df[LABEL_COLUMN].mean()


# ============================================================
# IPW estimator
# ============================================================
def ipw_estimate(cal_df, target_df, max_cal_samples=50_000):
    """Inverse probability weighting prevalence estimate."""
    if len(cal_df) > max_cal_samples:
        cal_sub = cal_df.sample(n=max_cal_samples, random_state=42)
    else:
        cal_sub = cal_df
    combined = pd.concat([
        cal_sub[ALL_FEATURE_COLS].assign(_t=0),
        target_df[ALL_FEATURE_COLS].assign(_t=1),
    ], ignore_index=True)
    X = pd.get_dummies(
        combined[ALL_FEATURE_COLS], columns=CATEGORICAL_COLUMNS, drop_first=True
    ).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal_sub)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal_sub[LABEL_COLUMN].values, weights=w), 0, 1)


# ============================================================
# Bias computation for each method
# ============================================================
def compute_bias(target, method_name):
    """Return bias in percentage points for a given method and target sample."""
    tp = target[LABEL_COLUMN].mean()
    d = cal_tpr - cal_fpr

    if method_name.startswith('Classify'):
        return ((target[BASE_MODEL_COL] >= THRESHOLD).mean() - tp) * 100
    elif method_name.startswith('Rogan'):
        ap = (target[BASE_MODEL_COL] >= THRESHOLD).mean()
        if abs(d) > 1e-10:
            return (np.clip((ap - cal_fpr) / d, 0, 1) - tp) * 100
        # Degenerate case (TPR == FPR): RG is undefined, fall back to CC.
        return (ap - tp) * 100
    elif method_name == 'IPW':
        return (ipw_estimate(calibration_df, target) - tp) * 100
    elif method_name.startswith('Isotonic'):
        return (target[IR_COL].mean() - tp) * 100
    elif method_name == 'MCGrad':
        return (target[MCGRAD_COL].mean() - tp) * 100
    raise ValueError(f"Unknown method: {method_name}")


# ============================================================
# Evaluate all methods x all scenarios
# ============================================================
methods = ['Classify &\nCount', 'Rogan-\nGladen', 'IPW', 'Isotonic\nRegression', 'MCGrad']

age_shifts = {
    'Original': 'original',
    'Young-skewed': 'young',
    'Old-skewed': 'old',
    'Bimodal': 'bimodal',
}
markers = ['o', 's', 'D', '^']

within_cal_scenarios = [
    (name, lambda rs, s=shift: resample_with_age_shift(test_df, shift=s, random_state=rs), marker)
    for (name, shift), marker in zip(age_shifts.items(), markers)
]
ood_scenarios = [
    (name, lambda rs, s=shift: resample_with_age_shift(ood_df, shift=s, random_state=rs), marker)
    for (name, shift), marker in zip(age_shifts.items(), markers)
]

print("Computing biases...")


def get_biases(scenario_list):
    out = {}
    for method in methods:
        out[method] = []
        for sc_name, sample_fn, _ in scenario_list:
            target = sample_fn(rs=42)
            bias = compute_bias(target, method)
            out[method].append(np.abs(bias))
            print(f"    {sc_name} x {method.split(chr(10))[0]}: {np.abs(bias):.2f}pp")
    return out


print("  Within-calibration (in-dist states)...")
within_biases = get_biases(within_cal_scenarios)
print("  Out-of-distribution states...")
ood_biases = get_biases(ood_scenarios)

# ============================================================
# Figure 2: Two-panel dot plot
# ============================================================
print("Generating Figure 2 (ACS dot plot)...")

DOT_COLOR = '#333333'
MEAN_COLOR = '#333333'
n_methods = len(methods)
x_methods = np.arange(n_methods)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True,
                                gridspec_kw={'width_ratios': [1, 1]})

# --- Panel 1: Within calibration (in-dist states) ---
n_within = len(within_cal_scenarios)
for i, method in enumerate(methods):
    biases = within_biases[method]
    for j, (sc_name, _, marker) in enumerate(within_cal_scenarios):
        offset = (j - (n_within - 1) / 2) * 0.12
        ax1.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=marker, s=35, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=sc_name if i == 0 else None)
    mean_val = np.mean(biases)
    ax1.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax1.set_xticks(x_methods)
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_ylabel('|Bias| (percentage points)')
ax1.set_title('In-distribution states', fontsize=10)
ax1.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax1.legend(fontsize=7, loc='upper left', title='Age shift', title_fontsize=7,
           framealpha=0.95)

# --- Panel 2: OOD states ---
n_ood = len(ood_scenarios)
for i, method in enumerate(methods):
    biases = ood_biases[method]
    for j, (sc_name, _, marker) in enumerate(ood_scenarios):
        offset = (j - (n_ood - 1) / 2) * 0.12
        ax2.scatter(i + offset, biases[j],
                    color=DOT_COLOR, marker=marker, s=35, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=sc_name if i == 0 else None)
    mean_val = np.mean(biases)
    ax2.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
             color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)

ax2.set_xticks(x_methods)
ax2.set_xticklabels(methods, fontsize=8)
ax2.set_title('Out-of-distribution states', fontsize=10)
ax2.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax2.legend(fontsize=7, loc='upper left', title='Age shift', title_fontsize=7,
           framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure_acs_v5.png'), dpi=300, bbox_inches='tight')
print(f"  Saved {os.path.join(IMG_DIR, 'figure_acs_v5.png')}")
plt.close(fig)

# Dump absolute biases (pp) so the combined main-text figure can be rebuilt
# without re-running the full analysis.
_bias_dump = {
    'methods': methods,
    'within_scenarios': [n for n, _, _ in within_cal_scenarios],
    'ood_scenarios': [n for n, _, _ in ood_scenarios],
    'within': {m: list(map(float, within_biases[m])) for m in methods},
    'ood': {m: list(map(float, ood_biases[m])) for m in methods},
}
_dump_path = os.path.join(IMG_DIR, 'acs_biases.json')
with open(_dump_path, 'w') as _f:
    json.dump(_bias_dump, _f, indent=2)
print(f"  Saved {_dump_path}")

# ============================================================
# Results summary
# ============================================================
print("\n" + "=" * 70)
print("ACS ANALYSIS RESULTS SUMMARY")
print("=" * 70)

shift_names = list(age_shifts.keys())

for setting_name, biases_dict in [("IN-DISTRIBUTION", within_biases),
                                   ("OUT-OF-DISTRIBUTION", ood_biases)]:
    print(f"\n{setting_name} states:")
    header = f"  {'Method':<20}" + "".join(f"{s:>14}" for s in shift_names) + f"{'Mean':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in methods:
        label = method.replace('\n', ' ')
        biases = biases_dict[method]
        row = f"  {label:<20}"
        for b in biases:
            row += f"{b:>13.2f}pp"
        row += f"{np.mean(biases):>9.2f}pp"
        print(row)

print()
print("Done. Figure saved to paper/images/figure_acs_v5.png")
