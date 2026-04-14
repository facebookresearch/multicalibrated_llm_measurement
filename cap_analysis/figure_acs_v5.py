"""ACS figure in same format as CAP v5: single color, dot plot, two panels."""
import pandas as pd
import numpy as np
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
os.makedirs('../paper/images', exist_ok=True)

# Import ACS helpers
sys.path.insert(0, '../acs_analysis')
from helpers import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMN, NUMERICAL_COLUMNS,
    create_logistic_pipeline, load_acs_employment_data, resample_with_age_shift,
)

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
# Load and prepare ACS data (same as notebook)
# ============================================================
print("Loading ACS data...")
TRAIN_STATES = ["TX", "MI", "PA", "OH", "IL", "GA", "NC", "VA"]
OOD_STATES = ["CA", "NY", "FL", "WA", "AZ", "CO"]
SURVEY_YEARS = ["2016", "2017", "2018"]

train_source_df = load_acs_employment_data(
    states=TRAIN_STATES, survey_years=SURVEY_YEARS,
    include_state=True, include_year=True,
)
ood_df = load_acs_employment_data(
    states=OOD_STATES, survey_years=SURVEY_YEARS,
    include_state=True, include_year=True,
)

print(f"Train: {len(train_source_df):,}, OOD: {len(ood_df):,}")

# Split
train_df, test_df = train_test_split(
    train_source_df, test_size=0.30, random_state=42,
    stratify=train_source_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1)
)
model_train_df, calibration_df = train_test_split(
    train_df, test_size=0.30, random_state=42,
    stratify=train_df[[LABEL_COLUMN, "STATE"]].apply(tuple, axis=1)
)

# Train model
print("Training logistic regression...")
feature_cols = [c for c in model_train_df.columns if c not in [LABEL_COLUMN, "STATE", "YEAR"]]
X_train = model_train_df[feature_cols]
y_train = model_train_df[LABEL_COLUMN].astype(int).to_numpy()
pipeline = create_logistic_pipeline()
pipeline.fit(X_train, y_train)

BASE_MODEL_COL = 'base_model_prediction'
for df in [calibration_df, test_df, ood_df]:
    X = df[[c for c in df.columns if c not in [LABEL_COLUMN, "STATE", "YEAR"]]]
    df[BASE_MODEL_COL] = pipeline.predict_proba(X)[:, 1]

# Calibrate
print("Fitting calibration methods...")
categorical_segment_features = CATEGORICAL_COLUMNS + BINARY_COLUMNS
numerical_segment_features = NUMERICAL_COLUMNS

isotonic = mcgrad_methods.IsotonicRegression().fit(
    calibration_df, BASE_MODEL_COL, LABEL_COLUMN)
mcgrad = mcgrad_methods.MCGrad(save_training_performance=True)
mcgrad = mcgrad.fit(calibration_df, BASE_MODEL_COL, LABEL_COLUMN,
    categorical_feature_column_names=categorical_segment_features,
    numerical_feature_column_names=numerical_segment_features)

IR_COL = 'isotonic_prediction'
MCGRAD_COL = 'mcgrad_prediction'
for df in [test_df, ood_df]:
    df[IR_COL] = isotonic.predict(df, BASE_MODEL_COL)
    df[MCGRAD_COL] = mcgrad.predict(df, BASE_MODEL_COL,
        categorical_feature_column_names=categorical_segment_features,
        numerical_feature_column_names=numerical_segment_features)

# Calibration params
from helpers import calibrate_threshold_prevalence_matching, estimate_classifier_error_rates
THRESHOLD = calibrate_threshold_prevalence_matching(
    calibration_df[LABEL_COLUMN], calibration_df[BASE_MODEL_COL])
cal_tpr, cal_fpr = estimate_classifier_error_rates(
    calibration_df[LABEL_COLUMN], calibration_df[BASE_MODEL_COL], THRESHOLD)
pacc_pos = calibration_df[calibration_df[LABEL_COLUMN]==1][BASE_MODEL_COL].mean()
pacc_neg = calibration_df[calibration_df[LABEL_COLUMN]==0][BASE_MODEL_COL].mean()
src_prev = calibration_df[LABEL_COLUMN].mean()

# IPW
ALL_FEATURE_COLS = NUMERICAL_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
def ipw_estimate(cal_df, target_df, max_cal_samples=50_000):
    if len(cal_df) > max_cal_samples:
        cal_sub = cal_df.sample(n=max_cal_samples, random_state=42)
    else:
        cal_sub = cal_df
    combined = pd.concat([
        cal_sub[ALL_FEATURE_COLS].assign(_t=0),
        target_df[ALL_FEATURE_COLS].assign(_t=1),
    ], ignore_index=True)
    X = pd.get_dummies(combined[ALL_FEATURE_COLS], columns=CATEGORICAL_COLUMNS, drop_first=True).values.astype(float)
    z = combined['_t'].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, z)
    n = len(cal_sub)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return np.clip(np.average(cal_sub[LABEL_COLUMN].values, weights=w), 0, 1)

# ============================================================
# Methods and scenarios
# ============================================================
methods = ['Classify &\nCount', 'Rogan-\nGladen', 'IPW', 'Isotonic\nRegression', 'MCGrad']

def compute_bias(target, method_name):
    tp = target[LABEL_COLUMN].mean()
    d = cal_tpr - cal_fpr
    if method_name.startswith('Classify'):
        binary_preds = (target[BASE_MODEL_COL] >= THRESHOLD).mean()
        return (binary_preds - tp) * 100
    elif method_name.startswith('Rogan'):
        ap = (target[BASE_MODEL_COL] >= THRESHOLD).mean()
        return (np.clip((ap - cal_fpr) / d, 0, 1) - tp) * 100 if abs(d) > 1e-10 else 0
    elif method_name == 'IPW':
        return (ipw_estimate(calibration_df, target) - tp) * 100
    elif method_name.startswith('Isotonic'):
        return (target[IR_COL].mean() - tp) * 100
    elif method_name == 'MCGrad':
        return (target[MCGRAD_COL].mean() - tp) * 100

age_shifts = {
    'Original': 'original',
    'Young-skewed': 'young',
    'Old-skewed': 'old',
    'Bimodal': 'bimodal',
}

within_cal_scenarios = [
    (name, lambda rs, s=shift: resample_with_age_shift(test_df, shift=s, random_state=rs), marker)
    for (name, shift), marker in zip(age_shifts.items(), ['o', 's', 'D', '^'])
]
ood_scenarios = [
    (name, lambda rs, s=shift: resample_with_age_shift(ood_df, shift=s, random_state=rs), marker)
    for (name, shift), marker in zip(age_shifts.items(), ['o', 's', 'D', '^'])
]

# Bootstrap
print("Running bootstrap (200 iterations)...")
N_BOOT = 200

def get_biases(scenario_list):
    out = {}
    for method in methods:
        out[method] = []
        for sc_name, sample_fn, _ in scenario_list:
            biases = [compute_bias(sample_fn(rs=b), method) for b in range(N_BOOT)]
            out[method].append(np.abs(np.mean(biases)))
            print(f"    {sc_name} x {method.split(chr(10))[0]}: {np.abs(np.mean(biases)):.2f}pp")
    return out

print("  Within-cal...")
within_biases = get_biases(within_cal_scenarios)
print("  OOD...")
ood_biases = get_biases(ood_scenarios)

# ============================================================
# Figure
# ============================================================
print("Generating figure...")

DOT_COLOR = '#333333'
MEAN_COLOR = '#333333'
n_methods = len(methods)
x_methods = np.arange(n_methods)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
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
fig.savefig('../paper/images/figure_acs_v5.png', dpi=300, bbox_inches='tight')
print("  Saved figure_acs_v5.png")
print("Done.")
