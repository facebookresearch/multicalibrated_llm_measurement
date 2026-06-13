# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce simulation results: prevalence estimation bias under covariate shift.

Generates:
  - Figure 1 (paper):  4-method bias line plot (CC, RG, Global recal., MCGrad), clipped ±40%
  - Figure S2 (SI):    RMSE for the same 4 methods
  - Figure S3 (SI):    All 7 methods bias line plot
  - Console summary table

Usage:
    conda run -n mcgrad_tutorials python3 simulation/run_simulation.py
"""

import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure helpers is importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))
from helpers import compute_bias_curves_bootstrap

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
# Simulation parameters (same as notebook)
# ============================================================
B = 50
N_SAMPLES = 10_000
N_CALIBRATION = 10_000
P_X0 = 0.5
P_Y_GIVEN_X1 = 0.85
BIAS_B = 0.9
BIAS_C = 1.1
SHIFT_RANGE = (0.01, 0.99)
N_POINTS = 20

# ============================================================
# Run bootstrap simulation
# ============================================================
print(f"Running simulation (B={B} bootstrap iterations, n={N_SAMPLES})...")
np.random.seed(42)

results = compute_bias_curves_bootstrap(
    B=B,
    n_samples=N_SAMPLES,
    n_calibration=N_CALIBRATION,
    p_x0=P_X0,
    p_y_given_x1=P_Y_GIVEN_X1,
    bias_b=BIAS_B,
    bias_c=BIAS_C,
    original_p_x0=P_X0,
    shift_range=SHIFT_RANGE,
    n_points=N_POINTS,
)

deltas = results['deltas']

# ============================================================
# Figure 1: 4-method bias line plot (paper main text)
# ============================================================
print("Generating Figure 1 (4-method bias line plot)...")

methods_main = [
    ('Classify & Count', results['avg_cc'], '#e41a1c', '-'),
    ('Rogan-Gladen',     results['avg_rg'], '#377eb8', '-'),
    ('Global recalibration', results['avg_calibrated'], '#ff7f00', '-'),
    ('MCGrad',           results['avg_multicalibrated'], '#4daf4a', '-'),
]

fig1, ax1 = plt.subplots(figsize=(7, 3.5))

for name, bias_curve, color, ls in methods_main:
    # Mask values outside the visible range with NaN so the line breaks at
    # the boundary instead of flatlining along it.
    visible = np.where((bias_curve >= -40) & (bias_curve <= 40),
                       bias_curve, np.nan)
    ax1.plot(deltas, visible, color=color, linestyle=ls, linewidth=1.8,
             label=name, zorder=3)

ax1.axhline(y=0, color='#cccccc', linewidth=0.8, zorder=1)
ax1.axvline(x=0, color='#eeeeee', linewidth=0.5, zorder=0)
ax1.set_xlabel('Distribution shift: $\\Delta P(X\\!=\\!0)$')
ax1.set_ylabel('Average bias (%)')
ax1.set_ylim(-42, 42)
ax1.legend(fontsize=8, loc='lower left')
fig1.tight_layout()
fig1.savefig(os.path.join(IMG_DIR, 'figure_sim_lineplot.png'), dpi=300, bbox_inches='tight')
print(f"  Saved {os.path.join(IMG_DIR, 'figure_sim_lineplot.png')}")
plt.close(fig1)

# ============================================================
# Figure S2: RMSE (same 4 methods)
# ============================================================
print("Generating Figure S2 (RMSE)...")

methods_rmse = [
    ('Classify & Count',    np.sqrt(results['mse_cc']),            '#e41a1c', '-'),
    ('Rogan-Gladen',        np.sqrt(results['mse_rg']),            '#377eb8', '-'),
    ('Global recalibration', np.sqrt(results['mse_calibrated']),    '#ff7f00', '-'),
    ('MCGrad',              np.sqrt(results['mse_multicalibrated']), '#4daf4a', '-'),
]

fig2, ax2 = plt.subplots(figsize=(7, 3.5))

for name, rmse_curve, color, ls in methods_rmse:
    visible = np.where((rmse_curve >= 0) & (rmse_curve <= 40),
                       rmse_curve, np.nan)
    ax2.plot(deltas, visible, color=color, linestyle=ls, linewidth=1.8,
             label=name, zorder=3)

ax2.axhline(y=0, color='#cccccc', linewidth=0.8, zorder=1)
ax2.axvline(x=0, color='#eeeeee', linewidth=0.5, zorder=0)
ax2.set_xlabel('Distribution shift: $\\Delta P(X\\!=\\!0)$')
ax2.set_ylabel('RMSE (percentage points)')
ax2.set_ylim(0, 42)
ax2.legend(fontsize=8, loc='upper left')
fig2.tight_layout()
fig2.savefig(os.path.join(IMG_DIR, 'figure_sim_rmse.png'), dpi=300, bbox_inches='tight')
print(f"  Saved {os.path.join(IMG_DIR, 'figure_sim_rmse.png')}")
plt.close(fig2)

# ============================================================
# Figure S3: All 7 methods bias line plot
# ============================================================
print("Generating Figure S3 (all 7 methods)...")

methods_all = [
    ('Uncalibrated',        results['avg_uncalibrated'], '#1f77b4', '--'),
    ('Classify & Count',    results['avg_cc'],           '#e41a1c', '-'),
    ('Rogan-Gladen',        results['avg_rg'],           '#2ca02c', '-'),
    ('PACC',                results['avg_pacc'],         '#bcbd22', '-'),
    ('SLD (EMQ)',           results['avg_sld'],          '#17becf', '-'),
    ('Global recalibration', results['avg_calibrated'],   '#ff7f00', '-'),
    ('MCGrad',              results['avg_multicalibrated'], '#4daf4a', '-'),
]

fig3, ax3 = plt.subplots(figsize=(7, 3.5))

for name, bias_curve, color, ls in methods_all:
    lw = 2.0 if name in ('MCGrad', 'Classify & Count') else 1.2
    alpha = 1.0 if name in ('MCGrad', 'Classify & Count', 'Global recalibration') else 0.6
    ax3.plot(deltas, bias_curve, color=color, linestyle=ls, linewidth=lw,
             label=name, zorder=3, alpha=alpha)

ax3.axhline(y=0, color='#cccccc', linewidth=0.8, zorder=1)
ax3.axvline(x=0, color='#eeeeee', linewidth=0.5, zorder=0)
ax3.set_xlabel('Distribution shift: $\\Delta P(X\\!=\\!0)$')
ax3.set_ylabel('Average bias (%)')
ax3.legend(fontsize=7, loc='lower left', ncol=2)
fig3.tight_layout()
fig3.savefig(os.path.join(IMG_DIR, 'figure_sim_lineplot_all.png'), dpi=300, bbox_inches='tight')
print(f"  Saved {os.path.join(IMG_DIR, 'figure_sim_lineplot_all.png')}")
plt.close(fig3)

# ============================================================
# Results summary table
# ============================================================
print("\n" + "=" * 70)
print("SIMULATION RESULTS SUMMARY")
print("=" * 70)
print(f"Parameters: B={B}, n={N_SAMPLES}, P(Y=1|X=1)={P_Y_GIVEN_X1}, "
      f"bias_b={BIAS_B}, bias_c={BIAS_C}")
print()

# Bin shifts for summary
bins = [
    ('No shift (|d|<0.05)',     np.abs(deltas) < 0.05),
    ('Mild (|d|<0.15)',         (np.abs(deltas) >= 0.05) & (np.abs(deltas) < 0.15)),
    ('Moderate (|d|<0.30)',     (np.abs(deltas) >= 0.15) & (np.abs(deltas) < 0.30)),
    ('Extreme (|d|>=0.30)',     np.abs(deltas) >= 0.30),
]

all_methods_summary = [
    ('Uncalibrated',        results['avg_uncalibrated'],     results['mse_uncalibrated']),
    ('Classify & Count',    results['avg_cc'],               results['mse_cc']),
    ('Rogan-Gladen',        results['avg_rg'],               results['mse_rg']),
    ('PACC',                results['avg_pacc'],             results['mse_pacc']),
    ('SLD (EMQ)',           results['avg_sld'],              results['mse_sld']),
    ('Global recalibration', results['avg_calibrated'],       results['mse_calibrated']),
    ('MCGrad',              results['avg_multicalibrated'],  results['mse_multicalibrated']),
]

header = f"{'Method':<22}" + "".join(f"{'|Bias|':>10}" for _, _ in bins)
header += f"  {'RMSE (all)':>10}"
print(header)
print("-" * len(header))

for name, bias_curve, mse_curve in all_methods_summary:
    row = f"{name:<22}"
    for _, mask in bins:
        if mask.sum() > 0:
            mean_abs = np.mean(np.abs(bias_curve[mask]))
            row += f"{mean_abs:>9.1f}%"
        else:
            row += f"{'N/A':>10}"
    overall_rmse = np.sqrt(np.mean(mse_curve))
    row += f"  {overall_rmse:>9.2f}pp"
    print(row)

print()
print("Done. Figures saved to paper/images/")
