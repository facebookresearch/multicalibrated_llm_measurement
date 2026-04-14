"""Prototype two figure options for the simulation."""
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Run simulation (same params as notebook)
# ============================================================
print("Running simulation (B=50 bootstrap iterations)...")
np.random.seed(42)

results = compute_bias_curves_bootstrap(
    B=50,
    n_samples=10_000,
    n_calibration=10_000,
    p_x0=0.5,
    p_y_given_x1=0.85,
    bias_b=0.9,
    bias_c=1.1,
    original_p_x0=0.5,
    shift_range=(0.01, 0.99),
    n_points=20,
)

deltas = results['deltas']

# Methods to show (reduced set — drop RG, PACC, SLD to appendix)
methods_line = [
    ('Classify & Count', results['avg_cc'], '#e41a1c', '-'),
    ('Rogan-Gladen', results['avg_rg'], '#377eb8', '-'),
    ('Isotonic Regression', results['avg_calibrated'], '#ff7f00', '-'),
    ('MCGrad', results['avg_multicalibrated'], '#4daf4a', '-'),
]

methods_all = [
    ('Uncalibrated', results['avg_uncalibrated'], '#1f77b4', '--'),
    ('Classify & Count', results['avg_cc'], '#e41a1c', '-'),
    ('Rogan-Gladen', results['avg_rg'], '#2ca02c', '-'),
    ('PACC', results['avg_pacc'], '#bcbd22', '-'),
    ('SLD (EMQ)', results['avg_sld'], '#17becf', '-'),
    ('Isotonic Regression', results['avg_calibrated'], '#ff7f00', '-'),
    ('MCGrad', results['avg_multicalibrated'], '#4daf4a', '-'),
]

# ============================================================
# Option 1: Dot plot (binned shifts)
# ============================================================
print("Generating Option 1: dot plot...")

# Bin shifts into 4 categories
bins = [
    ('No shift', np.abs(deltas) < 0.05),
    ('Mild\n(|Δ|<0.15)', (np.abs(deltas) >= 0.05) & (np.abs(deltas) < 0.15)),
    ('Moderate\n(|Δ|<0.30)', (np.abs(deltas) >= 0.15) & (np.abs(deltas) < 0.30)),
    ('Extreme\n(|Δ|≥0.30)', np.abs(deltas) >= 0.30),
]

methods_dot = [
    ('Classify &\nCount', results['avg_cc']),
    ('Isotonic\nRegression', results['avg_calibrated']),
    ('MCGrad', results['avg_multicalibrated']),
]

DOT_COLOR = '#333333'
markers = ['o', 's', 'D', '^']

fig1, ax1 = plt.subplots(figsize=(7, 3.5))
x_methods = np.arange(len(methods_dot))

for i, (method_name, bias_curve) in enumerate(methods_dot):
    for j, (bin_name, mask) in enumerate(bins):
        if mask.sum() == 0:
            continue
        mean_abs_bias = np.mean(np.abs(bias_curve[mask]))
        offset = (j - 1.5) * 0.12
        ax1.scatter(i + offset, mean_abs_bias,
                    color=DOT_COLOR, marker=markers[j], s=40, zorder=5,
                    edgecolors='white', linewidth=0.5,
                    label=bin_name if i == 0 else None)
    overall_mean = np.mean(np.abs(bias_curve))
    ax1.plot([i - 0.25, i + 0.25], [overall_mean, overall_mean],
             color=DOT_COLOR, linewidth=2, alpha=0.4, zorder=4)

ax1.set_xticks(x_methods)
ax1.set_xticklabels([m[0] for m in methods_dot], fontsize=8)
ax1.set_ylabel('|Bias| (% relative)')
ax1.set_title('Simulation: Prevalence Estimation Bias')
ax1.axhline(y=0, color='#eeeeee', linewidth=0.5)
ax1.legend(fontsize=7, loc='upper left', title='Shift magnitude', title_fontsize=7)

fig1.tight_layout()
fig1.savefig(os.path.join(IMG_DIR, 'figure_sim_dotplot.png'), dpi=300, bbox_inches='tight')
print("  Saved figure_sim_dotplot.png")

# ============================================================
# Option 2: Clean line plot (reduced methods, single panel)
# ============================================================
print("Generating Option 2: line plot...")

fig2, ax2 = plt.subplots(figsize=(7, 3.5))

for name, bias_curve, color, ls in methods_line:
    # Clip for display
    clipped = np.clip(bias_curve, -40, 40)
    ax2.plot(deltas, clipped, color=color, linestyle=ls, linewidth=1.8,
             label=name, zorder=3)

    # Add arrow indicator if clipped
    if np.any(bias_curve < -40):
        idx = np.where(bias_curve < -40)[0][0]
        ax2.annotate('', xy=(deltas[idx], -40), xytext=(deltas[idx], -36),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax2.axhline(y=0, color='#cccccc', linewidth=0.8, zorder=1)
ax2.axvline(x=0, color='#eeeeee', linewidth=0.5, zorder=0)
ax2.set_xlabel('Distribution shift: $\\Delta P(X\\!=\\!0)$')
ax2.set_ylabel('Average bias (%)')
ax2.set_ylim(-42, 42)

ax2.legend(fontsize=8, loc='lower left')

fig2.tight_layout()
fig2.savefig(os.path.join(IMG_DIR, 'figure_sim_lineplot.png'), dpi=300, bbox_inches='tight')
print("  Saved figure_sim_lineplot.png")

# ============================================================
# Option 2b: Line plot with all methods for comparison
# ============================================================
print("Generating Option 2b: line plot (all methods)...")

fig3, ax3 = plt.subplots(figsize=(7, 3.5))

for name, bias_curve, color, ls in methods_all:
    lw = 2.0 if name in ('MCGrad', 'Classify & Count') else 1.2
    alpha = 1.0 if name in ('MCGrad', 'Classify & Count', 'Isotonic Regression') else 0.6
    ax3.plot(deltas, bias_curve, color=color, linestyle=ls, linewidth=lw,
             label=name, zorder=3, alpha=alpha)

ax3.axhline(y=0, color='#cccccc', linewidth=0.8, zorder=1)
ax3.axvline(x=0, color='#eeeeee', linewidth=0.5, zorder=0)
ax3.set_xlabel('Distribution shift: $\\Delta P(X\\!=\\!0)$')
ax3.set_ylabel('Average bias (%)')
ax3.legend(fontsize=7, loc='lower left', ncol=2)

fig3.tight_layout()
fig3.savefig(os.path.join(IMG_DIR, 'figure_sim_lineplot_all.png'), dpi=300, bbox_inches='tight')
print("  Saved figure_sim_lineplot_all.png")

print("Done.")
