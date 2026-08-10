# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Build the combined main-text figure (CAP + ACS) from the two bias dumps.

Reads paper/images/cap_biases.json and paper/images/acs_biases.json, written by
cap_analysis/opus/run_cap_opus.py and acs_analysis/run_acs.py respectively, and
draws a 2x2 panel figure: rows are the two applications, columns are the shift
regime. All panels share one y-axis so biases are comparable across panels.

Usage:
    conda run -n mcgrad_tutorials python3 paper/make_figure_combined.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})

DOT_COLOR = '#333333'
MEAN_COLOR = '#333333'
MARKERS = ['o', 's', 'D', '^']

with open(os.path.join(IMG_DIR, 'cap_biases.json')) as f:
    cap = json.load(f)
with open(os.path.join(IMG_DIR, 'acs_biases.json')) as f:
    acs = json.load(f)

# The main text reports MCGrad on the bare Yes/No labels only; the elicited-score
# variant stays in SI Table S2. Dropping it here also makes the two rows show the
# same five methods.
DROP_FROM_CAP = ['MC\n(scores)']
cap['methods'] = [m for m in cap['methods'] if m not in DROP_FROM_CAP]
cap['within'] = {m: v for m, v in cap['within'].items() if m not in DROP_FROM_CAP}
cap['ood'] = {m: v for m, v in cap['ood'].items() if m not in DROP_FROM_CAP}
# With only one MCGrad variant left, drop the now-redundant "(binary)" qualifier.
_RENAME = {'MC\n(binary)': 'MCGrad'}
cap['methods'] = [_RENAME.get(m, m) for m in cap['methods']]
cap['within'] = {_RENAME.get(m, m): v for m, v in cap['within'].items()}
cap['ood'] = {_RENAME.get(m, m): v for m, v in cap['ood'].items()}

ROWS = [
    ('CAP: Law & Crime coding (Claude Opus 4.6)', cap),
    ('ACS: employment (logistic regression)', acs),
]
COLS = [('within', 'Within calibration'), ('ood', 'Out-of-distribution')]


def panel(ax, data, key, show_ylabel):
    """Dot plot of |bias| per method, one marker per scenario, mean as a bar."""
    methods = data['methods']
    scenarios = data[f'{key}_scenarios']
    n_sc = len(scenarios)
    for i, method in enumerate(methods):
        biases = data[key][method]
        for j, sc in enumerate(scenarios):
            offset = (j - (n_sc - 1) / 2) * 0.12
            ax.scatter(i + offset, biases[j],
                       color=DOT_COLOR, marker=MARKERS[j % len(MARKERS)], s=32,
                       zorder=5, edgecolors='white', linewidth=0.5,
                       label=sc if i == 0 else None)
        ax.plot([i - 0.3, i + 0.3], [np.mean(biases)] * 2,
                color=MEAN_COLOR, linewidth=2, zorder=4, alpha=0.4)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(y=0, color='#eeeeee', linewidth=0.5)
    if show_ylabel:
        ax.set_ylabel('|Bias| (percentage points)')
    ax.legend(loc='upper left', title='Scenario', title_fontsize=7, framealpha=0.95)


fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True)

for r, (row_title, data) in enumerate(ROWS):
    for c, (key, col_title) in enumerate(COLS):
        ax = axes[r][c]
        panel(ax, data, key, show_ylabel=(c == 0))
        # Column titles sit on every row so each panel is self-describing
        # without colliding with the bold row banner above it.
        ax.set_title(col_title, fontsize=9, color='#444444')

fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.subplots_adjust(hspace=0.42)

# Bold row banners, placed above each row of panels.
for r, (row_title, _) in enumerate(ROWS):
    top = axes[r][0].get_position().y1
    fig.text(0.5, top + 0.035, row_title, fontsize=11, fontweight='bold',
             ha='center', va='bottom')
out = os.path.join(IMG_DIR, 'figure_combined_v1.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved {out}")
