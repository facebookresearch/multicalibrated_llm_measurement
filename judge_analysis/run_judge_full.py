# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Full-scale (17,211-doc) LLM-as-judge prevalence analysis under language shift.

Binary campaign only (score methods require the phelp campaign, run at pilot
scale). Methods: Classify&Count, Rogan-Gladen (global sens/spec -- the current
judge-reporting SOTA), IPW, MCGrad (conditioning on language + response length).

Usage:
    python judge_analysis/run_judge_full.py
"""

import glob
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("mcgrad").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mcgrad import methods as mcgrad_methods

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BINARY_DIR = os.path.join(DATA_DIR, "inference_output", "sonnet-binary")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "images")
os.makedirs(IMG_DIR, exist_ok=True)

CAL_LANGS = ["en", "es", "ru", "de"]
OOD_LANGS = ["fr", "zh"]
GOLD_THRESHOLD = float(os.environ.get("GOLD_THRESHOLD", "0.6"))
SEED = 42
N = 6000
N_BOOT = 200
np.random.seed(SEED)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "font.family": "sans-serif",
})

CAT = ["judge_label", "lang"]
NUM = ["log_len"]
METHODS = ["Classify &\nCount", "Rogan-\nGladen", "IPW", "MCGrad"]
SCENARIOS = [
    ("Baseline\n(balanced)", "test", None, "within"),
    ("Shift→ru\n(lenient lang)", "test", "ru", "within"),
    ("Shift→de\n(high base rate)", "test", "de", "within"),
    ("OOD: French\n(held-out lang)", "fr", None, "ood"),
    ("OOD: Chinese\n(held-out lang)", "zh", None, "ood"),
]


def load():
    master = pd.read_csv(os.path.join(DATA_DIR, "oasst_full.csv"))
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(BINARY_DIR, "shard_*.csv"))]
    b = pd.concat(frames, ignore_index=True)
    b["judge_yes"] = (b["answer"].str.strip().str.lower() == "yes").astype(int)
    df = master.merge(b[["id", "judge_yes"]], left_on="doc_id", right_on="id")
    df["label"] = (df["helpfulness"] >= GOLD_THRESHOLD).astype(int)
    df["judge_label"] = df["judge_yes"].map({1: "yes", 0: "no"})
    df["log_len"] = np.log1p(df["resp_len"])
    return df


def ipw_estimate(cal, target, feats=("lang", "log_len")):
    feats = list(feats)
    combined = pd.concat([cal[feats].assign(_t=0), target[feats].assign(_t=1)], ignore_index=True)
    X = pd.get_dummies(combined[feats], columns=["lang"], drop_first=True).values.astype(float)
    z = combined["_t"].values
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return float(np.clip(np.average(cal["label"].values, weights=w), 0, 1))


def resample(frame, wlang, rs):
    w = np.ones(len(frame)) if wlang is None else np.where(frame.lang == wlang, 4.0, 1.0)
    return frame.sample(n=min(N, len(frame)), weights=w / w.sum(), replace=True, random_state=rs)


def run_pipeline(cal, pools, rs):
    base = cal["label"].mean()
    cal = cal.copy(); cal["init"] = base
    mc = mcgrad_methods.MCGrad().fit(cal, "init", "label",
                                     categorical_feature_column_names=CAT,
                                     numerical_feature_column_names=NUM)
    y = cal["label"].values; yh = cal["judge_yes"].values
    tpr = yh[y == 1].mean(); fpr = yh[y == 0].mean(); d = tpr - fpr
    out = {}
    for name, pool_key, wlang, _ in SCENARIOS:
        tgt = resample(pools[pool_key], wlang, rs).copy(); tgt["init"] = base
        true = tgt["label"].mean(); cc = tgt["judge_yes"].mean()
        rg = float(np.clip((cc - fpr) / d, 0, 1)) if abs(d) > 1e-9 else cc
        ipw = ipw_estimate(cal, tgt)
        mcp = mc.predict(tgt, "init", categorical_feature_column_names=CAT,
                         numerical_feature_column_names=NUM).mean()
        out[name] = [(cc - true) * 100, (rg - true) * 100, (ipw - true) * 100, (mcp - true) * 100]
    return out


def main():
    df = load()
    print(f"Loaded {len(df)} judged docs (threshold {GOLD_THRESHOLD}). "
          f"gold={df.label.mean():.3f} judgeYes={df.judge_yes.mean():.3f}")
    print("per-lang gold base rate:",
          {lg: round(df[df.lang == lg].label.mean(), 3) for lg in CAL_LANGS + OOD_LANGS})

    cal_parts, test_parts = [], []
    for lg in CAL_LANGS:
        sub = df[df.lang == lg]
        c, t = train_test_split(sub, test_size=0.40, random_state=SEED, stratify=sub["label"])
        cal_parts.append(c); test_parts.append(t)
    cal = pd.concat(cal_parts, ignore_index=True)
    pools = {"test": pd.concat(test_parts, ignore_index=True)}
    for lg in OOD_LANGS:
        pools[lg] = df[df.lang == lg]
    print(f"cal={len(cal)}  test={len(pools['test'])}  "
          f"fr={len(pools['fr'])}  zh={len(pools['zh'])}")

    point = run_pipeline(cal, pools, SEED)

    print(f"\nBootstrapping {N_BOOT} iterations...")
    boot = {name: [[] for _ in METHODS] for name, *_ in SCENARIOS}
    for b in range(N_BOOT):
        cal_b = cal.sample(n=len(cal), replace=True, random_state=1000 + b)
        try:
            res = run_pipeline(cal_b, pools, 2000 + b)
        except Exception:
            continue
        for name in boot:
            for mi in range(len(METHODS)):
                boot[name][mi].append(res[name][mi])

    def ci(name, mi):
        a = np.array(boot[name][mi]); return np.percentile(a, 2.5), np.percentile(a, 97.5)

    print("\n" + "=" * 90)
    print("FULL-SCALE (17,211 docs) SIGNED BIAS (pp) with 95% bootstrap CI")
    print("=" * 90)
    hdr = ["CC", "RG", "IPW", "MCGrad"]
    print(f"{'Scenario':<26}" + "".join(f"{h:>16}" for h in hdr))
    print("-" * 90)
    for name, *_ in SCENARIOS:
        line = f"{name.replace(chr(10), ' '):<26}"
        for mi in range(len(METHODS)):
            lo, hi = ci(name, mi)
            line += f"{point[name][mi]:>+6.1f}[{lo:>+5.1f},{hi:>+5.1f}]".rjust(16)
        print(line)
    print("-" * 90)

    def mabs(kind, mi):
        return np.mean([abs(point[n][mi]) for n, *r in SCENARIOS if r[2] == kind])
    print("\nMean |bias| (pp):     " + "".join(f"{h:>9}" for h in hdr))
    for kind in ["within", "ood"]:
        print(f"  {kind:<18}" + "".join(f"{mabs(kind, mi):>9.1f}" for mi in range(len(METHODS))))

    # figure
    markers = ["o", "s", "D", "^", "v"]
    x = np.arange(len(METHODS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, title in [(ax1, "within", "Within calibration support"),
                            (ax2, "ood", "Out of support (held-out language)")]:
        scen = [s for s in SCENARIOS if s[3] == kind]
        for j, (name, *_r) in enumerate(scen):
            off = (j - (len(scen) - 1) / 2) * 0.16
            for mi in range(len(METHODS)):
                lo, hi = ci(name, mi); val = point[name][mi]
                ax.errorbar(mi + off, val, yerr=[[val - lo], [hi - val]], fmt=markers[j],
                            color="#333", ms=6, capsize=2, elinewidth=1,
                            markeredgecolor="white", markeredgewidth=0.5,
                            label=name.split("\n")[0] if mi == 0 else None)
        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, title="Scenario", title_fontsize=7, framealpha=0.95)
    ax1.set_ylabel("Prevalence bias (pp) [signed]")
    fig.suptitle("LLM-judge helpfulness prevalence under language shift "
                 "(OASST full, n=17,211, Sonnet judge)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "figure_judge_full.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
