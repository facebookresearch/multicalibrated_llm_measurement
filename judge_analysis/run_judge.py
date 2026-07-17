# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""LLM-as-judge prevalence estimation under cross-language covariate shift.

Estimand: fraction of assistant responses that are "helpful" (human gold),
measured by a Sonnet judge. We show that a globally-corrected judge
(Rogan-Gladen -- the estimator recommended by the current LLM-judge reporting
literature) is biased when the language mix shifts, because judge error rates
differ by language; MCGrad (conditioning on language + response length) fixes it.

Binary campaign only (pilot). Score-based methods (Isotonic, SLD, MC-scores)
require the phelp campaign and are omitted here.

Usage:
    python judge_analysis/run_judge.py
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
PHELP_DIR = os.path.join(DATA_DIR, "inference_output", "sonnet-phelp")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "paper", "images")
os.makedirs(IMG_DIR, exist_ok=True)

CAL_LANGS = ["en", "es", "ru", "de"]
OOD_LANGS = ["fr", "zh"]
# Gold binarization threshold on OASST human helpfulness (0-1). Override with
# GOLD_THRESHOLD env var to check robustness (paper uses 0.6; 0.75 = median).
GOLD_THRESHOLD = float(os.environ.get("GOLD_THRESHOLD", "0.6"))
SEED = 42
N = 4000        # resampled target size per scenario
N_BOOT = 200    # bootstrap iterations (full-pipeline, like ACS)
np.random.seed(SEED)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "font.family": "sans-serif",
})


def load():
    master = pd.read_csv(os.path.join(DATA_DIR, "oasst_pilot.csv"))
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(BINARY_DIR, "shard_*.csv"))]
    binary = pd.concat(frames, ignore_index=True)
    binary["judge_yes"] = (binary["answer"].str.strip().str.lower() == "yes").astype(int)
    pframes = [pd.read_csv(f) for f in glob.glob(os.path.join(PHELP_DIR, "shard_*.csv"))]
    phelp = pd.concat(pframes, ignore_index=True)[["id", "score"]]

    df = master.merge(binary[["id", "judge_yes"]], left_on="doc_id", right_on="id")
    df = df.merge(phelp, on="id")
    df["label"] = (df["helpfulness"] >= GOLD_THRESHOLD).astype(int)  # re-binarize gold
    df["judge_label"] = df["judge_yes"].map({1: "yes", 0: "no"})
    # log response length: raw char counts span orders of magnitude across langs
    df["log_len"] = np.log1p(df["resp_len"])
    return df


def sld_estimate(scores, src_prev, max_iter=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (EMQ) label-shift prevalence estimator."""
    p = src_prev
    for _ in range(max_iter):
        rp = p / src_prev
        rn = (1 - p) / (1 - src_prev)
        adj = (rp * scores) / (rp * scores + rn * (1 - scores))
        pn = adj.mean()
        if abs(pn - p) < tol:
            break
        p = pn
    return float(p)


def ipw_estimate(cal, target, feats=("lang", "log_len")):
    feats = list(feats)
    combined = pd.concat([cal[feats].assign(_t=0), target[feats].assign(_t=1)],
                         ignore_index=True)
    X = pd.get_dummies(combined[feats], columns=["lang"], drop_first=True).values.astype(float)
    z = combined["_t"].values
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(X, z)
    n = len(cal)
    p = clf.predict_proba(X[:n])[:, 1]
    w = p / np.maximum(1 - p, 1e-10)
    return float(np.clip(np.average(cal["label"].values, weights=w), 0, 1))


CAT = ["judge_label", "lang"]
NUM = ["log_len"]
METHODS = ["Classify &\nCount", "Rogan-\nGladen", "IPW", "Isotonic",
           "SLD", "MCGrad\n(binary)", "MCGrad\n(scores)"]
SCENARIOS = [
    ("Baseline\n(balanced)", "test", None, "within"),
    ("Shift→ru\n(lenient lang)", "test", "ru", "within"),
    ("Shift→de\n(high base rate)", "test", "de", "within"),
    ("OOD: French\n(held-out lang)", "fr", None, "ood"),
    ("OOD: Chinese\n(held-out lang)", "zh", None, "ood"),
]


def resample(frame, weight_lang, rs, n=N):
    w = np.ones(len(frame)) if weight_lang is None else \
        np.where(frame.lang == weight_lang, 4.0, 1.0)
    return frame.sample(n=min(n, len(frame)), weights=w / w.sum(),
                        replace=True, random_state=rs)


def run_pipeline(cal, pools, rs):
    """Fit all calibrators on `cal`, return signed bias (pp) per scenario x method."""
    base = cal["label"].mean()
    cal = cal.copy(); cal["init"] = base
    # MCGrad on binary judge label (base-rate init + label/metadata features)
    mc_bin = mcgrad_methods.MCGrad().fit(
        cal, "init", "label",
        categorical_feature_column_names=CAT, numerical_feature_column_names=NUM)
    # MCGrad on elicited P(helpful) scores
    mc_sc = mcgrad_methods.MCGrad().fit(
        cal, "score", "label",
        categorical_feature_column_names=CAT, numerical_feature_column_names=NUM)
    # Isotonic global recalibration on scores
    iso = mcgrad_methods.IsotonicRegression().fit(cal, "score", "label")
    # Rogan-Gladen global error rates
    y = cal["label"].values; yh = cal["judge_yes"].values
    tpr = yh[y == 1].mean(); fpr = yh[y == 0].mean(); d = tpr - fpr
    src_prev = base

    out = {}
    for name, pool_key, wlang, _ in SCENARIOS:
        tgt = resample(pools[pool_key], wlang, rs).copy()
        tgt["init"] = base
        true = tgt["label"].mean()
        cc = tgt["judge_yes"].mean()
        rg = float(np.clip((cc - fpr) / d, 0, 1)) if abs(d) > 1e-9 else cc
        ipw = ipw_estimate(cal, tgt)
        iso_p = iso.predict(tgt, "score").mean()
        sld = sld_estimate(tgt["score"].values, src_prev)
        mcb = mc_bin.predict(tgt, "init", categorical_feature_column_names=CAT,
                             numerical_feature_column_names=NUM).mean()
        mcs = mc_sc.predict(tgt, "score", categorical_feature_column_names=CAT,
                            numerical_feature_column_names=NUM).mean()
        out[name] = [(cc - true) * 100, (rg - true) * 100, (ipw - true) * 100,
                     (iso_p - true) * 100, (sld - true) * 100,
                     (mcb - true) * 100, (mcs - true) * 100]
    return out


def main():
    df = load()
    print(f"Loaded {len(df)} judged docs. gold={df.label.mean():.3f} "
          f"judgeYes={df.judge_yes.mean():.3f}\n")

    cal_parts, test_parts = [], []
    for lg in CAL_LANGS:
        sub = df[df.lang == lg]
        c, t = train_test_split(sub, test_size=0.40, random_state=SEED, stratify=sub["label"])
        cal_parts.append(c); test_parts.append(t)
    cal = pd.concat(cal_parts, ignore_index=True)
    pools = {"test": pd.concat(test_parts, ignore_index=True)}
    for lg in OOD_LANGS:
        pools[lg] = df[df.lang == lg]

    # --- point estimates (seed run) ---
    point = run_pipeline(cal, pools, SEED)

    # --- bootstrap: resample calibration set + targets, refit each time ---
    print(f"Bootstrapping {N_BOOT} full-pipeline iterations...")
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
        arr = np.array(boot[name][mi])
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    # --- table ---
    hdr = ["CC", "RG", "IPW", "Isotonic", "SLD", "MC-bin", "MC-scores"]
    print("\n" + "=" * 132)
    print("SIGNED BIAS (pp) with 95% bootstrap CI   [estimate - true prevalence]")
    print("=" * 132)
    print(f"{'Scenario':<26}" + "".join(f"{h:>15}" for h in hdr))
    print("-" * 132)
    for name, *_ in SCENARIOS:
        line = f"{name.replace(chr(10), ' '):<26}"
        for mi in range(len(METHODS)):
            lo, hi = ci(name, mi)
            line += f"{point[name][mi]:>+5.1f}[{lo:>+4.0f},{hi:>+4.0f}]".rjust(15)
        print(line)
    print("-" * 132)

    def mabs(kind, mi):
        vals = [abs(point[n][mi]) for n, *rest in SCENARIOS if rest[2] == kind]
        return np.mean(vals)
    print("\nMean |bias| (pp):   " + "".join(f"{h:>11}" for h in hdr))
    for kind in ["within", "ood"]:
        print(f"  {kind:<16}" + "".join(f"{mabs(kind, mi):>11.1f}" for mi in range(len(METHODS))))

    # --- figure: two panels, signed bias + 95% CI whiskers ---
    make_figure(point, boot, ci)
    print(f"\nSaved figure to {os.path.join(IMG_DIR, 'figure_judge_v1.png')}")


def make_figure(point, boot, ci):
    within = [s for s in SCENARIOS if s[3] == "within"]
    oods = [s for s in SCENARIOS if s[3] == "ood"]
    markers = ["o", "s", "D", "^", "v"]
    x = np.arange(len(METHODS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)

    for ax, scen, title in [(ax1, within, "Within calibration support"),
                            (ax2, oods, "Out of support (held-out language)")]:
        for j, (name, *_rest) in enumerate(scen):
            off = (j - (len(scen) - 1) / 2) * 0.16
            for mi in range(len(METHODS)):
                lo, hi = ci(name, mi)
                val = point[name][mi]
                ax.errorbar(mi + off, val, yerr=[[val - lo], [hi - val]],
                            fmt=markers[j], color="#333", ms=6, capsize=2,
                            elinewidth=1, markeredgecolor="white", markeredgewidth=0.5,
                            label=name.split("\n")[0] if mi == 0 else None)
        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, title="Scenario", title_fontsize=7, framealpha=0.95)
    ax1.set_ylabel("Prevalence bias (pp)  [signed]")
    fig.suptitle("LLM-judge helpfulness prevalence under language shift "
                 "(OASST, Sonnet judge)", fontsize=12)
    fig.tight_layout()
    tag = "v1" if abs(GOLD_THRESHOLD - 0.6) < 1e-9 else f"thr{GOLD_THRESHOLD:g}"
    fig.savefig(os.path.join(IMG_DIR, f"figure_judge_{tag}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
