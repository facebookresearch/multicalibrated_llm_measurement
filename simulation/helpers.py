# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings

import numpy as np
import pandas as pd
from mcgrad import methods as mcgrad_methods
from scipy.special import expit

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("mcgrad").setLevel(logging.WARNING)

METHODS = ["uncalibrated", "cc", "rg", "pacc", "sld", "isotonic", "mcgrad"]


def generate_data(n_samples, p_x0, intercepts, slope, score_offsets, rng):
    """Draw (X, Y, score) under covariate shift in X.

    X is a binary feature (e.g. language), U ~ N(0, 1) is document content
    independent of X. The outcome follows P(Y=1 | X, U) = sigmoid(a_X + b U),
    which is stable across populations; only P(X) shifts. The classifier's
    score adds a group-specific logit offset delta_X, so it is miscalibrated
    in a way that depends on X. Scores are continuous and overlap across X,
    so a global recalibration map on the score cannot recover the per-group
    errors.
    """
    X = (rng.random(n_samples) >= p_x0).astype(int)
    U = rng.standard_normal(n_samples)
    true_logit = np.where(X == 1, intercepts[1], intercepts[0]) + slope * U
    offset = np.where(X == 1, score_offsets[1], score_offsets[0])

    Y = rng.binomial(1, expit(true_logit))
    score = expit(true_logit + offset)

    return pd.DataFrame({"X": X.astype(str), "Y": Y, "score": score})


def apply_rogan_gladen(apparent_prevalence, tpr, fpr):
    """Binary Rogan-Gladen: adjusts classify-and-count proportion using binary TPR/FPR."""
    return (apparent_prevalence - fpr) / (tpr - fpr)


def pacc_estimate(scores, pos_mean, neg_mean):
    """Probabilistic Adjusted Classify & Count.

    Soft-score generalization of Rogan-Gladen: uses E[h(X)|Y=1] and E[h(X)|Y=0]
    instead of binary TPR/FPR.
    """
    pcc = scores.mean()
    denom = pos_mean - neg_mean
    if abs(denom) < 1e-10:
        return pcc
    return np.clip((pcc - neg_mean) / denom, 0.0, 1.0)


def sld_estimate(scores, source_prevalence, max_iter=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker (EMQ) prevalence estimator.

    EM algorithm that iteratively re-estimates prevalence by adjusting
    posteriors for a new prior. Assumes label shift (P(X|Y) stable).
    """
    p_hat = source_prevalence
    for _ in range(max_iter):
        ratio_pos = p_hat / source_prevalence
        ratio_neg = (1 - p_hat) / (1 - source_prevalence)
        adjusted = (ratio_pos * scores) / (ratio_pos * scores + ratio_neg * (1 - scores))
        p_new = adjusted.mean()
        if abs(p_new - p_hat) < tol:
            break
        p_hat = p_new
    return p_hat


def fit_estimators(calib_data):
    """Fit every estimator's parameters on the labeled calibration sample."""
    source_prevalence = calib_data["Y"].mean()

    # Classify & Count threshold: the score quantile that reproduces the
    # calibration prevalence.
    cc_threshold = np.quantile(calib_data["score"], 1 - source_prevalence)
    positives = calib_data[calib_data["Y"] == 1]["score"]
    negatives = calib_data[calib_data["Y"] == 0]["score"]

    isotonic = mcgrad_methods.IsotonicRegression().fit(calib_data, "score", "Y")
    mcgrad = mcgrad_methods.MCGrad().fit(
        calib_data, "score", "Y", categorical_feature_column_names=["X"]
    )

    return {
        "source_prevalence": source_prevalence,
        "cc_threshold": cc_threshold,
        "rg_tpr": (positives >= cc_threshold).mean(),
        "rg_fpr": (negatives >= cc_threshold).mean(),
        "pacc_pos_mean": positives.mean(),
        "pacc_neg_mean": negatives.mean(),
        "isotonic": isotonic,
        "mcgrad": mcgrad,
    }


def estimate_prevalences(target, fitted):
    """Apply every estimator, with parameters fixed from calibration, to an unlabeled target."""
    scores = target["score"].values
    cc = (scores >= fitted["cc_threshold"]).mean()

    return {
        "uncalibrated": scores.mean(),
        "cc": cc,
        "rg": apply_rogan_gladen(cc, fitted["rg_tpr"], fitted["rg_fpr"]),
        "pacc": pacc_estimate(scores, fitted["pacc_pos_mean"], fitted["pacc_neg_mean"]),
        "sld": sld_estimate(scores, fitted["source_prevalence"]),
        "isotonic": fitted["isotonic"].predict(target, "score").mean(),
        "mcgrad": fitted["mcgrad"]
        .predict(target, "score", categorical_feature_column_names=["X"])
        .mean(),
    }


def compute_bias_curves_bootstrap(
    B,
    n_samples,
    n_calibration,
    p_x0,
    intercepts,
    slope,
    score_offsets,
    shift_range=(0.01, 0.99),
    n_points=20,
    seed=42,
):
    """Relative bias (%) and squared error (pp^2) of each method across target shifts.

    Each of the B runs draws a fresh calibration sample at P(X=0) = p_x0, fits
    all estimators once, and evaluates them on fresh unlabeled targets with
    P(X=0) on a grid over shift_range.
    """
    rng = np.random.default_rng(seed)
    target_p_x0 = np.linspace(shift_range[0], shift_range[1], n_points)

    bias = {m: np.zeros((B, n_points)) for m in METHODS}
    sq_err = {m: np.zeros((B, n_points)) for m in METHODS}

    for b in range(B):
        calib_data = generate_data(
            n_calibration, p_x0, intercepts, slope, score_offsets, rng
        )
        fitted = fit_estimators(calib_data)

        for j, p in enumerate(target_p_x0):
            target = generate_data(n_samples, p, intercepts, slope, score_offsets, rng)
            true_prevalence = target["Y"].mean()
            for m, estimate in estimate_prevalences(target, fitted).items():
                bias[m][b, j] = 100 * (estimate - true_prevalence) / true_prevalence
                sq_err[m][b, j] = (100 * (estimate - true_prevalence)) ** 2

    results = {"deltas": target_p_x0 - p_x0}
    for m in METHODS:
        results[f"all_{m}"] = bias[m]
        results[f"avg_{m}"] = bias[m].mean(axis=0)
        results[f"mse_{m}"] = sq_err[m].mean(axis=0)
    return results
