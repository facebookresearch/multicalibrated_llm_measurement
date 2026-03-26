# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

# pyre-ignore[21]: This code is only used in our open source CI where the dependency is present, so we can ignore this
from folktables import ACSDataSource, ACSEmployment
from plotly import graph_objects as go, io as pio
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Module level attribute to store start time for elapsed time logger formatting
_ELAPSED_LOGGER_START_TIME: Optional[float] = None


class ElapsedFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "[%(levelname)s]"
                "[%(asctime)s]"
                "%(elapsed)s"
                "[%(module)s.py:%(lineno)d]: %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        start_time = _ELAPSED_LOGGER_START_TIME
        if start_time is not None:
            elapsed_seconds = time.monotonic() - start_time
            total_seconds = int(elapsed_seconds)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            # pyre-ignore[16]: Dynamic attribute for elapsed time formatting
            record.elapsed = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        else:
            # pyre-ignore[16]: Dynamic attribute for elapsed time formatting
            record.elapsed = ""

        return super().format(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger with our formatter, removing existing handlers."""
    root = logging.getLogger()

    for h in root.handlers[:]:
        root.removeHandler(h)

    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    root.addHandler(handler)


def get_plotting_template() -> go.layout.Template:
    template = pio.templates["plotly_white"]
    template.layout.width = 800
    template.layout.height = 500
    template.layout.autosize = False
    return template


def setup_plotting() -> None:
    """Configure plotly with custom template and logging."""
    pio.templates["custom"] = get_plotting_template()
    pio.templates.default = "custom"
    configure_logging()


NUMERICAL_COLUMNS = ["AGEP", "SCHL"]
BINARY_COLUMNS = ["DIS", "ESP", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX"]
CATEGORICAL_COLUMNS = ["MAR", "RELP", "CIT", "MIG", "MIL", "ANC", "RAC1P"]
LABEL_COLUMN = "employment_label"


def compute_rogan_gladen_estimate(
    apparent_prevalence: float,
    tpr: float,
    fpr: float,
) -> float:
    """
    Compute the Rogan-Gladen adjusted prevalence estimate.

    The Rogan-Gladen estimator corrects for classifier error rates:
    adjusted_prevalence = (apparent_prevalence - FPR) / (TPR - FPR)

    :param apparent_prevalence: Proportion of positive predictions in target population
    :param tpr: True positive rate (sensitivity) estimated from calibration data
    :param fpr: False positive rate (1 - specificity) estimated from calibration data
    :returns: Adjusted prevalence estimate
    """
    denominator = tpr - fpr
    if abs(denominator) < 1e-10:
        # Degenerate case: classifier has no discriminative power
        return apparent_prevalence
    adjusted = (apparent_prevalence - fpr) / denominator
    # Clip to valid probability range
    return max(0.0, min(1.0, adjusted))


def estimate_classifier_error_rates(
    labels: pd.Series,
    predictions: pd.Series,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Estimate TPR and FPR from calibration data.

    :param labels: True binary labels
    :param predictions: Predicted probabilities
    :param threshold: Classification threshold (default 0.5)
    :returns: Tuple of (tpr, fpr)
    """
    binary_preds = (predictions >= threshold).astype(int)
    labels_arr = labels.astype(int)

    # True positives, false positives, etc.
    tp = ((binary_preds == 1) & (labels_arr == 1)).sum()
    fp = ((binary_preds == 1) & (labels_arr == 0)).sum()
    tn = ((binary_preds == 0) & (labels_arr == 0)).sum()
    fn = ((binary_preds == 0) & (labels_arr == 1)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


def calibrate_threshold_prevalence_matching(
    labels: pd.Series,
    predictions: pd.Series,
) -> float:
    """
    Find the threshold where apparent prevalence matches true prevalence.

    This is the optimal threshold for Classify & Count when applying to the
    same distribution as calibration data.

    :param labels: True binary labels
    :param predictions: Predicted probabilities
    :returns: Calibrated threshold
    """
    true_prevalence = labels.mean()
    # Threshold = (1 - true_prevalence) quantile of predictions
    # This ensures that the proportion of predictions above threshold equals true prevalence
    threshold = predictions.quantile(1 - true_prevalence)
    return float(threshold)


def create_logistic_pipeline() -> Pipeline:
    """
    Create a logistic regression pipeline with preprocessing for ACS Employment data.

    The pipeline includes:
    - StandardScaler for numerical columns
    - Passthrough for binary columns
    - OneHotEncoder for categorical columns
    - LogisticRegression classifier

    :returns: Unfitted sklearn Pipeline ready for training.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def load_acs_employment_data(
    states: Optional[list[str]] = None,
    survey_years: Optional[list[str]] = None,
    horizon: str = "1-Year",
    root_dir: str = "./data/acs",
    include_state: bool = False,
    include_year: bool = False,
) -> pd.DataFrame:
    """
    Load ACS Employment dataset using folktables.

    :param states: List of state abbreviations to download (e.g., ["TX", "MI"]).
                   Defaults to ["TX", "MI"] if not provided.
    :param survey_years: List of ACS survey years (e.g., ["2017", "2018"]).
                         Defaults to ["2018"] if not provided.
    :param horizon: Survey horizon ("1-Year" or "5-Year"). Defaults to "1-Year".
    :param root_dir: Directory to cache downloaded data. Defaults to "./data/acs".
    :param include_state: If True, include a STATE column with state abbreviations.
    :param include_year: If True, include a YEAR column with survey year.
    :returns: DataFrame with ACS features and employment_label column, with proper dtypes.
    """
    if states is None:
        states = ["TX", "MI"]
    if survey_years is None:
        survey_years = ["2018"]

    all_dfs = []

    for year in survey_years:
        data_source = ACSDataSource(
            survey_year=year,
            horizon=horizon,
            survey="person",
            root_dir=root_dir,
        )

        if include_state:
            # Load each state separately to preserve state information
            for state in states:
                acs_data = data_source.get_data(states=[state], download=True)
                features, labels, _ = ACSEmployment.df_to_numpy(acs_data)
                state_df = pd.DataFrame(data=features, columns=ACSEmployment.features)
                state_df[LABEL_COLUMN] = labels
                state_df["STATE"] = state
                if include_year:
                    state_df["YEAR"] = int(year)
                all_dfs.append(state_df)
        else:
            acs_data = data_source.get_data(states=states, download=True)
            features, labels, _ = ACSEmployment.df_to_numpy(acs_data)
            year_df = pd.DataFrame(data=features, columns=ACSEmployment.features)
            year_df[LABEL_COLUMN] = labels
            if include_year:
                year_df["YEAR"] = int(year)
            all_dfs.append(year_df)

    df = pd.concat(all_dfs, ignore_index=True)

    # Convert columns to proper dtypes
    df[NUMERICAL_COLUMNS] = df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[BINARY_COLUMNS] = (
        df[BINARY_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("Int64")
    )
    for c in CATEGORICAL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    print(f"Dataset has {len(df)} samples")

    return df


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


def compute_all_prevalence_estimates(
    target_df: pd.DataFrame,
    base_col: str,
    ir_col: str,
    mcgrad_col: str,
    label_col: str,
    cal_tpr: float,
    cal_fpr: float,
    pacc_pos_mean: float,
    pacc_neg_mean: float,
    source_prevalence: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute prevalence estimates using all methods."""
    true_prevalence = target_df[label_col].mean()

    # Raw base model scores
    raw_estimate = target_df[base_col].mean()

    # Classify and count (threshold-based, no adjustment)
    binary_preds = (target_df[base_col] >= threshold).astype(int)
    classify_count = binary_preds.mean()

    # Rogan-Gladen adjusted count
    rogan_gladen = compute_rogan_gladen_estimate(classify_count, cal_tpr, cal_fpr)

    # PACC (soft-score Rogan-Gladen)
    pacc = pacc_estimate(target_df[base_col].values, pacc_pos_mean, pacc_neg_mean)

    # SLD (EMQ)
    sld = sld_estimate(target_df[base_col].values, source_prevalence)

    # Isotonic regression calibrated scores
    isotonic_estimate = target_df[ir_col].mean()

    # MCGrad calibrated scores
    mcgrad_estimate = target_df[mcgrad_col].mean()

    return {
        "True Prevalence": true_prevalence,
        "Raw Scores": raw_estimate,
        "Classify & Count": classify_count,
        "Rogan-Gladen": rogan_gladen,
        "PACC": pacc,
        "SLD (EMQ)": sld,
        "Isotonic Regression": isotonic_estimate,
        "MCGrad": mcgrad_estimate,
    }


def resample_with_age_shift(
    df: pd.DataFrame,
    age_col: str = "AGEP",
    shift: str = "original",
    n_samples: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample df with importance weights that shift the age distribution.

    shift options:
      - "original": uniform weights (baseline)
      - "young": heavily oversample ages 16-30
      - "old": heavily oversample ages 60+
      - "bimodal": oversample both young and old, undersample middle
    """
    ages = df[age_col].values

    if shift == "original":
        weights = np.ones(len(df))
    elif shift == "young":
        # Exponentially favor younger ages
        weights = np.exp(-0.08 * (ages - 16))
        weights = np.where(ages <= 30, weights * 5, weights)
    elif shift == "old":
        # Exponentially favor older ages
        weights = np.exp(0.08 * (ages - 50))
        weights = np.where(ages >= 60, weights * 5, weights)
    elif shift == "bimodal":
        # Favor both tails — young and old
        center = 40
        weights = np.exp(0.04 * np.abs(ages - center))
        weights = np.where((ages <= 25) | (ages >= 65), weights * 3, weights)
    else:
        raise ValueError(f"Unknown shift: {shift}")

    weights = weights / weights.sum()

    return df.sample(n=n_samples, weights=weights, replace=True, random_state=random_state)


def compute_bootstrap_mse(
    source_df: pd.DataFrame,
    shift: str,
    base_col: str,
    ir_col: str,
    mcgrad_col: str,
    label_col: str,
    cal_tpr: float,
    cal_fpr: float,
    pacc_pos_mean: float,
    pacc_neg_mean: float,
    source_prevalence: float,
    threshold: float,
    n_bootstrap: int = 200,
    n_samples: int = 20_000,
) -> dict[str, dict[str, float]]:
    """Bootstrap resampling to compute bias, variance, and RMSE of prevalence estimators."""
    methods_list = ["Raw Scores", "Classify & Count", "Rogan-Gladen", "PACC",
                    "SLD (EMQ)", "Isotonic Regression", "MCGrad"]
    estimates_by_method = {m: [] for m in methods_list}
    true_prevs = []

    for b in range(n_bootstrap):
        syn_df = resample_with_age_shift(source_df, shift=shift, n_samples=n_samples, random_state=b)
        est = compute_all_prevalence_estimates(
            syn_df, base_col, ir_col, mcgrad_col, label_col,
            cal_tpr, cal_fpr, pacc_pos_mean, pacc_neg_mean, source_prevalence, threshold
        )
        true_prevs.append(est["True Prevalence"])
        for m in methods_list:
            estimates_by_method[m].append(est[m])

    results = {}
    for m in methods_list:
        ests = np.array(estimates_by_method[m])
        trues = np.array(true_prevs)
        errors = ests - trues
        results[m] = {
            "bias": np.mean(errors) * 100,  # in percentage points
            "variance": np.var(errors) * 100**2,  # in pp²
            "rmse": np.sqrt(np.mean(errors**2)) * 100,  # in pp
        }
    results["True Prevalence"] = np.mean(true_prevs)
    return results


def compute_bias_table(estimates: dict[str, float]) -> pd.DataFrame:
    """Create a DataFrame showing estimates and bias for each method."""
    true_prev = estimates["True Prevalence"]
    rows = []
    for method, estimate in estimates.items():
        if method == "True Prevalence":
            continue
        bias = estimate - true_prev
        rows.append({
            "Method": method,
            "Estimate": estimate,
            "Bias": bias,
            "Relative Bias (%)": 100 * bias / true_prev if true_prev > 0 else 0,
        })
    return pd.DataFrame(rows).set_index("Method")
