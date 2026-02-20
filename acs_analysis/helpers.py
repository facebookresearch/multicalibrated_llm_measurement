# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

# pyre-ignore[21]: This code is only used in our open source CI where the dependency is present, so we can ignore this
from folktables import ACSDataSource, ACSEmployment
from plotly import express as px, graph_objects as go, io as pio
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


def format_calibration_metrics_table(
    calibration_metrics: dict[str, Any],
    max_digits: int = 4,
) -> pd.DataFrame:
    metrics_table = pd.DataFrame(calibration_metrics).T.round(max_digits)
    col_index = pd.MultiIndex.from_tuples(
        [
            (
                ("Calibration", colname)
                if "ECCE" in colname
                else ("Multicalibration", colname)
            )
            for colname in metrics_table.columns
        ]
    )
    metrics_table.columns = col_index
    return metrics_table


def combine_segment_calibration_plots(
    segment_plots: dict[str, go.Figure],
    quantity: str = "segment_ecces_sigma_scale",
) -> go.Figure:
    """
    Combines multiple segment calibration error plots into a single plot with different colors for each method.

    :param segment_plots: Dictionary where keys are method names and values are plotly figures from plot_segment_calibration_errors
    :param quantity: The MCE quantity being plotted, corresponding to availabel attributes in `metrics.MulticalibrationError`
    :returns: A Plotly Figure object
    """
    if not segment_plots:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, (method_name, plot_fig) in enumerate(segment_plots.items()):
        # Extract data from the original plot
        for trace in plot_fig.data:
            # Skip non-scatter traces (like bars from histograms)
            if trace.type != "scatter":
                continue

            # Extract the original hover template and preserve all segment information
            original_hovertemplate = (
                trace.hovertemplate if hasattr(trace, "hovertemplate") else ""
            )

            # Build new hover template that includes method name and preserves original segment info
            if original_hovertemplate:
                template_lines = original_hovertemplate.split("<br>")
                new_template_lines = [f"<b>{method_name}</b>"]
                new_template_lines.append("Segment Size: %{x}")
                new_template_lines.append(f"{quantity}: %{{y}}")

                # Add all the segment-defining columns from the original template
                for line in template_lines:
                    # Skip lines that are just the main quantity or empty
                    if (
                        quantity not in line
                        and "segment_size" not in line.lower()
                        and line.strip()
                        and "<extra></extra>" not in line
                        and "%{x}" not in line
                        and "%{y}" not in line
                    ):
                        new_template_lines.append(line)

                hovertemplate = "<br>".join(new_template_lines) + "<extra></extra>"
            else:
                # Fallback if no original template
                hovertemplate = (
                    f"<b>{method_name}</b><br>"
                    + "Segment Size: %{x}<br>"
                    + f"{quantity}: %{{y}}<br>"
                    + "<extra></extra>"
                )

            # Create a new trace with the method name and unique color
            new_trace = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode="markers",
                name=method_name,
                marker={"color": colors[i % len(colors)], "size": 5, "opacity": 0.5},
                hovertemplate=hovertemplate,
                # Copy all hover-related data from the original trace
                customdata=trace.customdata if hasattr(trace, "customdata") else None,
                # Copy any other hover-related attributes
                **{
                    k: v
                    for k, v in trace.to_plotly_json().items()
                    if k.startswith("hover") and k != "hovertemplate"
                },
            )
            fig.add_trace(new_trace)

    # Add threshold line for sigma-scale quantities (only for sigma-scale quantities)
    if "sigma" in quantity.lower():
        # Get x-axis range to draw the threshold line across the full plot
        all_x_values = []
        for trace in fig.data:
            if hasattr(trace, "x") and trace.x is not None:
                all_x_values.extend([x for x in trace.x if x is not None])

        if all_x_values:
            x_min = min(all_x_values)
            x_max = max(all_x_values)
            # Extend slightly beyond data range for better visibility
            x_range = [x_min * 0.8, x_max * 1.2]
        else:
            # Fallback range
            x_range = [1, 1000000]

        # Add threshold line as a trace that appears in the legend
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[5, 5],  # Horizontal line at y=5
                mode="lines",
                line={"color": "rgba(247, 152, 111, 1)", "width": 2, "dash": "dot"},
                name="Significant miscalibration</br></br>above this line (5σ)",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Segment Calibration Errors",
        xaxis_title="Segment Size",
        xaxis_type="log",
        template="plotly_white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
        # Add right margin for legend
        margin={"r": 150},
    )

    # Set y-axis title based on MCE quantity
    if quantity == "segment_ecces":
        fig.update_yaxes(title="ECCE", ticksuffix="%")
    elif quantity == "segment_sigmas":
        fig.update_yaxes(title="Standard deviation")
    elif quantity == "segment_p_values":
        fig.update_yaxes(title="P-value")
    elif quantity == "segment_ecces_sigma_scale":
        fig.update_yaxes(title="ECCE / Standard Deviation", ticksuffix="σ")
    else:
        fig.update_yaxes(title=quantity)

    return fig


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


def calibrate_threshold_optimal_accuracy(
    labels: pd.Series,
    predictions: pd.Series,
    n_thresholds: int = 100,
) -> float:
    """
    Find the threshold that maximizes classification accuracy.

    :param labels: True binary labels
    :param predictions: Predicted probabilities
    :param n_thresholds: Number of thresholds to evaluate
    :returns: Optimal threshold
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    labels_arr = labels.astype(int).values
    preds_arr = predictions.values

    best_accuracy = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        binary_preds = (preds_arr >= thresh).astype(int)
        accuracy = (binary_preds == labels_arr).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    return best_threshold


def compute_prevalence_estimates(
    target_df: pd.DataFrame,
    label_col: str,
    score_col: str,
    calibration_tpr: float,
    calibration_fpr: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute prevalence estimates using multiple methods.

    :param target_df: DataFrame containing target population
    :param label_col: Column name for true labels
    :param score_col: Column name for predicted probabilities
    :param calibration_tpr: TPR estimated from calibration set
    :param calibration_fpr: FPR estimated from calibration set
    :param threshold: Classification threshold for binary methods
    :returns: Dictionary with prevalence estimates from each method
    """
    true_prevalence = target_df[label_col].mean()

    # Raw score average
    raw_estimate = target_df[score_col].mean()

    # Classify and count (no adjustment)
    binary_preds = (target_df[score_col] >= threshold).astype(int)
    classify_count_estimate = binary_preds.mean()

    # Rogan-Gladen adjusted count
    rogan_gladen_estimate = compute_rogan_gladen_estimate(
        apparent_prevalence=classify_count_estimate,
        tpr=calibration_tpr,
        fpr=calibration_fpr,
    )

    return {
        "true_prevalence": true_prevalence,
        "raw_scores": raw_estimate,
        "classify_count": classify_count_estimate,
        "rogan_gladen": rogan_gladen_estimate,
    }


def compute_prevalence_bias(
    estimates: dict[str, float],
) -> dict[str, float]:
    """
    Compute bias (estimate - true) for each estimation method.

    :param estimates: Dictionary from compute_prevalence_estimates
    :returns: Dictionary with bias for each method
    """
    true_prev = estimates["true_prevalence"]
    return {
        method: value - true_prev
        for method, value in estimates.items()
        if method != "true_prevalence"
    }


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
