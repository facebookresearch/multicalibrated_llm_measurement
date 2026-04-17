"""Helper functions for the ACS Employment prevalence-estimation analysis."""

from typing import Optional

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_COLUMNS = ["AGEP", "SCHL"]
BINARY_COLUMNS = ["DIS", "ESP", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX"]
CATEGORICAL_COLUMNS = ["MAR", "RELP", "CIT", "MIG", "MIL", "ANC", "RAC1P"]
LABEL_COLUMN = "employment_label"


def estimate_classifier_error_rates(
    labels: pd.Series,
    predictions: pd.Series,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Estimate TPR and FPR from calibration data at a given threshold."""
    binary_preds = (predictions >= threshold).astype(int)
    labels_arr = labels.astype(int)

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
    """Find the threshold where apparent prevalence matches true prevalence."""
    true_prevalence = labels.mean()
    threshold = predictions.quantile(1 - true_prevalence)
    return float(threshold)


def create_logistic_pipeline() -> Pipeline:
    """Create a logistic regression pipeline with preprocessing for ACS data."""
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
    """Load ACS Employment dataset using folktables."""
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

    df[NUMERICAL_COLUMNS] = df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[BINARY_COLUMNS] = (
        df[BINARY_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("Int64")
    )
    for c in CATEGORICAL_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    print(f"Dataset has {len(df)} samples")

    return df


def resample_with_age_shift(
    df: pd.DataFrame,
    age_col: str = "AGEP",
    shift: str = "original",
    n_samples: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Resample df with importance weights that shift the age distribution.

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
        weights = np.exp(-0.08 * (ages - 16))
        weights = np.where(ages <= 30, weights * 5, weights)
    elif shift == "old":
        weights = np.exp(0.08 * (ages - 50))
        weights = np.where(ages >= 60, weights * 5, weights)
    elif shift == "bimodal":
        center = 40
        weights = np.exp(0.04 * np.abs(ages - center))
        weights = np.where((ages <= 25) | (ages >= 65), weights * 3, weights)
    else:
        raise ValueError(f"Unknown shift: {shift}")

    weights = weights / weights.sum()

    return df.sample(n=n_samples, weights=weights, replace=True, random_state=random_state)
