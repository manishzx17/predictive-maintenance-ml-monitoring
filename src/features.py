"""
Feature engineering and preprocessing pipeline for Machine Failure Prediction.
Preserves stable, validated logic from Phases 1-3.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RAW_NUMERICAL_COLS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_NUMERICAL_COLS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Diff",
    "Power",
    "Wear_Load",
]

CATEGORICAL_COLS: List[str] = ["Type"]

TARGET_COL: str = "Machine failure"

INTENDED_RAW_FEATURES: List[str] = CATEGORICAL_COLS + RAW_NUMERICAL_COLS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-engineered features from raw sensor data:
    - Temp_Diff: Process temperature [K] - Air temperature [K]
    - Power: Mechanical power = Torque [Nm] * Rotational speed [rad/s]
    - Wear_Load: Cumulative stress interaction = Tool wear [min] * Torque [Nm]

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw sensor columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with added domain features.
    """
    out = df.copy()
    out["Temp_Diff"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * (2 * np.pi / 60)
    out["Wear_Load"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    return out


def build_preprocessor() -> ColumnTransformer:
    """
    Construct the standard scikit-learn ColumnTransformer:
    - StandardScaler applied to all 8 continuous engineered numerical features.
    - OneHotEncoder (handle_unknown='ignore', sparse_output=False) applied to 'Type'.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor instance.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), ENGINEERED_NUMERICAL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


def load_and_split_data(
    raw_data_path: str,
    target_col: str = TARGET_COL,
    random_seed: int = 42,
    return_test: bool = False,
) -> Union[
    Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]],
    Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ],
]:
    """
    Load raw dataset and partition into stratified train, validation, and test sets (70/15/15).

    BEHAVIORAL QUARANTINE GUARANTEE:
    By default (return_test=False), the test split is strictly withheld from the return tuple,
    preventing any accidental data leakage into training, validation, or drift monitoring.

    Parameters
    ----------
    raw_data_path : str
        Path to raw CSV file.
    target_col : str
        Name of binary failure target column.
    random_seed : int
        Deterministic random seed (default 42).
    return_test : bool
        Whether to return the quarantined test set (default False).

    Returns
    -------
    If return_test is False:
        ((X_train, y_train), (X_val, y_val))
    If return_test is True:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    df = pd.read_csv(raw_data_path)
    X = df[INTENDED_RAW_FEATURES].copy()
    y = df[target_col].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    if not return_test:
        return (X_train, y_train), (X_val, y_val)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def validate_schema(
    df: pd.DataFrame,
    require_target: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate input dataframe against expected sensor schema and physical bounds:
    - Required columns present (INTENDED_RAW_FEATURES, and optionally TARGET_COL)
    - Valid machine Type categories ('L', 'M', 'H')
    - Sensor values within plausible physical bounds:
      * Air & Process temperatures > 0 Kelvin
      * Rotational speed > 0 rpm
      * Torque >= 0 Nm
      * Tool wear >= 0 min
    - No null/missing values in required columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    require_target : bool
        Whether TARGET_COL must be present.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of error messages)
    """
    errors: List[str] = []

    # 1. Required column check
    required_cols = list(INTENDED_RAW_FEATURES)
    if require_target:
        required_cols.append(TARGET_COL)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # 2. Null values check
    null_counts = df[required_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Columns contain null values: {cols_with_nulls}")

    # 3. Categorical values check
    if "Type" in df.columns:
        invalid_types = set(df["Type"].dropna().unique()) - {"L", "M", "H"}
        if invalid_types:
            errors.append(f"Invalid Type values found: {invalid_types}. Expected subset of {{'L', 'M', 'H'}}")

    # 4. Physical sensor bounds
    if "Air temperature [K]" in df.columns:
        if (df["Air temperature [K]"] <= 0).any():
            errors.append("Air temperature [K] contains non-positive values (must be > 0 K)")

    if "Process temperature [K]" in df.columns:
        if (df["Process temperature [K]"] <= 0).any():
            errors.append("Process temperature [K] contains non-positive values (must be > 0 K)")

    if "Rotational speed [rpm]" in df.columns:
        if (df["Rotational speed [rpm]"] <= 0).any():
            errors.append("Rotational speed [rpm] contains non-positive values (must be > 0 rpm)")

    if "Torque [Nm]" in df.columns:
        if (df["Torque [Nm]"] < 0).any():
            errors.append("Torque [Nm] contains negative values")

    if "Tool wear [min]" in df.columns:
        if (df["Tool wear [min]"] < 0).any():
            errors.append("Tool wear [min] contains negative values")

    # 5. Target column validity (if present)
    if TARGET_COL in df.columns:
        invalid_targets = set(df[TARGET_COL].dropna().unique()) - {0, 1}
        if invalid_targets:
            errors.append(f"Target column contains non-binary values: {invalid_targets}")

    is_valid = len(errors) == 0
    return is_valid, errors

