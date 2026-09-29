"""
Shared pytest fixtures for the Machine Failure Drift Monitor test suite.
Keeps tests fast, deterministic, and isolated.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import (
    create_features,
    build_preprocessor,
    load_and_split_data,
)
from src.models import (
    train_models,
    calibrate_model,
)

RAW_DATA_PATH = "data/raw/ai4i2020.csv"
REF_BASELINE_PATH = "data/processed/reference_baseline.csv"
SCENARIO_B_PATH = "data/processed/drift_scenario_b_torque.csv"
SCENARIO_C_PATH = "data/processed/drift_scenario_c_variance.csv"


@pytest.fixture(scope="session")
def train_val_data():
    """Load train and validation splits with strict test quarantine (return_test=False)."""
    (X_train, y_train), (X_val, y_val) = load_and_split_data(
        RAW_DATA_PATH, random_seed=42, return_test=False
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }


@pytest.fixture(scope="session")
def trained_pipeline(train_val_data):
    """Fit preprocessor and calibrated champion model strictly on X_train."""
    X_train = train_val_data["X_train"]
    y_train = train_val_data["y_train"]

    preprocessor = build_preprocessor()
    X_train_feat = create_features(X_train)
    X_train_proc = preprocessor.fit_transform(X_train_feat)

    base_model = train_models(X_train_proc, y_train, random_seed=42)
    champion_model = calibrate_model(base_model, X_train_proc, y_train, cv=5)

    return {
        "preprocessor": preprocessor,
        "champion_model": champion_model,
    }


@pytest.fixture(scope="session")
def reference_batch():
    """Load Phase 10 reference baseline dataset."""
    return pd.read_csv(REF_BASELINE_PATH)


@pytest.fixture(scope="session")
def scenario_b_batch():
    """Load Phase 10 torque drift scenario dataset."""
    return pd.read_csv(SCENARIO_B_PATH)


@pytest.fixture(scope="session")
def scenario_c_batch():
    """Load Phase 10 variance expansion scenario dataset."""
    return pd.read_csv(SCENARIO_C_PATH)
