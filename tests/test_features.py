"""
Tests for feature engineering and data schema validation.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import (
    create_features,
    validate_schema,
    RAW_NUMERICAL_COLS,
    ENGINEERED_NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)


class TestFeatureEngineering:
    """Validate mathematical exactness and schema of domain-engineered features."""

    def test_feature_engineering_exactness(self):
        """Verify Temp_Diff, Power, and Wear_Load compute exact physical formulas."""
        df = pd.DataFrame({
            "Air temperature [K]": [300.0, 298.1],
            "Process temperature [K]": [310.0, 308.6],
            "Rotational speed [rpm]": [1500.0, 1800.0],
            "Torque [Nm]": [40.0, 35.5],
            "Tool wear [min]": [120.0, 50.0],
            "Type": ["M", "L"],
        })

        out = create_features(df)

        # Temp_Diff = Process - Air
        np.testing.assert_allclose(out["Temp_Diff"], [10.0, 10.5], rtol=1e-5)

        # Power = Torque * Rotational_speed * (2 * pi / 60)
        expected_power_0 = 40.0 * 1500.0 * (2 * np.pi / 60)
        expected_power_1 = 35.5 * 1800.0 * (2 * np.pi / 60)
        np.testing.assert_allclose(out["Power"], [expected_power_0, expected_power_1], rtol=1e-5)

        # Wear_Load = Tool wear * Torque
        np.testing.assert_allclose(out["Wear_Load"], [4800.0, 1775.0], rtol=1e-5)

    def test_feature_columns_preservation(self):
        """Verify original columns are preserved and new engineered columns added."""
        df = pd.DataFrame({
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500.0],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [0.0],
            "Type": ["H"],
        })
        out = create_features(df)

        for col in ENGINEERED_NUMERICAL_COLS:
            assert col in out.columns, f"Missing engineered column: {col}"
        assert "Type" in out.columns


class TestDataValidation:
    """Validate data schema, required columns, and physical sensor boundaries."""

    @pytest.fixture
    def valid_sample_df(self):
        return pd.DataFrame({
            "Air temperature [K]": [298.1, 300.5],
            "Process temperature [K]": [308.6, 310.2],
            "Rotational speed [rpm]": [1551.0, 1400.0],
            "Torque [Nm]": [42.8, 48.0],
            "Tool wear [min]": [0.0, 120.0],
            "Type": ["M", "L"],
            "Machine failure": [0, 1],
        })

    def test_valid_schema_passes(self, valid_sample_df):
        """Verify a clean dataset passes schema validation."""
        is_valid, errors = validate_schema(valid_sample_df, require_target=True)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_column_fails(self, valid_sample_df):
        """Verify missing sensor column is flagged as invalid."""
        broken_df = valid_sample_df.drop(columns=["Torque [Nm]"])
        is_valid, errors = validate_schema(broken_df)
        assert is_valid is False
        assert any("Torque [Nm]" in err for err in errors)

    def test_missing_target_fails_when_required(self, valid_sample_df):
        """Verify missing target column fails when require_target=True."""
        unlabeled_df = valid_sample_df.drop(columns=["Machine failure"])
        is_valid_unlabeled, errors = validate_schema(unlabeled_df, require_target=False)
        assert is_valid_unlabeled is True

        is_valid_req, errors_req = validate_schema(unlabeled_df, require_target=True)
        assert is_valid_req is False
        assert any("Machine failure" in err for err in errors_req)

    def test_null_values_flagged(self, valid_sample_df):
        """Verify null/NaN values are detected."""
        null_df = valid_sample_df.copy()
        null_df.loc[0, "Rotational speed [rpm]"] = np.nan
        is_valid, errors = validate_schema(null_df)
        assert is_valid is False
        assert any("null values" in err for err in errors)

    def test_invalid_categorical_type(self, valid_sample_df):
        """Verify invalid machine Type category is rejected."""
        bad_type_df = valid_sample_df.copy()
        bad_type_df.loc[0, "Type"] = "X"
        is_valid, errors = validate_schema(bad_type_df)
        assert is_valid is False
        assert any("Invalid Type values" in err for err in errors)

    def test_invalid_physical_sensor_bounds(self, valid_sample_df):
        """Verify negative or impossible sensor values are caught."""
        # Non-positive temperature (Kelvin)
        bad_temp_df = valid_sample_df.copy()
        bad_temp_df.loc[0, "Air temperature [K]"] = -5.0
        is_valid, errors = validate_schema(bad_temp_df)
        assert is_valid is False
        assert any("Air temperature" in err for err in errors)

        # Negative torque
        bad_torque_df = valid_sample_df.copy()
        bad_torque_df.loc[0, "Torque [Nm]"] = -10.0
        is_valid, errors = validate_schema(bad_torque_df)
        assert is_valid is False
        assert any("Torque" in err for err in errors)

        # Negative tool wear
        bad_wear_df = valid_sample_df.copy()
        bad_wear_df.loc[0, "Tool wear [min]"] = -1.0
        is_valid, errors = validate_schema(bad_wear_df)
        assert is_valid is False
        assert any("Tool wear" in err for err in errors)
