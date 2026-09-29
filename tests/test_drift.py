"""
Tests for input drift (KS, PSI), prediction drift (Wasserstein), and alert logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.drift import (
    calculate_psi,
    run_ks_test,
    calculate_input_drift,
    calculate_prediction_drift,
)


class TestDriftDetection:
    """Validate statistical drift metrics and severity thresholds."""

    def test_psi_identical_distributions(self):
        """Verify PSI is near zero for identical distributions."""
        rng = np.random.RandomState(42)
        ref = rng.normal(loc=100.0, scale=15.0, size=2000)
        curr = ref.copy()

        psi = calculate_psi(ref, curr)
        assert psi == pytest.approx(0.0, abs=1e-3)

    def test_psi_shifted_distribution(self):
        """Verify PSI exceeds alert threshold (>0.25) under substantial shift."""
        rng = np.random.RandomState(42)
        ref = rng.normal(loc=100.0, scale=15.0, size=2000)
        curr = rng.normal(loc=120.0, scale=15.0, size=2000)  # +1.33 standard deviations

        psi = calculate_psi(ref, curr)
        assert psi > 0.25, f"Expected alert PSI > 0.25, got {psi:.4f}"

    def test_ks_test_statistical_properties(self):
        """Verify KS statistic is in [0, 1] and detects distribution discrepancy."""
        rng = np.random.RandomState(42)
        sample_a = rng.uniform(0, 10, size=500)
        sample_b = rng.uniform(0, 10, size=500)
        sample_shifted = rng.uniform(5, 15, size=500)

        # Same distribution: low D, p-value not near zero
        res_same = run_ks_test(sample_a, sample_b)
        assert 0.0 <= res_same["statistic"] <= 1.0
        assert res_same["p_value"] > 0.01

        # Shifted distribution: high D, p-value very small
        res_shift = run_ks_test(sample_a, sample_shifted)
        assert res_shift["statistic"] > 0.4
        assert res_shift["p_value"] < 1e-10

    def test_calculate_input_drift_table_and_severity_logic(self):
        """Verify input drift table generates correct schema and severity tags."""
        rng = np.random.RandomState(42)
        ref_df = pd.DataFrame({
            "feat_ok": rng.normal(0, 1, 1000),
            "feat_alert": rng.normal(0, 1, 1000),
        })
        curr_df = pd.DataFrame({
            "feat_ok": rng.normal(0, 1, 1000),
            "feat_alert": rng.normal(3, 1, 1000),  # Major shift
        })

        drift_table = calculate_input_drift(ref_df, curr_df, feature_cols=["feat_ok", "feat_alert"])

        assert list(drift_table.columns) == ["Feature", "KS_Statistic", "KS_P_Value", "PSI", "Severity"]
        assert len(drift_table) == 2

        ok_row = drift_table[drift_table["Feature"] == "feat_ok"].iloc[0]
        alert_row = drift_table[drift_table["Feature"] == "feat_alert"].iloc[0]

        assert ok_row["Severity"] == "OK"
        assert ok_row["PSI"] < 0.10

        assert alert_row["Severity"] == "ALERT"
        assert alert_row["PSI"] > 0.25

    def test_prediction_drift_wasserstein_and_ks(self):
        """Verify prediction drift detects probability distribution displacement."""
        p_ref = np.linspace(0.01, 0.20, 1000)
        p_curr_same = p_ref.copy()
        p_curr_shifted = p_ref + 0.15  # Uniform displacement

        # Identical
        res_same = calculate_prediction_drift(p_ref, p_curr_same)
        assert res_same["wasserstein_distance"] == pytest.approx(0.0, abs=1e-5)
        assert res_same["ks_statistic"] == pytest.approx(0.0, abs=1e-5)

        # Shifted
        res_shift = calculate_prediction_drift(p_ref, p_curr_shifted)
        assert res_shift["wasserstein_distance"] == pytest.approx(0.15, abs=1e-3)
        assert res_shift["ks_statistic"] > 0.5
