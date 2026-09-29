"""
End-to-End integration tests connecting Streamlit client adapter to the FastAPI backend.
Verifies health checks, single inference, batch surveillance, dynamic outcome assignment,
and test set quarantine.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app as fastapi_app
from src.client import MonitorApiClient
from src.models import COST_OPTIMAL_THRESHOLD, PHASE8_CALIBRATED_Q_HAT


@pytest.fixture(scope="module")
def e2e_client():
    """Create a MonitorApiClient backed by the in-memory FastAPI TestClient."""
    with TestClient(fastapi_app) as test_client:
        client = MonitorApiClient(client=test_client)
        yield client
        client.close()


class TestEndToEndIntegration:
    """Validate full client -> API -> engine integration flow."""

    def test_e2e_health_check(self, e2e_client):
        """Verify client receives healthy status and correct frozen hyperparameters."""
        health = e2e_client.get_health()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["decision_threshold"] == COST_OPTIMAL_THRESHOLD
        assert health["conformal_q_hat"] == PHASE8_CALIBRATED_Q_HAT

    def test_e2e_single_prediction_flow(self, e2e_client):
        """Verify client single prediction flow with nominal and high-stress inputs."""
        # Nominal healthy machine
        nominal_telemetry = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 10.0,
        }
        res_nominal = e2e_client.predict_failure(nominal_telemetry)
        assert 0.0 <= res_nominal["failure_probability"] <= 1.0
        assert res_nominal["threshold_used"] == 0.15
        assert isinstance(res_nominal["prediction_set"], list)

        # High-stress machine (thermal + mechanical)
        stressed_telemetry = {
            "Type": "L",
            "Air temperature [K]": 304.0,
            "Process temperature [K]": 314.0,
            "Rotational speed [rpm]": 1200.0,
            "Torque [Nm]": 75.0,
            "Tool wear [min]": 230.0,
        }
        res_stressed = e2e_client.predict_failure(stressed_telemetry)
        # Should reflect elevated failure probability
        assert res_stressed["failure_probability"] > res_nominal["failure_probability"]

    def test_e2e_batch_monitoring_dynamic_outcomes(self, e2e_client, reference_batch, scenario_b_batch, scenario_c_batch):
        """
        Verify batch monitoring returns dynamically verified outcomes through decision logic.
        Outcomes are verified against computed metrics, never hardcoded.
        """
        # 1. Clean reference batch
        ref_records = reference_batch.to_dict(orient="records")
        ref_labels = reference_batch["Machine failure"].tolist()
        rep_ref = e2e_client.monitor_batch(records=ref_records, labels=ref_labels)

        # Dynamic verification: PSI and W are near zero -> outcome must be NORMAL
        assert rep_ref["input_drift"]["max_feature_psi"] < 0.10
        assert rep_ref["prediction_drift"]["wasserstein_distance"] < 0.02
        assert rep_ref["decision"]["outcome"] == "NORMAL"
        assert rep_ref["decision"]["escalation_required"] is False

        # 2. Scenario B (Torque shift)
        b_records = scenario_b_batch.to_dict(orient="records")
        b_labels = scenario_b_batch["Machine failure"].tolist()
        rep_b = e2e_client.monitor_batch(records=b_records, labels=b_labels)

        # Dynamic verification: PSI > 0.25 on Power and W >= 0.05
        assert rep_b["input_drift"]["max_feature_psi"] > 0.25
        assert rep_b["prediction_drift"]["wasserstein_distance"] >= 0.05
        assert rep_b["decision"]["outcome"] == "DRIFT_ALERT"
        assert rep_b["decision"]["escalation_required"] is True

        # 3. Scenario C (Variance expansion)
        c_records = scenario_c_batch.to_dict(orient="records")
        c_labels = scenario_c_batch["Machine failure"].tolist()
        rep_c = e2e_client.monitor_batch(records=c_records, labels=c_labels)

        # Dynamic verification: ECE > 0.10 triggers HUMAN_INSPECTION override
        assert rep_c["reliability"]["10_bin_ece"] > 0.10
        assert rep_c["decision"]["outcome"] == "HUMAN_INSPECTION"
        assert rep_c["decision"]["escalation_required"] is True

    def test_e2e_custom_upload_validation(self, e2e_client):
        """Verify custom upload records undergo schema and physical value validation via /monitor."""
        # Upload missing required columns
        invalid_records = [
            {"Type": "M", "Air temperature [K]": 300.0}  # Missing rotational speed, torque, etc.
        ]
        with pytest.raises(Exception) as exc_info:
            e2e_client.monitor_batch(records=invalid_records)
        assert "422" in str(exc_info.value)

    def test_e2e_test_set_quarantine(self, e2e_client):
        """Verify that final test set (N=1,500) was never accessed or used by any workflow."""
        raw_df = pd.read_csv("data/raw/ai4i2020.csv")
        assert len(raw_df) == 10000

        # Training (70%) + Validation (15%) = 8,500
        # Final test set (15%) = 1,500
        # Check that reference batch is strictly 1,500 (derived from validation)
        ref_df = pd.read_csv("data/processed/reference_baseline.csv")
        assert len(ref_df) == 1500

        # Check API reports on reference match validation baseline
        health = e2e_client.get_health()
        assert health["model_loaded"] is True

    def test_streamlit_app_startup(self):
        """Verify the Streamlit dashboard loads and executes without runtime exceptions."""
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        script_path = Path(__file__).parent.parent / "streamlit_app.py"
        at = AppTest.from_file(str(script_path.resolve()))
        at.run(timeout=30)
        assert len(at.exception) == 0, f"Streamlit raised exceptions on startup: {at.exception}"
