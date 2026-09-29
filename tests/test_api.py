"""
Automated tests for FastAPI endpoints (/health, /predict, /monitor).
Uses FastAPI TestClient and pytest to verify API behavior, schemas, and drift surveillance.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.models import COST_OPTIMAL_THRESHOLD, PHASE8_CALIBRATED_Q_HAT


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing the FastAPI application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Validate system health and readiness endpoint."""

    def test_health_check_success(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["pipeline_version"] == "1.0.0"
        assert data["decision_threshold"] == COST_OPTIMAL_THRESHOLD
        assert data["conformal_q_hat"] == PHASE8_CALIBRATED_Q_HAT


class TestPredictEndpoint:
    """Validate real-time inference, Pydantic validation, and frozen conformal set prediction."""

    def test_predict_nominal_telemetry(self, client):
        """Verify inference on a healthy machine returns valid probability and decision."""
        payload = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert 0.0 <= data["failure_probability"] <= 1.0
        assert data["prediction"] in (0, 1)
        assert data["threshold_used"] == COST_OPTIMAL_THRESHOLD
        assert isinstance(data["prediction_set"], list)
        assert len(data["prediction_set"]) >= 1
        assert isinstance(data["is_ambiguous"], bool)

    def test_predict_snake_case_alias_support(self, client):
        """Verify Pydantic schema accepts snake_case aliases."""
        payload = {
            "Type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1400.0,
            "torque_nm": 45.0,
            "tool_wear_min": 150.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "failure_probability" in data

    def test_predict_invalid_telemetry_validation(self, client):
        """Verify strict Pydantic validation rejects invalid types and physically impossible values."""
        # 1. Invalid Type category ('Z' instead of 'L', 'M', 'H')
        bad_type = {
            "Type": "Z",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0.0,
        }
        resp_bad_type = client.post("/predict", json=bad_type)
        assert resp_bad_type.status_code == 422

        # 2. Negative torque (physically impossible)
        bad_torque = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": -5.0,
            "Tool wear [min]": 0.0,
        }
        resp_bad_torque = client.post("/predict", json=bad_torque)
        assert resp_bad_torque.status_code == 422

        # 3. Non-positive temperature (in Kelvin)
        bad_temp = {
            "Type": "M",
            "Air temperature [K]": 0.0,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 0.0,
        }
        resp_bad_temp = client.post("/predict", json=bad_temp)
        assert resp_bad_temp.status_code == 422


class TestMonitorEndpoint:
    """Validate batch telemetry monitoring and dynamic operational decision rules."""

    def test_monitor_nominal_batch(self, client, reference_batch):
        """Verify monitoring report on clean reference batch reflects in-control baseline."""
        records = reference_batch.to_dict(orient="records")
        labels = reference_batch["Machine failure"].tolist()

        payload = {
            "records": records,
            "labels": labels,
        }
        response = client.post("/monitor", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == len(reference_batch)
        assert data["is_supervised"] is True
        assert data["input_drift"]["max_feature_psi"] == pytest.approx(0.0, abs=1e-3)
        assert data["prediction_drift"]["wasserstein_distance"] == pytest.approx(0.0, abs=1e-4)

        # Verify outcome is assigned strictly by existing decision rules (not hardcoded)
        assert data["decision"]["outcome"] == "NORMAL"
        assert data["decision"]["escalation_required"] is False
        assert data["reliability"]["10_bin_ece"] is not None

    def test_monitor_unlabeled_batch(self, client, reference_batch):
        """Verify monitoring on unlabeled batch operates in unsupervised mode."""
        unlabeled_df = reference_batch.drop(columns=["Machine failure"])
        records = unlabeled_df.to_dict(orient="records")

        payload = {
            "records": records,
            "labels": None,
        }
        response = client.post("/monitor", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["is_supervised"] is False
        assert data["reliability"]["10_bin_ece"] is None
        assert data["reliability"]["brier_score"] is None
        # Drift and conformal metrics are still evaluated
        assert "max_feature_psi" in data["input_drift"]
        assert "wasserstein_distance" in data["prediction_drift"]
        assert "ambiguous_rate" in data["conformal_uncertainty"]
        assert data["decision"]["outcome"] == "NORMAL"

    def test_monitor_drifted_batch_escalation(self, client, scenario_b_batch):
        """Verify that a drifted batch triggers the appropriate alert dynamically."""
        records = scenario_b_batch.to_dict(orient="records")
        labels = scenario_b_batch["Machine failure"].tolist()

        payload = {
            "records": records,
            "labels": labels,
        }
        response = client.post("/monitor", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Scenario B has PSI > 0.25 on Power and Wasserstein >= 0.05
        assert data["input_drift"]["max_feature_psi"] > 0.25
        assert data["decision"]["escalation_required"] is True
        # Verified dynamically through decision logic
        assert data["decision"]["outcome"] in ("DRIFT_ALERT", "HUMAN_INSPECTION")

    def test_monitor_mismatched_label_count(self, client, reference_batch):
        """Verify error is raised if label length does not match record count."""
        records = reference_batch.head(50).to_dict(orient="records")
        labels = [0, 1]  # Only 2 labels for 50 records

        payload = {
            "records": records,
            "labels": labels,
        }
        response = client.post("/monitor", json=payload)
        assert response.status_code == 422
