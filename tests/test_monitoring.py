"""
Tests for integrated monitoring orchestration, outcome classification, and test set isolation.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import load_and_split_data
from src.app import (
    classify_monitoring_outcome,
    generate_monitoring_report,
)
from src.models import PHASE8_CALIBRATED_Q_HAT


class TestMonitoringEngine:
    """Validate integrated monitoring report generation and operational outcome assignment."""

    def test_outcome_classification_rules(self):
        """Verify hierarchical decision rules for all operational states."""
        # 1. Normal
        out_normal, _ = classify_monitoring_outcome(max_psi=0.05, w_dist=0.01, ece=0.01, ambig_rate=90.0)
        assert out_normal == "NORMAL"

        # 2. Warning
        out_warn, _ = classify_monitoring_outcome(max_psi=0.15, w_dist=0.01, ece=0.01, ambig_rate=90.0)
        assert out_warn == "WARNING"

        # 3. Drift Alert (PSI > 0.25)
        out_alert_psi, _ = classify_monitoring_outcome(max_psi=0.35, w_dist=0.01, ece=0.05, ambig_rate=90.0)
        assert out_alert_psi == "DRIFT_ALERT"

        # 4. Drift Alert (Wasserstein >= 0.05)
        out_alert_w, _ = classify_monitoring_outcome(max_psi=0.05, w_dist=0.06, ece=0.05, ambig_rate=90.0)
        assert out_alert_w == "DRIFT_ALERT"

        # 5. Human Inspection (Severe ECE > 0.10 override)
        out_insp_ece, _ = classify_monitoring_outcome(max_psi=0.50, w_dist=0.10, ece=0.12, ambig_rate=90.0)
        assert out_insp_ece == "HUMAN_INSPECTION"

        # 6. Human Inspection (High Conformal Ambiguity >= 95.0% override)
        out_insp_ambig, _ = classify_monitoring_outcome(max_psi=0.05, w_dist=0.01, ece=0.01, ambig_rate=96.5)
        assert out_insp_ambig == "HUMAN_INSPECTION"

    def test_supervised_monitoring_report(self, trained_pipeline, reference_batch):
        """Verify report generation when labels are supplied (computes ECE, Brier, and coverage)."""
        champion_model = trained_pipeline["champion_model"]
        preprocessor = trained_pipeline["preprocessor"]

        report = generate_monitoring_report(
            ref_df=reference_batch,
            curr_df=reference_batch,
            model=champion_model,
            preprocessor=preprocessor,
            q_hat=PHASE8_CALIBRATED_Q_HAT,
            label_col="Machine failure",
        )

        assert report["is_supervised"] is True
        assert report["decision"]["outcome"] == "NORMAL"
        assert report["decision"]["escalation_required"] is False
        assert report["reliability"]["10_bin_ece"] is not None
        assert report["reliability"]["brier_score"] is not None
        assert report["conformal_uncertainty"]["empirical_coverage"] is not None

    def test_unsupervised_monitoring_report(self, trained_pipeline, reference_batch):
        """Verify report generation when labels are absent (skips ECE/Brier, still evaluates drift)."""
        champion_model = trained_pipeline["champion_model"]
        preprocessor = trained_pipeline["preprocessor"]

        unlabeled_batch = reference_batch.drop(columns=["Machine failure"])

        report = generate_monitoring_report(
            ref_df=reference_batch,
            curr_df=unlabeled_batch,
            model=champion_model,
            preprocessor=preprocessor,
            q_hat=PHASE8_CALIBRATED_Q_HAT,
            y_true=None,
        )

        assert report["is_supervised"] is False
        assert report["decision"]["outcome"] == "NORMAL"
        assert report["reliability"]["10_bin_ece"] is None
        assert report["reliability"]["brier_score"] is None
        assert report["conformal_uncertainty"]["empirical_coverage"] is None
        # Unsupervised drift and uncertainty metrics still computed
        assert report["input_drift"]["max_feature_psi"] is not None
        assert report["prediction_drift"]["wasserstein_distance"] is not None
        assert report["conformal_uncertainty"]["ambiguous_rate"] is not None

    def test_drift_scenario_escalations(self, trained_pipeline, reference_batch, scenario_b_batch, scenario_c_batch):
        """Verify realistic scenario escalations on Phase 10 synthetic drift batches."""
        champion_model = trained_pipeline["champion_model"]
        preprocessor = trained_pipeline["preprocessor"]

        # Scenario B (Torque shift) -> DRIFT_ALERT
        rep_b = generate_monitoring_report(
            ref_df=reference_batch,
            curr_df=scenario_b_batch,
            model=champion_model,
            preprocessor=preprocessor,
            label_col="Machine failure",
        )
        assert rep_b["decision"]["outcome"] == "DRIFT_ALERT"
        assert rep_b["decision"]["escalation_required"] is True

        # Scenario C (Variance expansion) -> HUMAN_INSPECTION (ECE > 0.10)
        rep_c = generate_monitoring_report(
            ref_df=reference_batch,
            curr_df=scenario_c_batch,
            model=champion_model,
            preprocessor=preprocessor,
            label_col="Machine failure",
        )
        assert rep_c["decision"]["outcome"] == "HUMAN_INSPECTION"
        assert rep_c["decision"]["escalation_required"] is True


class TestTestSetQuarantineBehavior:
    """Verify behavioral quarantine of the test set through actual code execution."""

    def test_default_data_loader_never_returns_test_split(self):
        """Verify load_and_split_data strictly withholds test set by default."""
        splits = load_and_split_data("data/raw/ai4i2020.csv", return_test=False)
        # Length of returned tuple is exactly 2: ((X_train, y_train), (X_val, y_val))
        assert len(splits) == 2

        (X_tr, y_tr), (X_val, y_val) = splits
        assert len(X_tr) == 7000
        assert len(X_val) == 1500

        # Verify no test data was returned or leaked in the outputs
        total_rows = len(X_tr) + len(X_val)
        assert total_rows == 8500  # 1500 test rows completely omitted

    def test_pipeline_execution_without_test_split(self, trained_pipeline, reference_batch):
        """Verify end-to-end monitoring runs exclusively on validation-derived batches without test data."""
        champion_model = trained_pipeline["champion_model"]
        preprocessor = trained_pipeline["preprocessor"]

        # Run report on reference baseline
        report = generate_monitoring_report(
            ref_df=reference_batch,
            curr_df=reference_batch,
            model=champion_model,
            preprocessor=preprocessor,
            y_true=reference_batch["Machine failure"],
        )

        assert report["decision"]["outcome"] == "NORMAL"
        assert report["batch_size"] == 1500
