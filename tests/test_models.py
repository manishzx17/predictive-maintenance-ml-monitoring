"""
Tests for model predictions, probability calibration, ECE, and decision thresholding.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import create_features
from src.models import (
    train_models,
    calibrate_model,
    calculate_ece,
    predict_with_threshold,
    COST_OPTIMAL_THRESHOLD,
)


class TestModelTrainingAndCalibration:
    """Validate model probability output properties and calibration."""

    def test_model_predictions_and_probability_bounds(self, trained_pipeline, train_val_data):
        """Verify model predictions exist and output probabilities are strictly in [0, 1]."""
        preprocessor = trained_pipeline["preprocessor"]
        champion_model = trained_pipeline["champion_model"]

        X_val = train_val_data["X_val"]
        X_val_feat = create_features(X_val)
        X_val_proc = preprocessor.transform(X_val_feat)

        probs = champion_model.predict_proba(X_val_proc)

        # Output shape matches validation size
        assert probs.shape == (len(X_val), 2)

        # Each row sums to 1.0
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

        # Probabilities are strictly bounded in [0, 1]
        p_failure = probs[:, 1]
        assert np.all(p_failure >= 0.0), "Negative probability detected"
        assert np.all(p_failure <= 1.0), "Probability > 1.0 detected"

    def test_ece_calculation_behavior(self):
        """Verify 10-bin ECE metric computation under known toy cases."""
        # 1. Perfectly calibrated distribution
        y_true_perfect = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob_perfect = np.array([0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95])
        ece_perfect = calculate_ece(y_true_perfect, y_prob_perfect, n_bins=10)
        assert ece_perfect < 0.06

        # 2. Severely uncalibrated (overconfident wrong predictions)
        y_true_bad = np.zeros(100)
        y_prob_bad = np.ones(100) * 0.95  # Model predicts 95% failure, true is 0%
        ece_bad = calculate_ece(y_true_bad, y_prob_bad, n_bins=10)
        assert ece_bad == pytest.approx(0.95, abs=1e-3)

        # 3. Empty input handling
        assert calculate_ece([], []) == 0.0

    def test_cost_optimal_thresholding(self):
        """Verify binary predictions emitted at operational threshold (t* = 0.15)."""
        probs = np.array([0.05, 0.149, 0.150, 0.151, 0.85])
        preds = predict_with_threshold(probs, threshold=COST_OPTIMAL_THRESHOLD)

        expected = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(preds, expected)
