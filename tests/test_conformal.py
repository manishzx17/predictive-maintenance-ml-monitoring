"""
Tests for split-conformal Adaptive Prediction Sets (APS) and coverage calculation.
"""

import pytest
import numpy as np

from src.models import (
    calibrate_conformal_aps,
    predict_conformal_sets,
    evaluate_conformal_aps,
    PHASE8_CALIBRATED_Q_HAT,
)


class TestConformalPrediction:
    """Validate conformal set generation, ambiguity rates, and coverage computation."""

    def test_predict_conformal_sets_masks(self):
        """Verify indicator masks for singletons and ambiguous sets partition predictions."""
        # Setup specific test probabilities:
        # p0 = 1 - p1
        # If p0 >= q_hat (p1 <= 1 - q_hat), singleton_0
        # If p1 >= q_hat, singleton_1
        # If max(p0, p1) < q_hat, ambiguous
        q_hat = 0.90
        probs = np.array([
            0.05,  # p0 = 0.95 >= 0.90 -> singleton_0
            0.95,  # p1 = 0.95 >= 0.90 -> singleton_1
            0.50,  # p0 = 0.50, p1 = 0.50 -> max = 0.50 < 0.90 -> ambiguous
        ])

        sets = predict_conformal_sets(probs, q_hat=q_hat)

        np.testing.assert_array_equal(sets["singleton_0"], [True, False, False])
        np.testing.assert_array_equal(sets["singleton_1"], [False, True, False])
        np.testing.assert_array_equal(sets["ambiguous"], [False, False, True])

        # Every instance must be classified into exactly one category
        total_counts = (
            sets["singleton_0"].astype(int)
            + sets["singleton_1"].astype(int)
            + sets["ambiguous"].astype(int)
        )
        np.testing.assert_array_equal(total_counts, [1, 1, 1])

    def test_evaluate_conformal_without_labels(self):
        """Verify evaluation without ground truth computes rates and leaves coverage as None."""
        probs = np.array([0.05, 0.95, 0.50, 0.40])
        res = evaluate_conformal_aps(probs, q_hat=0.90, y_true=None)

        assert res["ambiguous_rate"] == 50.0  # 2 out of 4 are ambiguous (0.50, 0.40)
        assert res["singleton_0_rate"] == 25.0
        assert res["singleton_1_rate"] == 25.0
        assert res.get("empirical_coverage") is None
        assert res.get("mean_set_size") is None

    def test_evaluate_conformal_with_labels_coverage(self):
        """Verify empirical coverage and mean set size are calculated when y_true is supplied."""
        q_hat = 0.90
        # 4 instances:
        # 0: p=0.05 (singleton 0), true=0 -> covered
        # 1: p=0.05 (singleton 0), true=1 -> NOT covered (misclassification)
        # 2: p=0.50 (ambiguous {0, 1}), true=0 -> covered (both in set)
        # 3: p=0.95 (singleton 1), true=1 -> covered
        probs = np.array([0.05, 0.05, 0.50, 0.95])
        y_true = np.array([0, 1, 0, 1])

        res = evaluate_conformal_aps(probs, q_hat=q_hat, y_true=y_true)

        # 3 out of 4 covered -> 75%
        assert res["empirical_coverage"] == 75.0
        # Set sizes: 1, 1, 2, 1 -> mean = 5/4 = 1.25
        assert res["mean_set_size"] == 1.25

    def test_calibrate_conformal_aps_quantile(self):
        """Verify conformal quantile calibration function returns valid nonconformity threshold."""
        y_val_prob = np.array([0.1, 0.2, 0.05, 0.9, 0.85, 0.02, 0.03, 0.01, 0.95, 0.04])
        y_val_true = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0])

        q_hat = calibrate_conformal_aps(y_val_prob, y_val_true, alpha=0.10)

        assert 0.0 <= q_hat <= 1.0
        # At 90% target coverage, q_hat should be high
        assert q_hat > 0.80
