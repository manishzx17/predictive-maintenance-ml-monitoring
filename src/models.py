"""
Model training, probability calibration, and uncertainty estimation.
Preserves stable, validated logic from Phases 5-8.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
import lightgbm as lgb

# IMPORTANT ARTIFACT NOTE:
# q_hat = 0.99923 is the empirical nonconformity score quantile computed during Phase 8
# on the clean validation set for this exact model, feature set, and calibration setup at alpha = 0.10.
# It is an empirical artifact for this pipeline setup, NOT a universal mathematical constant.
PHASE8_CALIBRATED_Q_HAT: float = 0.99923

# Phase 7 Cost-Optimal Operational Decision Threshold
COST_OPTIMAL_THRESHOLD: float = 0.15


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
    **kwargs: Any,
) -> lgb.LGBMClassifier:
    """
    Train the Champion LightGBM classifier using validated Phase 5 hyperparameters.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training feature matrix.
    y_train : np.ndarray
        Training binary target labels.
    random_seed : int
        Deterministic random seed (default 42).
    **kwargs : Any
        Optional hyperparameter overrides.

    Returns
    -------
    lgb.LGBMClassifier
        Trained base LightGBM classifier.
    """
    params = {
        "learning_rate": 0.05,
        "n_estimators": 100,
        "num_leaves": 15,
        "scale_pos_weight": 15.0,
        "random_state": random_seed,
        "verbose": -1,
        "n_jobs": 1,
    }
    params.update(kwargs)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def calibrate_model(
    base_estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling (sigmoid calibration) with k-fold cross-validation on training data.
    Preserves Phase 6 calibration methodology.

    Parameters
    ----------
    base_estimator : Any
        Trained or uncalibrated classifier.
    X_train : np.ndarray
        Preprocessed training feature matrix.
    y_train : np.ndarray
        Training binary target labels.
    cv : int
        Number of cross-validation folds (default 5).
    method : str
        Calibration method: 'sigmoid' (Platt scaling) or 'isotonic'.

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated classifier.
    """
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def calculate_ece(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    n_bins: int = 10,
) -> float:
    """
    Compute 10-bin equal-width Expected Calibration Error (ECE).
    Matches the exact formulation from Phases 6, 13, and 14.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0 or 1).
    y_prob : array-like
        Calibrated failure probabilities in [0, 1].
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE metric value.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    N = len(y_t)
    if N == 0:
        return 0.0

    for k in range(n_bins):
        mask = bin_indices == k
        nk = np.sum(mask)
        if nk > 0:
            pk_bar = np.mean(y_p[mask])
            yk_bar = np.mean(y_t[mask])
            ece += (nk / N) * np.abs(pk_bar - yk_bar)
    return float(ece)


def predict_with_threshold(
    y_prob: np.ndarray,
    threshold: float = COST_OPTIMAL_THRESHOLD,
) -> np.ndarray:
    """
    Apply operational decision threshold to emit binary classifications.
    Default is Phase 7 cost-optimal threshold (t* = 0.15).

    Parameters
    ----------
    y_prob : np.ndarray
        Emitted failure probabilities.
    threshold : float
        Decision threshold (default 0.15).

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1).
    """
    return (np.asarray(y_prob) >= threshold).astype(int)


def calibrate_conformal_aps(
    y_val_prob: np.ndarray,
    y_val_true: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Compute conformal nonconformity score quantile q_hat using split-conformal APS.
    For binary classification, APS score is cumulative probability mass required
    to include the true class y_i:
    - If true class is top predicted class: s_i = max(p0, p1)
    - If true class is second predicted class: s_i = p0 + p1 = 1.0
    q_hat is computed at min(1.0, ceiling((n + 1) * (1 - alpha)) / n) quantile.

    Parameters
    ----------
    y_val_prob : np.ndarray
        Predicted probabilities for class 1 on calibration set.
    y_val_true : np.ndarray
        True binary labels on calibration set.
    alpha : float
        Significance level (default 0.10 for 90% coverage).

    Returns
    -------
    float
        Calibrated nonconformity quantile q_hat.
    """
    p1 = np.asarray(y_val_prob)
    p0 = 1.0 - p1
    y_t = np.asarray(y_val_true).astype(int)

    probs = np.column_stack([p0, p1])
    n = len(y_t)
    scores = np.empty(n, dtype=float)

    for i in range(n):
        true_class = y_t[i]
        row_probs = probs[i]
        order = np.argsort(-row_probs)
        cum_probs = np.cumsum(row_probs[order])
        rank = np.where(order == true_class)[0][0]
        scores[i] = cum_probs[rank]

    q_val = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat = float(np.quantile(scores, q_val, method="higher"))
    return q_hat


def predict_conformal_sets(
    y_prob: np.ndarray,
    q_hat: float = PHASE8_CALIBRATED_Q_HAT,
) -> Dict[str, np.ndarray]:
    """
    Generate split-conformal Adaptive Prediction Sets (APS) for each instance.
    For binary classification:
    - Include class 0 if p0 >= 1 - q_hat (or equivalently max(p0, p1) >= q_hat for singletons)
    - If both classes included, prediction set is {No Failure, Machine Failure} (Ambiguous)

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted failure probabilities.
    q_hat : float
        Calibrated quantile threshold (default: Phase 8 artifact 0.99923).

    Returns
    -------
    Dict[str, np.ndarray]
        Boolean masks: 'singleton_0', 'singleton_1', 'ambiguous'.
    """
    p1 = np.asarray(y_prob)
    p0 = 1.0 - p1

    # In binary APS with q_hat:
    # A class c is included if its cumulative probability reaches the quantile.
    # An observation is a singleton if max(p0, p1) >= q_hat.
    max_p = np.maximum(p0, p1)
    singleton_0 = (p0 >= q_hat)
    singleton_1 = (p1 >= q_hat)
    ambiguous = (max_p < q_hat)

    return {
        "singleton_0": singleton_0,
        "singleton_1": singleton_1,
        "ambiguous": ambiguous,
    }


def evaluate_conformal_aps(
    y_prob: np.ndarray,
    q_hat: float = PHASE8_CALIBRATED_Q_HAT,
    y_true: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, float]:
    """
    Evaluate conformal uncertainty metrics across a batch.
    Computes set-size proportions (ambiguous rate, singleton rates).
    Computes empirical coverage and mean set size ONLY if y_true is supplied.

    Parameters
    ----------
    y_prob : np.ndarray
        Calibrated failure probabilities.
    q_hat : float
        Calibrated nonconformity quantile (default: Phase 8 artifact 0.99923).
    y_true : Optional[array-like]
        Ground truth labels (optional).

    Returns
    -------
    Dict[str, float]
        Uncertainty metrics: 'ambiguous_rate', 'singleton_0_rate', 'singleton_1_rate',
        and optionally 'empirical_coverage' and 'mean_set_size'.
    """
    sets = predict_conformal_sets(y_prob, q_hat=q_hat)
    n = len(y_prob)

    res = {
        "ambiguous_rate": float(np.mean(sets["ambiguous"]) * 100.0) if n > 0 else 0.0,
        "singleton_0_rate": float(np.mean(sets["singleton_0"]) * 100.0) if n > 0 else 0.0,
        "singleton_1_rate": float(np.mean(sets["singleton_1"]) * 100.0) if n > 0 else 0.0,
    }

    if y_true is not None:
        y_t = np.asarray(y_true).astype(int)
        # Covered if true class is in prediction set:
        # If ambiguous, both classes in set -> covered = True
        # If singleton_0, covered if y_true == 0
        # If singleton_1, covered if y_true == 1
        covered = (
            sets["ambiguous"]
            | (sets["singleton_0"] & (y_t == 0))
            | (sets["singleton_1"] & (y_t == 1))
        )
        res["empirical_coverage"] = float(np.mean(covered) * 100.0)
        # Set size: ambiguous = 2, singleton = 1
        set_sizes = np.where(sets["ambiguous"], 2, 1)
        res["mean_set_size"] = float(np.mean(set_sizes))

    return res
