"""
Monitoring orchestration engine and report generator.
Preserves stable, validated logic from Phase 14.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.features import create_features, ENGINEERED_NUMERICAL_COLS
from src.models import (
    calculate_ece,
    evaluate_conformal_aps,
    PHASE8_CALIBRATED_Q_HAT,
)
from src.drift import (
    calculate_input_drift,
    calculate_prediction_drift,
)

# Operational Heuristics (Project-Specific):
# Note: These thresholds are operational engineering heuristics tailored to this predictive
# maintenance pipeline, not universal statistical laws.
INPUT_PSI_ALERT_THRESHOLD: float = 0.25
PRED_W_ALERT_THRESHOLD: float = 0.05
INPUT_PSI_WARN_THRESHOLD: float = 0.10
PRED_W_WARN_THRESHOLD: float = 0.02

# Reliability & Uncertainty Safety Override Thresholds:
CONFORMAL_AMBIGUITY_THRESHOLD: float = 95.0  # Percentage
CALIBRATION_ECE_THRESHOLD: float = 0.10


def classify_monitoring_outcome(
    max_psi: float,
    w_dist: float,
    ece: Optional[float] = None,
    ambig_rate: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Hierarchical decision rules assigning exactly one outcome:
    NORMAL | WARNING | DRIFT_ALERT | HUMAN_INSPECTION

    Parameters
    ----------
    max_psi : float
        Maximum feature PSI across monitored input features.
    w_dist : float
        1-Wasserstein distance between reference and current predicted probabilities.
    ece : Optional[float]
        10-bin ECE (supplied in supervised mode; None if unsupervised).
    ambig_rate : Optional[float]
        Conformal prediction set ambiguity rate in percent (e.g., 92.5).

    Returns
    -------
    Tuple[str, str]
        Consolidated outcome name and operational justification text.
    """
    # Rule 1: Human Inspection Override (Safety / Reliability / Uncertainty Collapse)
    if ece is not None and ece > CALIBRATION_ECE_THRESHOLD:
        return (
            "HUMAN_INSPECTION",
            f"Severe probability calibration breakdown (ECE = {ece:.4f} > {CALIBRATION_ECE_THRESHOLD})",
        )

    if ambig_rate is not None and ambig_rate >= CONFORMAL_AMBIGUITY_THRESHOLD:
        return (
            "HUMAN_INSPECTION",
            f"High conformal ambiguity rate ({ambig_rate:.2f}% >= {CONFORMAL_AMBIGUITY_THRESHOLD}%)",
        )

    # Rule 2: Drift Alert (Project-Specific Operational Heuristic: PSI > 0.25 or W >= 0.05)
    if max_psi > INPUT_PSI_ALERT_THRESHOLD or w_dist >= PRED_W_ALERT_THRESHOLD:
        reasons = []
        if max_psi > INPUT_PSI_ALERT_THRESHOLD:
            reasons.append(f"Input PSI > {INPUT_PSI_ALERT_THRESHOLD} ({max_psi:.2f})")
        if w_dist >= PRED_W_ALERT_THRESHOLD:
            reasons.append(f"Pred W >= {PRED_W_ALERT_THRESHOLD} ({w_dist:.4f})")
        return "DRIFT_ALERT", "; ".join(reasons)

    # Rule 3: Warning (Moderate Shift / Early Telemetry Warning)
    if (INPUT_PSI_WARN_THRESHOLD <= max_psi <= INPUT_PSI_ALERT_THRESHOLD) or (
        PRED_W_WARN_THRESHOLD <= w_dist < PRED_W_ALERT_THRESHOLD
    ):
        reasons = []
        if INPUT_PSI_WARN_THRESHOLD <= max_psi <= INPUT_PSI_ALERT_THRESHOLD:
            reasons.append(f"Input PSI in [{INPUT_PSI_WARN_THRESHOLD}, {INPUT_PSI_ALERT_THRESHOLD}] ({max_psi:.2f})")
        if PRED_W_WARN_THRESHOLD <= w_dist < PRED_W_ALERT_THRESHOLD:
            reasons.append(f"Pred W in [{PRED_W_WARN_THRESHOLD}, {PRED_W_ALERT_THRESHOLD}] ({w_dist:.4f})")
        return "WARNING", "; ".join(reasons)

    # Rule 4: Normal (In-Control Nominal Operation)
    return "NORMAL", "All metrics within in-control baseline limits"


def generate_monitoring_report(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    feature_cols: Optional[List[str]] = None,
    q_hat: float = PHASE8_CALIBRATED_Q_HAT,
    y_true: Optional[Union[np.ndarray, pd.Series, list]] = None,
    label_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a consolidated telemetry and monitoring report for an incoming batch.

    Evaluates:
    - Input drift across continuous features (KS statistic + PSI)
    - Prediction drift (KS statistic + 1-Wasserstein distance)
    - Conformal uncertainty (Ambiguity rate and set sizes at q_hat)
    - Supervised performance (Brier score + 10-bin ECE) ONLY when labels are supplied.

    Assigns exactly one consolidated outcome: NORMAL | WARNING | DRIFT_ALERT | HUMAN_INSPECTION.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference batch data (e.g., validation baseline).
    curr_df : pd.DataFrame
        Current incoming batch to monitor.
    model : Any
        Trained/calibrated champion model.
    preprocessor : Any
        Fitted ColumnTransformer preprocessor.
    feature_cols : Optional[List[str]]
        List of continuous feature columns for input drift (default: ENGINEERED_NUMERICAL_COLS).
    q_hat : float
        Conformal nonconformity score quantile (default: Phase 8 artifact 0.99923).
    y_true : Optional[array-like]
        Ground truth binary failure labels (optional).
    label_col : Optional[str]
        Column name in curr_df containing labels (optional).

    Returns
    -------
    Dict[str, Any]
        Structured report dictionary containing metrics and assigned outcome.
    """
    if feature_cols is None:
        feature_cols = ENGINEERED_NUMERICAL_COLS

    # Ensure domain features exist on both datasets
    ref_engineered = ref_df.copy()
    if not all(col in ref_engineered.columns for col in ["Temp_Diff", "Power", "Wear_Load"]):
        ref_engineered = create_features(ref_engineered)

    curr_engineered = curr_df.copy()
    if not all(col in curr_engineered.columns for col in ["Temp_Diff", "Power", "Wear_Load"]):
        curr_engineered = create_features(curr_engineered)

    # 1. Input Drift Layer
    input_drift_df = calculate_input_drift(ref_engineered, curr_engineered, feature_cols)
    max_psi_row = input_drift_df.loc[input_drift_df["PSI"].idxmax()]
    max_psi = float(max_psi_row["PSI"])
    top_drifting_feature = str(max_psi_row["Feature"])

    # 2. Prediction Drift Layer
    ref_proc = preprocessor.transform(ref_engineered)
    curr_proc = preprocessor.transform(curr_engineered)

    p_ref = model.predict_proba(ref_proc)[:, 1]
    p_curr = model.predict_proba(curr_proc)[:, 1]

    pred_drift = calculate_prediction_drift(p_ref, p_curr)
    w_dist = float(pred_drift["wasserstein_distance"])
    pred_ks_stat = float(pred_drift["ks_statistic"])
    pred_ks_p_value = float(pred_drift["ks_p_value"])

    # 3. Supervised Reliability Layer (Labels Optional)
    labels = None
    if y_true is not None:
        labels = np.asarray(y_true).astype(int)
    elif label_col is not None and label_col in curr_df.columns:
        labels = np.asarray(curr_df[label_col]).astype(int)

    ece = None
    brier = None
    if labels is not None:
        ece = calculate_ece(labels, p_curr)
        brier = float(brier_score_loss(labels, p_curr))

    # 4. Uncertainty Layer (Conformal APS)
    conformal_res = evaluate_conformal_aps(p_curr, q_hat=q_hat, y_true=labels)
    ambig_rate = conformal_res["ambiguous_rate"]

    # 5. Outcome Assignment
    outcome, justification = classify_monitoring_outcome(
        max_psi=max_psi,
        w_dist=w_dist,
        ece=ece,
        ambig_rate=ambig_rate,
    )

    report = {
        "batch_size": len(curr_df),
        "is_supervised": labels is not None,
        "input_drift": {
            "max_feature_psi": max_psi,
            "top_drifting_feature": top_drifting_feature,
            "feature_table": input_drift_df.to_dict(orient="records"),
        },
        "prediction_drift": {
            "wasserstein_distance": w_dist,
            "ks_statistic": pred_ks_stat,
            "ks_p_value": pred_ks_p_value,
        },
        "conformal_uncertainty": {
            "q_hat_used": q_hat,
            "ambiguous_rate": ambig_rate,
            "singleton_0_rate": conformal_res["singleton_0_rate"],
            "singleton_1_rate": conformal_res["singleton_1_rate"],
            "empirical_coverage": conformal_res.get("empirical_coverage"),
            "mean_set_size": conformal_res.get("mean_set_size"),
        },
        "reliability": {
            "10_bin_ece": ece,
            "brier_score": brier,
        },
        "decision": {
            "outcome": outcome,
            "justification": justification,
            "escalation_required": outcome in ["DRIFT_ALERT", "HUMAN_INSPECTION"],
        },
    }

    return report
