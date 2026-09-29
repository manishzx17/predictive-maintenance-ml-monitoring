"""
Machine Failure Drift Monitor - Production Reusable Source Code Package.
"""

from src.features import (
    create_features,
    build_preprocessor,
    load_and_split_data,
    validate_schema,
    RAW_NUMERICAL_COLS,
    ENGINEERED_NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    INTENDED_RAW_FEATURES,
)
from src.models import (
    train_models,
    calibrate_model,
    calculate_ece,
    predict_with_threshold,
    calibrate_conformal_aps,
    predict_conformal_sets,
    evaluate_conformal_aps,
    PHASE8_CALIBRATED_Q_HAT,
    COST_OPTIMAL_THRESHOLD,
)
from src.drift import (
    calculate_psi,
    run_ks_test,
    calculate_input_drift,
    calculate_prediction_drift,
)
from src.app import (
    classify_monitoring_outcome,
    generate_monitoring_report,
    INPUT_PSI_ALERT_THRESHOLD,
    PRED_W_ALERT_THRESHOLD,
    INPUT_PSI_WARN_THRESHOLD,
    PRED_W_WARN_THRESHOLD,
    CONFORMAL_AMBIGUITY_THRESHOLD,
    CALIBRATION_ECE_THRESHOLD,
)
from src.schemas import (
    HealthResponse,
    TelemetryInput,
    PredictionResponse,
    MonitorRequest,
)
from src.api import app

__all__ = [
    # Features
    "create_features",
    "build_preprocessor",
    "load_and_split_data",
    "validate_schema",
    "RAW_NUMERICAL_COLS",
    "ENGINEERED_NUMERICAL_COLS",
    "CATEGORICAL_COLS",
    "TARGET_COL",
    "INTENDED_RAW_FEATURES",
    # Models & Uncertainty
    "train_models",
    "calibrate_model",
    "calculate_ece",
    "predict_with_threshold",
    "calibrate_conformal_aps",
    "predict_conformal_sets",
    "evaluate_conformal_aps",
    "PHASE8_CALIBRATED_Q_HAT",
    "COST_OPTIMAL_THRESHOLD",
    # Drift
    "calculate_psi",
    "run_ks_test",
    "calculate_input_drift",
    "calculate_prediction_drift",
    # Monitoring Orchestration
    "classify_monitoring_outcome",
    "generate_monitoring_report",
    "INPUT_PSI_ALERT_THRESHOLD",
    "PRED_W_ALERT_THRESHOLD",
    "INPUT_PSI_WARN_THRESHOLD",
    "PRED_W_WARN_THRESHOLD",
    "CONFORMAL_AMBIGUITY_THRESHOLD",
    "CALIBRATION_ECE_THRESHOLD",
    # API & Schemas
    "app",
    "HealthResponse",
    "TelemetryInput",
    "PredictionResponse",
    "MonitorRequest",
]
