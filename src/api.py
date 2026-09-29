"""
FastAPI backend service for Machine Failure Prediction and Drift Monitoring.
Exposes /health, /predict, and /monitor endpoints wrapping the validated src/ pipeline.
"""

from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.features import (
    create_features,
    build_preprocessor,
    load_and_split_data,
    validate_schema,
    INTENDED_RAW_FEATURES,
)
from src.models import (
    train_models,
    calibrate_model,
    predict_conformal_sets,
    COST_OPTIMAL_THRESHOLD,
    PHASE8_CALIBRATED_Q_HAT,
)
from src.app import generate_monitoring_report
from src.schemas import (
    HealthResponse,
    TelemetryInput,
    PredictionResponse,
    MonitorRequest,
)

RAW_DATA_PATH = "data/raw/ai4i2020.csv"
REF_BASELINE_PATH = "data/processed/reference_baseline.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager:
    Initializes and caches the trained Champion Model, Preprocessor, and Reference Baseline.
    Guarantees that the final test partition (N=1,500) remains strictly quarantined and untouched.
    """
    # 1. Load training partition strictly with return_test=False (behavioral quarantine)
    (X_train, y_train), (X_val, y_val) = load_and_split_data(
        RAW_DATA_PATH, random_seed=42, return_test=False
    )

    # 2. Fit preprocessor and champion model strictly on X_train
    preprocessor = build_preprocessor()
    X_train_engineered = create_features(X_train)
    X_train_proc = preprocessor.fit_transform(X_train_engineered)

    base_model = train_models(X_train_proc, y_train, random_seed=42)
    champion_model = calibrate_model(base_model, X_train_proc, y_train, cv=5)

    # 3. Ingest reference baseline batch (derived from clean validation split)
    reference_df = pd.read_csv(REF_BASELINE_PATH)

    # 4. Bind to application state
    app.state.preprocessor = preprocessor
    app.state.champion_model = champion_model
    app.state.reference_baseline = reference_df
    app.state.q_hat = PHASE8_CALIBRATED_Q_HAT  # Frozen Phase 8 artifact: 0.99923

    yield

    # Teardown / cleanup
    app.state.preprocessor = None
    app.state.champion_model = None
    app.state.reference_baseline = None


app = FastAPI(
    title="Machine Failure Drift Monitor API",
    description="Production backend for real-time failure prediction and automated distribution drift surveillance.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check() -> HealthResponse:
    """
    System health and pipeline readiness check.
    """
    model_loaded = getattr(app.state, "champion_model", None) is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        pipeline_version="1.0.0",
        decision_threshold=COST_OPTIMAL_THRESHOLD,
        conformal_q_hat=getattr(app.state, "q_hat", PHASE8_CALIBRATED_Q_HAT),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_failure(telemetry: TelemetryInput) -> PredictionResponse:
    """
    Real-time failure inference for a single machine telemetry record.
    Emits Platt-calibrated failure probability, cost-optimal binary decision (t* = 0.15),
    and split-conformal prediction set using the frozen Phase 8 artifact (q_hat = 0.99923).
    """
    champion_model = getattr(app.state, "champion_model", None)
    preprocessor = getattr(app.state, "preprocessor", None)
    q_hat = getattr(app.state, "q_hat", PHASE8_CALIBRATED_Q_HAT)

    if champion_model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )

    # Convert Pydantic input to standardized DataFrame schema
    raw_dict = {
        "Type": telemetry.Type,
        "Air temperature [K]": telemetry.air_temperature_k,
        "Process temperature [K]": telemetry.process_temperature_k,
        "Rotational speed [rpm]": telemetry.rotational_speed_rpm,
        "Torque [Nm]": telemetry.torque_nm,
        "Tool wear [min]": telemetry.tool_wear_min,
    }
    input_df = pd.DataFrame([raw_dict])

    # Preprocessing and domain features
    feat_df = create_features(input_df)
    proc_X = preprocessor.transform(feat_df)

    # Calibrated probability
    probs = champion_model.predict_proba(proc_X)
    p_fail = float(probs[0, 1])

    # Operational decision at cost-optimal threshold t* = 0.15
    pred_binary = int(p_fail >= COST_OPTIMAL_THRESHOLD)

    # Conformal prediction set using frozen Phase 8 artifact q_hat (never recalibrated per request)
    sets = predict_conformal_sets(np.array([p_fail]), q_hat=q_hat)
    is_ambiguous = bool(sets["ambiguous"][0])

    if sets["singleton_0"][0]:
        prediction_set = ["No Failure"]
    elif sets["singleton_1"][0]:
        prediction_set = ["Machine Failure"]
    else:
        prediction_set = ["No Failure", "Machine Failure"]

    return PredictionResponse(
        failure_probability=round(p_fail, 5),
        prediction=pred_binary,
        threshold_used=COST_OPTIMAL_THRESHOLD,
        prediction_set=prediction_set,
        is_ambiguous=is_ambiguous,
    )


@app.post("/monitor", tags=["Surveillance"])
def monitor_batch(batch_request: MonitorRequest) -> Dict[str, Any]:
    """
    Automated batch monitoring surveillance endpoint.
    Evaluates:
    - Input sensor drift (KS test + PSI across all continuous features)
    - Prediction probability drift (KS test + 1-Wasserstein distance)
    - Conformal prediction uncertainty (Ambiguity rate with frozen q_hat = 0.99923)
    - Supervised probability calibration (10-bin ECE + Brier score) if labels are provided

    Assigns exactly one verified outcome: NORMAL | WARNING | DRIFT_ALERT | HUMAN_INSPECTION.
    Outcomes are determined strictly through the existing mathematical decision rules.
    """
    champion_model = getattr(app.state, "champion_model", None)
    preprocessor = getattr(app.state, "preprocessor", None)
    reference_df = getattr(app.state, "reference_baseline", None)
    q_hat = getattr(app.state, "q_hat", PHASE8_CALIBRATED_Q_HAT)

    if champion_model is None or preprocessor is None or reference_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring pipeline components are not loaded.",
        )

    # Construct and validate DataFrame
    batch_df = pd.DataFrame(batch_request.records)

    # Validate schema and physical bounds
    is_valid, errors = validate_schema(batch_df, require_target=False)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"schema_errors": errors},
        )

    # Validate labels alignment if provided
    labels = None
    if batch_request.labels is not None:
        if len(batch_request.labels) != len(batch_df):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Label count ({len(batch_request.labels)}) does not match "
                    f"record count ({len(batch_df)})."
                ),
            )
        labels = np.asarray(batch_request.labels, dtype=int)

    # Generate comprehensive monitoring report via src/app.py logic
    report = generate_monitoring_report(
        ref_df=reference_df,
        curr_df=batch_df,
        model=champion_model,
        preprocessor=preprocessor,
        q_hat=q_hat,
        y_true=labels,
    )

    return report
