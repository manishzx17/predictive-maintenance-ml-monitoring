"""
Pydantic v2 schemas for the FastAPI Machine Failure Drift Monitor API.
Provides strict type and boundary validation for machine telemetry and batch monitoring requests.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_version: str
    decision_threshold: float
    conformal_q_hat: float


class TelemetryInput(BaseModel):
    """
    Single machine telemetry input.
    Accepts both standard raw sensor column names and snake_case aliases.
    """
    model_config = ConfigDict(populate_by_name=True)

    Type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine variant product quality type: L (50%), M (30%), or H (20%)"
    )
    air_temperature_k: float = Field(
        ...,
        alias="Air temperature [K]",
        gt=0.0,
        description="Air temperature in Kelvin (must be positive, typically ~295-305 K)"
    )
    process_temperature_k: float = Field(
        ...,
        alias="Process temperature [K]",
        gt=0.0,
        description="Process temperature in Kelvin (must be positive, typically ~305-315 K)"
    )
    rotational_speed_rpm: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        gt=0.0,
        description="Rotational speed in revolutions per minute (must be positive)"
    )
    torque_nm: float = Field(
        ...,
        alias="Torque [Nm]",
        ge=0.0,
        description="Torque in Newton-meters (must be non-negative)"
    )
    tool_wear_min: float = Field(
        ...,
        alias="Tool wear [min]",
        ge=0.0,
        description="Cumulative tool wear in minutes (must be non-negative)"
    )


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="0 for No Failure, 1 for Failure")
    threshold_used: float
    prediction_set: List[str]
    is_ambiguous: bool


class MonitorRequest(BaseModel):
    """
    Batch telemetry request for distribution and reliability monitoring.
    Contains raw sensor records and optional ground-truth failure labels.
    """
    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of raw machine sensor telemetry records"
    )
    labels: Optional[List[int]] = Field(
        default=None,
        description="Optional ground-truth failure labels (0 or 1) for supervised calibration evaluation"
    )

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v, info):
        if v is not None:
            invalid = [val for val in v if val not in (0, 1)]
            if invalid:
                raise ValueError(f"All labels must be 0 or 1, found invalid values: {invalid[:5]}")
        return v
