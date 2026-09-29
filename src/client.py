"""
API Client adapter for communicating with the FastAPI backend.
Provides a clean interface for Streamlit and integration tests.
Supports both live HTTP servers via httpx and in-memory FastAPI TestClient instances.
"""

from typing import Dict, Any, List, Optional, Union
import httpx
from fastapi.testclient import TestClient


class MonitorApiClient:
    """Client adapter for Machine Failure Drift Monitor API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        client: Optional[Union[httpx.Client, TestClient]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def get_health(self) -> Dict[str, Any]:
        """Query GET /health endpoint."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def predict_failure(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Query POST /predict endpoint with machine telemetry."""
        response = self._client.post("/predict", json=telemetry)
        response.raise_for_status()
        return response.json()

    def monitor_batch(
        self,
        records: List[Dict[str, Any]],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Query POST /monitor endpoint with incoming batch telemetry."""
        payload: Dict[str, Any] = {"records": records}
        if labels is not None:
            payload["labels"] = labels

        response = self._client.post("/monitor", json=payload)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close underlying HTTP client if applicable."""
        if hasattr(self._client, "close") and not isinstance(self._client, TestClient):
            self._client.close()
