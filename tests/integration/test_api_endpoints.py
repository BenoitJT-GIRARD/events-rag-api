"""The routes answer, and reject a malformed question rather than passing it through.

Integration tier: the real FastAPI application object, wired to the real settings, answering
through a real client. Nothing is mocked but the model, which is behind a paid API.
"""

import pytest
from fastapi.testclient import TestClient

from events_rag.api.main import app

pytestmark = pytest.mark.integration


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert "location_field" in payload
    assert "location_value" in payload
    assert "language" in payload


def test_ask_endpoint_validation() -> None:
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "", "top_k": 5})

    assert response.status_code == 422
