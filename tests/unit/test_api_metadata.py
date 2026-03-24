from fastapi.testclient import TestClient

from events_rag.api.main import app


def test_metadata_endpoint_returns_expected_fields() -> None:
    with TestClient(app) as client:
        response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()

    assert "location_field" in payload
    assert "location_value" in payload
    assert "language" in payload
    assert "date_window_mode" in payload
    assert "date_window_days" in payload
    assert "retrieval_k" in payload
