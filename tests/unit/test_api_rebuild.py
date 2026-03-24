from fastapi.testclient import TestClient

from events_rag.api.main import app


def test_rebuild_requires_token() -> None:
    with TestClient(app) as client:
        response = client.post("/rebuild", json={"token": "wrong-token"})

    assert response.status_code in {403, 500}
