"""The routes answer, and reject a malformed question rather than passing it through.

Integration tier: the real FastAPI application object, wired to the real settings, answering
through a real client. Nothing is mocked but the model, which is behind a paid API.
"""

import pytest
from fastapi.testclient import TestClient

from events_rag.api import main
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


# --- What the route answers when the chain behind it fails -----------------


def test_a_missing_index_answers_503_and_names_the_script(monkeypatch) -> None:
    """The degraded state a reader meets first: the service is up, the index is not there."""

    def absent(**_kwargs):
        raise FileNotFoundError("Index not found. Run scripts/build_index.py first.")

    monkeypatch.setattr(main, "answer_question", absent)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Un concert a Albi ?", "top_k": 3})

    assert response.status_code == 503
    assert "scripts/build_index.py" in response.json()["detail"]


def test_a_question_the_retriever_refuses_answers_400(monkeypatch) -> None:
    def refused(**_kwargs):
        raise ValueError("The question carries no usable term.")

    monkeypatch.setattr(main, "answer_question", refused)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "?????", "top_k": 3})

    assert response.status_code == 400
    assert "no usable term" in response.json()["detail"]


def test_anything_else_answers_500_carrying_what_broke(monkeypatch) -> None:
    def broken(**_kwargs):
        raise RuntimeError("the embedding service refused the call")

    monkeypatch.setattr(main, "answer_question", broken)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Un concert a Albi ?", "top_k": 3})

    assert response.status_code == 500
    assert "the embedding service refused the call" in response.json()["detail"]


def test_an_answer_carries_the_events_it_was_built_from(monkeypatch) -> None:
    """The nominal path, with the paid call replaced: every sentence traces to a uid."""

    def answered(**_kwargs):
        return {
            "answer": "Un atelier velo a Toulouse le 19 mai.",
            "sources": [
                {
                    "uid": "4721721",
                    "title": "Atelier d'initiation a la mecanique",
                    "city": "Toulouse",
                    "date": "2026-05-19T13:00:00+02:00",
                    "score": 0.42,
                }
            ],
        }

    monkeypatch.setattr(main, "answer_question", answered)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Un atelier velo ?", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"][0]["uid"] == "4721721"
    assert payload["answer"].startswith("Un atelier")
