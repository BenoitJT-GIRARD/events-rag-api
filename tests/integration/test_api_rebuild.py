"""`/rebuild` refuses without its token.

The route re-embeds the whole corpus through a paid API. An open one is a bill anyone can
run up.
"""

import pytest
from fastapi.testclient import TestClient

from events_rag.api.main import app

pytestmark = pytest.mark.integration


def test_rebuild_requires_token() -> None:
    with TestClient(app) as client:
        response = client.post("/rebuild", json={"token": "wrong-token"})

    assert response.status_code in {403, 500}
