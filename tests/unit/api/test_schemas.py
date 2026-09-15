"""The request shapes refuse what the service cannot answer, before the model is called.

Each bound of the schema has its case at the edge, on both sides: three characters and two,
ten results and eleven. A bound that is documented and not enforced costs a paid call per
malformed request.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from events_rag.api.schemas import AskRequest, AskResponse, RebuildRequest, SourceItem


def test_a_question_and_its_default_depth() -> None:
    request = AskRequest(question="Quel concert a lieu à Albi ?")

    assert request.top_k == 5


@pytest.mark.parametrize("question", ["", "a", "ab"])
def test_a_question_too_short_to_retrieve_anything_is_refused(question: str) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_a_question_longer_than_the_bound_is_refused() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="a" * 501)


@pytest.mark.parametrize("top_k", [0, 11, -1])
def test_a_depth_outside_the_bounds_is_refused(top_k: int) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="Une question valide", top_k=top_k)


def test_an_answer_carries_its_sources() -> None:
    """`sources` is required, and an empty list is a valid value: no source found, and said."""
    response = AskResponse(
        answer="Un concert de l'orchestre national.",
        sources=[SourceItem(uid="evt-1", title="Concert", city="Albi")],
    )

    assert response.sources[0].uid == "evt-1"
    assert response.sources[0].date is None


def test_a_rebuild_without_a_token_is_refused() -> None:
    with pytest.raises(ValidationError):
        RebuildRequest(token="")
