"""The generator draws reproducibly for a seed, from the whole file, and only where the keyword is.

A selector that silently takes the head of the file builds an evaluation set about the
first few hundred events rather than about the corpus.
"""

import random

from events_rag.evaluation.generate_dataset import select_events_for_category


def _events(count: int) -> list[dict]:
    return [{"text": f"concert numero {i}", "metadata": {"uid": str(i)}} for i in range(count)]


def test_selection_is_reproducible_for_a_given_seed() -> None:
    first = select_events_for_category(_events(20), ["concert"], random.Random(0))
    second = select_events_for_category(_events(20), ["concert"], random.Random(0))

    assert [e["metadata"]["uid"] for e in first] == [e["metadata"]["uid"] for e in second]


def test_selection_is_not_limited_to_the_head_of_the_file() -> None:
    # The previous implementation always returned the first two matches, which biased
    # the evaluation set toward the start of the corpus.
    picks = {
        tuple(
            e["metadata"]["uid"]
            for e in select_events_for_category(_events(50), ["concert"], random.Random(seed))
        )
        for seed in range(10)
    }

    assert len(picks) > 1
    assert any(uid not in {"0", "1"} for pick in picks for uid in pick)


def test_selection_ignores_events_without_the_keyword() -> None:
    events = [
        {"text": "exposition de peinture", "metadata": {"uid": "a"}},
        {"text": "concert de jazz", "metadata": {"uid": "b"}},
    ]

    selected = select_events_for_category(events, ["concert"], random.Random(0))

    assert [e["metadata"]["uid"] for e in selected] == ["b"]
