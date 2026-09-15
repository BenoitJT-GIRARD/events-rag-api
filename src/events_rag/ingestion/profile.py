"""Count what the committed corpus holds, so the pages that describe it can cite a file.

Every figure a document states about the corpus comes from here: how many events, how many
towns, which towns dominate. The alternative is a number typed once and never re-counted,
which is how `data/raw/SOURCE.md` came to claim 415 towns for a corpus that holds 414.
"""

from collections import Counter
from collections.abc import Sequence


def profile_corpus(events: Sequence[dict], busiest: int = 5) -> dict:
    """The shape of the corpus, counted rather than remembered."""
    towns = Counter(
        (event.get("metadata") or {}).get("city")
        for event in events
        if (event.get("metadata") or {}).get("city")
    )
    lengths = [len(event.get("text") or "") for event in events]
    return {
        "events": len(events),
        "distinct_towns": len(towns),
        "events_without_a_town": sum(
            1 for event in events if not (event.get("metadata") or {}).get("city")
        ),
        "busiest_towns": [{"town": town, "events": n} for town, n in towns.most_common(busiest)],
        "mean_text_characters": round(sum(lengths) / len(lengths)) if lengths else 0,
    }
