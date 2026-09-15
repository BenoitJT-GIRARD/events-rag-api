"""The ingestion writes the corpus the rest of the project treats as frozen.

It is deliberately outside the reproduction path — running it replaces the corpus and
invalidates every published number — which is exactly why its shape is pinned here: the file
it writes, where it writes it, and the fact that it converts every event it was given.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from events_rag.ingestion import build_dataset as module


class FakeClient:
    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.asked: dict[str, int] = {}

    async def fetch_all_events(self, batch_size: int, max_records: int) -> list[dict]:
        self.asked = {"batch_size": batch_size, "max_records": max_records}
        return self._events


def test_the_corpus_is_written_where_the_settings_say(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    settings = module.get_settings()
    monkeypatch.setattr(settings, "raw_data_dir", raw)
    monkeypatch.setattr(settings, "ingestion_batch_size", 50)
    monkeypatch.setattr(settings, "ingestion_max_records", 120)

    client = FakeClient([{"uid": "evt-1", "title": {"fr": "Concert"}}])
    monkeypatch.setattr(module, "OpenAgendaClient", lambda: client)
    monkeypatch.setattr(module, "event_to_document", lambda event: {"text": event["uid"]})

    documents = asyncio.run(module.build_dataset())

    written = json.loads((raw / "events.json").read_text(encoding="utf-8"))
    assert written == [{"text": "evt-1"}]
    assert documents == written
    assert client.asked == {"batch_size": 50, "max_records": 120}


def test_the_ingestion_limits_come_from_the_settings(tmp_path: Path, monkeypatch) -> None:
    """The committed corpus was built with these values; a different pair is a different corpus."""
    settings = module.get_settings()
    monkeypatch.setattr(settings, "raw_data_dir", tmp_path)
    monkeypatch.setattr(settings, "ingestion_batch_size", 10)
    monkeypatch.setattr(settings, "ingestion_max_records", 30)

    client = FakeClient([])
    monkeypatch.setattr(module, "OpenAgendaClient", lambda: client)

    asyncio.run(module.build_dataset())

    assert client.asked == {"batch_size": 10, "max_records": 30}
