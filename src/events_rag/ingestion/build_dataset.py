"""Fetch the events and write the raw corpus.

Deliberately outside the reproduction path: running it replaces the frozen corpus with a
different one, and every published number stops applying. See the README.
"""

import json

from events_rag.config import get_settings
from events_rag.ingestion.openagenda_client import OpenAgendaClient
from events_rag.ingestion.preprocess import event_to_document
from events_rag.logger import get_logger

logger = get_logger(__name__)


async def build_dataset() -> list[dict]:
    settings = get_settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAgendaClient()
    events = await client.fetch_all_events(
        batch_size=settings.ingestion_batch_size,
        max_records=settings.ingestion_max_records,
    )

    documents = [event_to_document(event) for event in events]

    output_path = settings.raw_data_dir / "events.json"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)

    logger.info(
        "dataset.saved",
        path=str(output_path),
        count=len(documents),
        max_records=settings.ingestion_max_records,
    )
    return documents
