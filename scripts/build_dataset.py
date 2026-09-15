"""Re-ingest the corpus from the live upstream.

Outside the reproduction path on purpose: upstream is a rolling window, so a second run
returns other events, and every published figure would then describe a corpus nobody has.
Run it to start a new deployment, and expect to re-measure everything after it.
"""

import asyncio

from events_rag.ingestion.build_dataset import build_dataset

if __name__ == "__main__":
    asyncio.run(build_dataset())
