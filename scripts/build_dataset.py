"""Re-ingest the corpus from the live upstream.

Outside the reproduction path on purpose: upstream is a rolling window, so this builds a
*different* corpus and silently invalidates every published figure. It is how you would
bootstrap a corpus for another deployment.
"""

import asyncio

from events_rag.ingestion.build_dataset import build_dataset

if __name__ == "__main__":
    asyncio.run(build_dataset())
