import asyncio

from events_rag.ingestion.build_dataset import build_dataset

if __name__ == "__main__":
    asyncio.run(build_dataset())
