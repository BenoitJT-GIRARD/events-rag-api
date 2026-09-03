"""Embed the committed corpus into a FAISS index. About a minute, and it calls a paid API."""

from pprint import pprint

from events_rag.rag.indexer import build_and_save_index

if __name__ == "__main__":
    result = build_and_save_index()
    pprint(result)
