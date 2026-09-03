"""From the raw corpus to a FAISS index on disk: load, split, embed, save.

Chunking at 800 characters is the baseline the ablation measures the alternatives against
-- one chunk per event, and metadata folded into the text -- and neither beats it by a
margin twenty questions can support.

FAISS on disk rather than a vector database: a thousand events is a few thousand vectors,
and a managed store would add a service to run and a bill to pay for headroom this corpus
will never need.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from events_rag.config import get_settings
from events_rag.logger import get_logger
from events_rag.rag.retriever import build_embeddings

logger = get_logger(__name__)


def load_raw_documents(path: Path | None = None) -> list[dict]:
    settings = get_settings()
    input_path = path or (settings.raw_data_dir / "events.json")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Raw dataset must be a JSON list of documents.")

    logger.info("indexer.raw_documents_loaded", path=str(input_path), count=len(payload))
    return payload


def to_langchain_documents(items: Sequence[dict]) -> list[Document]:
    documents: list[Document] = []

    for item in items:
        text = item.get("text", "")
        metadata = item.get("metadata", {})

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    logger.info("indexer.langchain_documents_created", count=len(documents))
    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(list(documents))
    logger.info("indexer.documents_split", input_count=len(documents), chunk_count=len(chunks))
    return chunks


def build_faiss_index(chunks: Sequence[Document]) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build FAISS index from an empty chunk list.")

    embeddings = build_embeddings()
    vectorstore = FAISS.from_documents(list(chunks), embeddings)

    logger.info("indexer.faiss_built", chunk_count=len(chunks))
    return vectorstore


def save_faiss_index(
    vectorstore: FAISS,
    index_dir: Path | None = None,
    index_name: str | None = None,
    source_document_count: int | None = None,
    chunk_count: int | None = None,
) -> Path:
    settings = get_settings()
    base_dir = index_dir or settings.index_dir
    final_index_name = index_name or settings.faiss_index_name
    output_dir = base_dir / final_index_name

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))

    manifest = {
        "index_name": final_index_name,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "source_document_count": source_document_count,
        "chunk_count": chunk_count,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    logger.info("indexer.faiss_saved", path=str(output_dir))
    return output_dir


def build_and_save_index() -> dict:
    raw_items = load_raw_documents()
    documents = to_langchain_documents(raw_items)
    chunks = split_documents(documents)
    vectorstore = build_faiss_index(chunks)
    output_path = save_faiss_index(
        vectorstore=vectorstore,
        source_document_count=len(documents),
        chunk_count=len(chunks),
    )

    return {
        "status": "success",
        "indexed_documents": len(documents),
        "indexed_chunks": len(chunks),
        "index_path": str(output_path),
    }
