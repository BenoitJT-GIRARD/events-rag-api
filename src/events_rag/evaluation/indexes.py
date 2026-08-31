"""Build and cache one FAISS index per chunking variant.

Only chunking changes the vectors, so the ablation needs one index per variant and no
more. Indexes are cached on disk under a fingerprint of the variant, and the directory is
gitignored: they are derived data, rebuildable from the frozen corpus.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from events_rag.config import get_settings
from events_rag.logger import get_logger
from events_rag.rag.indexer import load_raw_documents, split_documents, to_langchain_documents
from events_rag.rag.retriever import build_embeddings

logger = get_logger(__name__)

HEADER_FIELDS = ("title", "location_name", "city", "date")


@dataclass(frozen=True)
class ChunkingVariant:
    """A way of turning events into indexable documents.

    `chunk_size=None` means one chunk per event: the document is indexed whole.
    `header=True` prefixes the text with its own metadata before embedding.
    """

    name: str
    chunk_size: int | None
    chunk_overlap: int | None
    header: bool = False


def variant_fingerprint(variant: ChunkingVariant) -> str:
    payload = f"{variant.chunk_size}:{variant.chunk_overlap}:{variant.header}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _with_header(document: Document) -> Document:
    """Prefix a document with its own metadata so proper nouns become searchable.

    Title, venue, city and date are stored in metadata and never reach the embedding, so
    a query naming a venue has nothing to match. This puts them in the text.
    """
    parts = [str(document.metadata.get(field, "")).strip() for field in HEADER_FIELDS]
    header = " | ".join(part for part in parts if part)
    if not header:
        return document
    return Document(
        page_content=f"{header}\n\n{document.page_content}",
        metadata=document.metadata,
    )


def split_for_variant(documents: list[Document], variant: ChunkingVariant) -> list[Document]:
    prepared = [_with_header(doc) for doc in documents] if variant.header else list(documents)
    if variant.chunk_size is None:
        return prepared
    return split_documents(prepared, variant.chunk_size, variant.chunk_overlap)


def chunks_for(variant: ChunkingVariant) -> list[Document]:
    documents = to_langchain_documents(load_raw_documents())
    return split_for_variant(documents, variant)


def index_for(variant: ChunkingVariant) -> FAISS:
    settings = get_settings()
    folder = Path(settings.index_dir) / variant_fingerprint(variant)
    embeddings = build_embeddings()

    if folder.exists():
        logger.info("indexes.cache_hit", variant=variant.name, path=str(folder))
        return FAISS.load_local(
            folder_path=str(folder),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    chunks = chunks_for(variant)
    logger.info("indexes.building", variant=variant.name, chunk_count=len(chunks))
    store = FAISS.from_documents(chunks, embeddings)
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))
    return store
