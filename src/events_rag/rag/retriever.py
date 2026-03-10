from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from events_rag.config import get_settings
from events_rag.logger import get_logger

logger = get_logger(__name__)


def build_embeddings() -> MistralAIEmbeddings:
    settings = get_settings()

    if not settings.mistral_api_key:
        raise ValueError(
            "EVENTS_MISTRAL_API_KEY is missing. Set it in your .env."
        )

    return MistralAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )


def load_vectorstore(index_dir: Path | None = None, index_name: str | None = None) -> FAISS:
    settings = get_settings()

    base_dir = index_dir or settings.index_dir
    final_index_name = index_name or settings.faiss_index_name
    input_dir = base_dir / final_index_name

    if not input_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {input_dir}. "
            "Build the index first with scripts/build_index.py."
        )

    embeddings = build_embeddings()

    vectorstore = FAISS.load_local(
        folder_path=str(input_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info("retriever.vectorstore_loaded", path=str(input_dir))
    return vectorstore


def retrieve_documents(question: str, top_k: int | None = None) -> list[Document]:
    settings = get_settings()
    final_top_k = top_k or settings.retrieval_k

    vectorstore = load_vectorstore()
    documents = vectorstore.similarity_search(question, k=final_top_k)

    logger.info(
        "retriever.documents_retrieved",
        question=question,
        top_k=final_top_k,
        count=len(documents),
    )
    return documents