"""Retrieve, format the context, ask the model, and return the answer with its sources.

`build_sources` is not decoration: it is what lets a reader check the answer against the
events it came from, and what the evaluation scores when it asks whether the right event
was retrieved.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from events_rag.config import get_settings
from events_rag.logger import get_logger
from events_rag.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from events_rag.rag.retriever import retrieve_documents

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def build_chat_model() -> ChatMistralAI:
    settings = get_settings()

    if not settings.mistral_api_key:
        raise ValueError("EVENTS_RAG_MISTRAL_API_KEY is missing. Set it in your .env.")

    return ChatMistralAI(
        model=settings.chat_model,
        api_key=settings.mistral_api_key,
        temperature=0,
    )


def format_context(documents: list[Document]) -> str:
    blocks: list[str] = []

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        block = f"""
[Document {index}]
Titre: {metadata.get("title", "")}
Ville: {metadata.get("city", "")}
Date: {metadata.get("date", "")}
Lieu: {metadata.get("location_name", "")}
Adresse: {metadata.get("location_address", "")}
Conditions: {metadata.get("conditions", "")}
Contenu:
{doc.page_content}
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)


def build_sources(documents: list[Document]) -> list[dict]:
    seen: set[tuple[str, str, str | None]] = set()
    sources: list[dict] = []

    for doc in documents:
        metadata = doc.metadata or {}
        key = (
            str(metadata.get("uid", "")),
            str(metadata.get("title", "")),
            metadata.get("date"),
        )

        if key in seen:
            continue
        seen.add(key)

        sources.append(
            {
                "uid": str(metadata.get("uid", "")),
                "title": metadata.get("title", ""),
                "city": metadata.get("city"),
                "date": metadata.get("date"),
                "score": None,
            }
        )

    return sources


def answer_question(question: str, top_k: int | None = None) -> dict:
    documents = retrieve_documents(question=question, top_k=top_k)
    context = format_context(documents)

    chat_model = build_chat_model()
    user_prompt = build_user_prompt(question=question, context=context)

    response = chat_model.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    answer = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(
        "rag.answer_generated",
        question=question,
        retrieved_documents=len(documents),
    )

    return {
        "answer": answer,
        "sources": build_sources(documents),
    }
