"""The four routes, and the lifespan that loads the index before the first question.

`/rebuild` is the one that needs a token: rebuilding the index re-embeds the whole corpus
through a paid API, so it is not something an anonymous caller should be able to trigger.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from events_rag.api.schemas import (
    AskRequest,
    AskResponse,
    MetadataResponse,
    RebuildRequest,
    RebuildResponse,
)
from events_rag.config import get_settings
from events_rag.logger import configure_logging, get_logger
from events_rag.rag.indexer import build_and_save_index
from events_rag.rag.service import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("app.startup", env=settings.env)

    yield

    logger.info("app.shutdown")


# The tags and their descriptions are what the generated page groups routes by: without
# them a reader meets four endpoints in declaration order and has to infer which one is the
# product and which one is plumbing.
TAGS = [
    {"name": "ask", "description": "Answer a question from the corpus, with its sources."},
    {"name": "operations", "description": "What this deployment serves, and how to rebuild it."},
]

app = FastAPI(
    title="Events RAG API",
    version="0.1.0",
    summary="Retrieval-augmented answers over a frozen corpus of public events.",
    lifespan=lifespan,
    openapi_tags=TAGS,
)


@app.get("/health", tags=["operations"], summary="Liveness, without touching the index")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["operations"],
    summary="The corpus and retrieval settings this deployment is serving",
)
async def metadata() -> MetadataResponse:
    settings = get_settings()

    return MetadataResponse(
        location_field=settings.location_field,
        location_value=settings.location_value,
        language=settings.lang,
        date_window_mode=settings.date_window_mode,
        date_window_days=settings.date_window_days,
        retrieval_k=settings.retrieval_k,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["ask"],
    summary="Ask a question, get an answer and the events it was built from",
)
async def ask(payload: AskRequest) -> AskResponse:
    try:
        result = answer_question(
            question=payload.question,
            top_k=payload.top_k,
        )
        return AskResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while answering question: {exc}",
        ) from exc


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    tags=["operations"],
    summary="Re-embed the committed corpus (token required, one paid call per chunk)",
)
async def rebuild(payload: RebuildRequest) -> RebuildResponse:
    settings = get_settings()

    if not settings.rebuild_token:
        raise HTTPException(
            status_code=500,
            detail="Rebuild token is not configured on the server.",
        )

    if payload.token != settings.rebuild_token:
        raise HTTPException(status_code=403, detail="Invalid rebuild token.")

    try:
        result = build_and_save_index()
        return RebuildResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while rebuilding index: {exc}",
        ) from exc
