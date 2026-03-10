from contextlib import asynccontextmanager

from fastapi import FastAPI

from events_rag.config import get_settings
from events_rag.logger import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("app.startup", env=settings.env)

    yield

    logger.info("app.shutdown")


app = FastAPI(
    title="Events RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metadata")
async def metadata():
    settings = get_settings()

    return {
        "location_field": settings.location_field,
        "location_value": settings.location_value,
        "language": settings.lang,
        "date_window_mode": settings.date_window_mode,
        "date_window_days": settings.date_window_days,
        "retrieval_k": settings.retrieval_k,
    }