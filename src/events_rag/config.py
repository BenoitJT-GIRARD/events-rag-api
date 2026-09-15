"""Every path, model name and threshold, in one settings object read from the environment.

`get_settings` is cached: the corpus path and the index directory are read once and the
same object is handed to the ingestion, the retriever and the API, so the three cannot
disagree about where the index lives.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from events_rag.utils import paths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EVENTS_RAG_",
        extra="ignore",
    )

    app_name: str = "Events RAG API"
    env: str = "dev"
    log_level: str = "INFO"

    location_field: str = "region"
    location_value: str = "Occitanie"
    lang: str = "fr"
    timezone: str = "Europe/Paris"

    date_window_mode: str = "rolling"
    date_window_days: int = Field(default=365, ge=1, le=3650)

    ingestion_batch_size: int = Field(default=100, ge=1, le=100)
    ingestion_max_records: int = Field(default=1000, ge=1, le=10000)

    mistral_api_key: str = ""
    embedding_model: str = "mistral-embed"
    chat_model: str = "mistral-small-latest"

    # Where things are is decided in one place (`events_rag.utils.paths`), and overridden
    # by the environment for a run that writes somewhere else. Two directories are inputs —
    # the corpus and the frozen question set — and two are outputs: the published results and
    # the index, which is rebuilt from the corpus and lives with what a run leaves behind.
    data_dir: Path = paths.DATA_DIR
    raw_data_dir: Path = paths.RAW_DIR
    questions_dir: Path = paths.QUESTIONS_DIR
    reports_dir: Path = paths.REPORTS_DIR
    index_dir: Path = paths.INDEX_DIR

    retrieval_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    faiss_index_name: str = "events_index"
    rebuild_token: str = ""

    # Fixed seed for evaluation-set sampling. The set must be identical across ablation
    # runs, otherwise two configurations are compared on different questions.
    eval_seed: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
