from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EVENTS_",
        extra="ignore",
    )

    app_name: str = "Events RAG API"
    env: str = "dev"
    log_level: str = "INFO"

    location_field: str = "region"   # "city" ou "region"
    location_value: str = "Occitanie"
    lang: str = "fr"
    timezone: str = "Europe/Paris"

    date_window_mode: str = "past"   # "past" ou "future"
    date_window_days: int = Field(default=365, ge=1, le=3650)

    mistral_api_key: str = ""
    embedding_model: str = "mistral-embed"
    chat_model: str = "mistral-small-latest"

    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    index_dir: Path = data_dir / "faiss"

    retrieval_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()