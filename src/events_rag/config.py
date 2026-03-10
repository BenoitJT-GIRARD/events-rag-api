from functools import lru_cache
from pathlib import Path

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

    openagenda_key: str = ""
    openagenda_agenda_uid: str = ""
    openagenda_city: str = "Montpellier"
    openagenda_lang: str = "fr"

    mistral_api_key: str = ""
    embedding_model: str = "mistral-embed"
    chat_model: str = "mistral-small-latest"

    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    index_dir: Path = data_dir / "faiss"

    retrieval_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()