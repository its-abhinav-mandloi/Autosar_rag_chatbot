"""
Configuration Management for AUTOSAR Document Intelligence Assistant.

Loads settings from environment variables / .env file using Pydantic Settings.
All configuration is centralized here — no magic strings elsewhere in the codebase.

Continuation Note:
    This module is complete. If adding new config values, add them as fields
    to the Settings class and update .env.example accordingly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Precedence: env vars > .env file > defaults defined here.
    """

    # --- Ollama Configuration ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3.2"

    # --- ChromaDB Configuration ---
    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "autosar_docs"

    # --- Chunking Configuration ---
    CHUNK_SIZE: int = 512  # Max tokens per chunk
    CHUNK_OVERLAP: int = 50  # Overlap tokens between adjacent chunks

    # --- Retrieval Configuration ---
    TOP_K_RETRIEVAL: int = 10  # Initial retrieval count
    TOP_K_RERANK: int = 5  # After re-ranking, keep top N

    # --- Server Configuration ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # --- Data Paths ---
    RAW_DATA_PATH: str = "./Database"
    METADATA_STORE_PATH: str = "./data/metadata"

    # --- Heartbeat Configuration ---
    HEARTBEAT_INTERVAL_SECONDS: int = 30

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def chroma_db_absolute_path(self) -> Path:
        """Resolve ChromaDB path relative to project root."""
        return Path(self.CHROMA_DB_PATH).resolve()

    @property
    def raw_data_absolute_path(self) -> Path:
        """Resolve raw data path relative to project root."""
        return Path(self.RAW_DATA_PATH).resolve()

    @property
    def metadata_store_absolute_path(self) -> Path:
        """Resolve metadata store path relative to project root."""
        return Path(self.METADATA_STORE_PATH).resolve()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache so the .env file is only read once per process lifetime.
    Call get_settings.cache_clear() in tests to reload.
    """
    return Settings()
