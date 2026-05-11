"""
App configuration loaded from .env file using pydantic-settings.
All Phase 1 constants are defined here and used across controllers and stores.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Mini-RAG"
    APP_VERSION: str = "0.1.0"

    # OpenAI
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_API_BASE: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")

    # Embedding
    EMBEDDINGS_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDINGS_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    MAX_INPUT_TOKENS: int = Field(default=8191, env="MAX_INPUT_TOKENS")

    # Generation (for Phase 2+)
    GENERATE_RESPONSE_MODEL: str = Field(default="gpt-4o-mini", env="GENERATE_RESPONSE_MODEL")
    MAX_RESPONSE_TOKENS: int = Field(default=1024, env="MAX_RESPONSE_TOKENS")
    TEMPERATURE: float = Field(default=0.2, env="TEMPERATURE")

    # Vector DB
    VECTOR_DB_PATH: str = Field(default="assets/db/qdrant_data", env="VECTOR_DB_PATH")
    VECTOR_DISTANCE_METRIC: str = Field(default="Cosine", env="VECTOR_DISTANCE_METRIC")
    COLLECTION_NAME: str = Field(default="job_documents", env="COLLECTION_NAME")

    # Chunking
    CHUNK_SIZE_CHARS: int = Field(default=2000, env="CHUNK_SIZE_CHARS")
    CHUNK_OVERLAP_CHARS: int = Field(default=200, env="CHUNK_OVERLAP_CHARS")

    # Files
    FILE_ALLOWED_EXTENSIONS: list = ["application/pdf", "plain/text"]
    FILE_MAX_SIZE_MB: int = Field(default=10, env="FILE_MAX_SIZE_MB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import this wherever config is needed
settings = Settings()
