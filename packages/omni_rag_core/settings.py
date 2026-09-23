from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./.data/omni_rag.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "knowledge_chunks"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    openai_api_url: str = "https://api.openai.com/v1"
    openai_models: str = "gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol"
    anthropic_api_url: str = "https://api.anthropic.com/v1"
    anthropic_models: str = "claude-haiku-4-5-20251001,claude-sonnet-5,claude-opus-5"
    huggingface_api_url: str = "https://router.huggingface.co/v1"
    huggingface_models: str = "Qwen3.8-27B,DeepSeek-R1,DeepSeek-V4.1-Flash,gemma4-31B"
    ollama_models: str = "gemma4:12b"
    llm_credentials_key: str | None = None
    upload_dir: Path = Path(".data/uploads")
    cors_origins: str = "http://localhost:5173,http://localhost:5174"
    request_timeout: float = 60.0
    embedding_size: int = 384
    web_fetch_timeout: float = Field(default=15.0, ge=1, le=60)
    web_fetch_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    web_fetch_max_redirects: int = Field(default=5, ge=0, le=10)
    chat_session_minutes: int = Field(default=10, ge=1)
    chat_session_sweep_seconds: int = Field(default=5, ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator(
        "openai_models", "anthropic_models", "huggingface_models", "ollama_models"
    )
    @classmethod
    def validate_model_list(cls, value: str) -> str:
        models = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
        if not models:
            raise ValueError("At least one model must be configured for each LLM provider")
        return ",".join(models)

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def llm_models(self) -> dict[str, tuple[str, ...]]:
        return {
            "openai": tuple(self.openai_models.split(",")),
            "anthropic": tuple(self.anthropic_models.split(",")),
            "huggingface": tuple(self.huggingface_models.split(",")),
            "ollama": tuple(self.ollama_models.split(",")),
        }

@lru_cache
def get_settings() -> Settings:
    return Settings()
