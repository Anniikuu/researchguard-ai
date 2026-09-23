from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_NUM_CTX: int = 4096

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Reranker Settings
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://researchguard:researchguard@localhost:5432/researchguard"

    # App Settings
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
