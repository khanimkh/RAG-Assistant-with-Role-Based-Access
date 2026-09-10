from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# FILE OVERVIEW: settings.py
# ============================================================================
# PURPOSE
#   All configuration in one typed, validated place. Using pydantic's
#   BaseSettings instead of manual os.getenv() calls means: field types are
#   enforced (RERANKER_ENABLED=true becomes a real bool, not a string you
#   have to .lower() == "true" yourself), and a malformed .env value fails
#   loudly at startup instead of silently becoming a wrong default deep
#   inside the app.
#
# GENERAL ALGORITHM
#   1. declare every setting as a typed field with a sensible default
#   2. SettingsConfigDict points pydantic-settings at .env - it reads that
#      file, matches each line to a field by name (case-insensitive), and
#      validates/converts the value to that field's declared type
#   3. `settings = Settings()` at the bottom runs this once at import time;
#      every other file does `from configs.settings import settings` and
#      gets the same already-loaded instance
# ============================================================================


class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 800
    chunk_overlap: int = 100

    embedding_batch_size: int = 500
    embedding_max_concurrent_batches: int = 5

    top_k: int = 5
    rerank_top_k: int = 15
    reranker_enabled: bool = False

    vector_dir: str = "vector_store"
    prompt_version: str = "v4"

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
