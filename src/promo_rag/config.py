"""App settings, read from environment variables and the repo's .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # Gemini
    gemini_api_key: SecretStr
    chat_model: str | None = None
    chat_fallback_model: str | None = None
    helper_model: str | None = None
    judge_model: str | None = None
    embed_model: str = "gemini-embedding-2"
    embed_dims: int = 1536

    # Postgres
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "promo_rag"
    pg_password: SecretStr
    pg_db: str = "rag"

    # Raw page bytes (local folder in dev)
    blob_dir: Path = REPO_ROOT / "data" / "blobs"

    @property
    def pg_conninfo(self) -> str:
        """libpq connection string for psycopg."""
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password.get_secret_value()}"
        )

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL (psycopg 3 driver)."""
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
