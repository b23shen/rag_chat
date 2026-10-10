"""Database engine and sessions (sync; ingestion is a batch job)."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from promo_rag.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        get_settings().database_url,
        # statement_timeout is set per connection; ingestion statements are small.
        connect_args={"options": "-c statement_timeout=30000"},
    )


def new_session() -> Session:
    return Session(get_engine())
