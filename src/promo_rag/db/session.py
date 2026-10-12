"""Database engine and sessions (sync; ingestion is a batch job)."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from promo_rag.config import get_settings

# Tables live in two schemas (rag, platform); with this search_path, unqualified names in
# raw SQL still resolve. statement_timeout stops a runaway query after 30 s.
CONNECT_ARGS = {"options": "-c search_path=rag,platform,public -c statement_timeout=30000"}


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, connect_args=CONNECT_ARGS)


def new_session() -> Session:
    return Session(get_engine())
