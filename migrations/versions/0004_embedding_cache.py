"""embedding_cache: embeddings keyed by model, dims and the exact input text

The schema is written as plain SQL in migrations/sql/0004_embedding_cache.{up,down}.sql.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-11
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0004_embedding_cache.up.sql").read_text("utf-8"))


def downgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0004_embedding_cache.down.sql").read_text("utf-8"))
