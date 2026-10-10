"""ingestion tables: artifacts, documents, promotions, promotion_sources, corpus_versions, chunks

The schema is written as plain SQL in migrations/sql/0001_ingestion_tables.{up,down}.sql.

Revision ID: 0001
Revises:
Create Date: 2026-10-10
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def _run_sql_file(name: str) -> None:
    # exec_driver_sql sends the file as-is (no SQLAlchemy bind-parameter parsing),
    # so it can hold several statements.
    op.get_bind().exec_driver_sql((SQL_DIR / name).read_text(encoding="utf-8"))


def upgrade() -> None:
    _run_sql_file("0001_ingestion_tables.up.sql")


def downgrade() -> None:
    _run_sql_file("0001_ingestion_tables.down.sql")
