"""corpus_versions: at most one active version

The schema is written as plain SQL in migrations/sql/0005_one_active_corpus.{up,down}.sql.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-11
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0005_one_active_corpus.up.sql").read_text("utf-8"))


def downgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0005_one_active_corpus.down.sql").read_text("utf-8"))
