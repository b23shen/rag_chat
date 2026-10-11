"""corpus_versions: artifact_set_hash → inputs_hash (pages + extracted facts)

The schema is written as plain SQL in migrations/sql/0003_corpus_inputs_hash.{up,down}.sql.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-11
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0003_corpus_inputs_hash.up.sql").read_text("utf-8"))


def downgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0003_corpus_inputs_hash.down.sql").read_text("utf-8"))
