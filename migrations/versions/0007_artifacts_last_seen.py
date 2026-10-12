"""artifacts.last_seen_at: the newest fetch that returned these bytes

The schema is written as plain SQL in migrations/sql/0007_artifacts_last_seen.{up,down}.sql.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-11
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0007_artifacts_last_seen.up.sql").read_text("utf-8"))


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        (SQL_DIR / "0007_artifacts_last_seen.down.sql").read_text("utf-8")
    )
