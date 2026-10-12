"""platform and rag schemas; datasets, workflows, workflow_runs, action_runs

The schema is written as plain SQL in migrations/sql/0006_schemas_and_workflows.{up,down}.sql.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-11
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql(
        (SQL_DIR / "0006_schemas_and_workflows.up.sql").read_text("utf-8")
    )


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        (SQL_DIR / "0006_schemas_and_workflows.down.sql").read_text("utf-8")
    )
