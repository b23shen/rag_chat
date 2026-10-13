"""promotions.cities: the city or county of a store notice, from its address

The schema is written as plain SQL in migrations/sql/0008_promotion_cities.{up,down}.sql.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-12
"""

from collections.abc import Sequence
from pathlib import Path

from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def upgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0008_promotion_cities.up.sql").read_text("utf-8"))


def downgrade() -> None:
    op.get_bind().exec_driver_sql((SQL_DIR / "0008_promotion_cities.down.sql").read_text("utf-8"))
