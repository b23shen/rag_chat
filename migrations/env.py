"""Alembic environment. The database URL comes from .env via promo_rag.config."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from promo_rag.config import get_settings
from promo_rag.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
SCHEMAS = {"platform", "rag"}
# alembic_version stays in public. Without this, Alembic would look for it in the first
# schema on the search_path (rag) and think no migration had ever run.
VERSION_TABLE_SCHEMA = "public"


def include_name(name, type_, parent_names):
    """Only compare our own schemas (not public's leftovers or system schemas)."""
    if type_ == "schema":
        return name in SCHEMAS
    return True


def run_migrations_offline() -> None:
    """Print the SQL instead of running it (`alembic upgrade head --sql`)."""
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        version_table_schema=VERSION_TABLE_SCHEMA,
        include_schemas=True,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # No app search_path here: Postgres reports tables in a search_path schema as being in
    # "the default schema", and Alembic would then think rag's tables were missing.
    engine = create_engine(get_settings().database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=VERSION_TABLE_SCHEMA,
            include_schemas=True,
            include_name=include_name,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
