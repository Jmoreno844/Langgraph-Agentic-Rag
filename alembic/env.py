from __future__ import annotations
import os
import sys
import pathlib
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure project root is on sys.path for 'import src'
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root if present
try:
    from dotenv import load_dotenv  # type: ignore

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=str(env_file), override=False)
except Exception:
    pass

# this is the Alembic Config object, which provides access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import target metadata from our models
from app.features.products.models import Base  # noqa: E42

target_metadata = Base.metadata

# Set URL from env
db_url = os.getenv("AWS_DB_URL", "")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

version_table_schema = os.getenv("DB_SCHEMA", "public")


def include_object(object_, name, type_, reflected, compare_to):
    # Ignore Alembic's own version table and langgraph checkpoint tables during autogenerate
    if name == "alembic_version":
        return False
    if name in {
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
    }:
        return False
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        version_table_schema=version_table_schema,
        include_schemas=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=version_table_schema,
            include_schemas=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
