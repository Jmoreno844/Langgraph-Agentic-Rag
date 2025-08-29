# src/db/url.py
from __future__ import annotations

from decouple import config
from pydantic import PostgresDsn


def _strip_driver(url: str) -> str:
    if url.startswith("postgresql+"):
        return "postgresql://" + url.split("://", 1)[1]
    return url


def get_sqlalchemy_url() -> str:
    """Return a SQLAlchemy-compatible URL. Driverless is fine."""
    return _strip_driver(config("AWS_DB_URL"))


def get_libpq_url() -> str:
    """Return a libpq/psycopg-compatible URL (no sqlachemy-style +driver)."""
    return _strip_driver(config("AWS_DB_URL"))


def get_sqlalchemy_async_url(driver: str = "asyncpg") -> str:
    user = config("POSTGRES_USER", default="postgres")
    password = config("POSTGRES_PASSWORD", default="postgres")
    host = config("POSTGRES_HOST", default="localhost")
    port = config("POSTGRES_PORT", default=5432, cast=int)
    db = config("POSTGRES_DB", default="postgres")
    return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{db}"
