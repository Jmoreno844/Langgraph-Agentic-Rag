# src/db/url.py
from __future__ import annotations

from decouple import config
from sqlalchemy.engine.url import make_url


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
    db_url = config("AWS_DB_URL")
    url = make_url(db_url)
    url = url.set(drivername=f"postgresql+{driver}")
    if "sslmode" in url.query:
        query = dict(url.query)
        del query["sslmode"]
        url = url.set(query=query)
    return str(url)
