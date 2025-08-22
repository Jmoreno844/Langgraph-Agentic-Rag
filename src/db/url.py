# src/db/url.py
from __future__ import annotations

from src.settings import settings


def _strip_driver(url: str) -> str:
    if url.startswith("postgresql+"):
        return "postgresql://" + url.split("://", 1)[1]
    return url


def get_sqlalchemy_url() -> str:
    """Return a SQLAlchemy-compatible URL. Driverless is fine."""
    return _strip_driver(settings.AWS_DB_URL)


def get_libpq_url() -> str:
    """Return a libpq/psycopg-compatible URL (no sqlachemy-style +driver)."""
    return _strip_driver(settings.AWS_DB_URL)
