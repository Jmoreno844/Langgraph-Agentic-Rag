# src/db/session.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .url import get_sqlalchemy_url


_engine = None
_SessionLocal = None


def _ensure_engine_and_factory() -> None:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = create_engine(get_sqlalchemy_url(), echo=False, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_engine():
    _ensure_engine_and_factory()
    return _engine


def get_session() -> Session:
    _ensure_engine_and_factory()
    return _SessionLocal()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = get_session()
    try:
        yield db
    finally:
        db.close()
