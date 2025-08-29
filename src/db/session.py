# src/db/session.py
from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
import ssl
import certifi
import asyncpg
from sqlalchemy.engine.url import make_url
from decouple import config

from .url import get_sqlalchemy_url, get_sqlalchemy_async_url


_engine = None
_SessionLocal = None

_async_engine = None
_AsyncSessionLocal = None


def _ensure_engine_and_factory() -> None:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = create_engine(get_sqlalchemy_url(), echo=False, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def _ensure_async_engine_and_factory() -> None:
    global _async_engine, _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        # Build SSL context like the working direct asyncpg connection
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Parse components from the configured URL
        raw_url = config("AWS_DB_URL")
        url = make_url(raw_url)
        user = url.username
        password = url.password
        host = url.host
        port = url.port
        database = url.database

        async def _async_creator():
            return await asyncpg.connect(
                user=user,
                password=password,
                host=host,
                port=port,
                database=database,
                ssl=ssl_context,
            )

        _async_engine = create_async_engine(
            "postgresql+asyncpg://",
            echo=False,
            pool_pre_ping=True,
            async_creator=_async_creator,
        )
        _AsyncSessionLocal = sessionmaker(
            bind=_async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )


def get_engine():
    _ensure_engine_and_factory()
    return _engine


async def get_async_engine():
    _ensure_async_engine_and_factory()
    return _async_engine


def get_session() -> Session:
    _ensure_engine_and_factory()
    return _SessionLocal()


def get_async_session() -> AsyncSession:
    _ensure_async_engine_and_factory()
    return _AsyncSessionLocal()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    db = get_async_session()
    try:
        yield db
    finally:
        await db.close()
