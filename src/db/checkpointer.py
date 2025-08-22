# src/db/checkpointer.py
from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver

from .url import get_libpq_url


def create_checkpointer_context():
    """Return a context manager for the LangGraph PostgresSaver.

    The caller is responsible for entering/exiting the context and keeping it alive
    for the lifetime of the app.
    """
    return PostgresSaver.from_conn_string(get_libpq_url())
