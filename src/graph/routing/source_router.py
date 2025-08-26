from __future__ import annotations

from typing import Any, Iterable
from langgraph.graph import MessagesState


_DB_TOOL_NAMES = {"query_products", "list_product_categories"}
_DOC_TOOL_HINTS = {"retrieve_rag_docs", "retriever", "retrieve", "rag"}
_POLICY_HINT_KEYWORDS = (
    "policy",
    "policies",
    "terms",
    "warranty",
    "return",
    "refund",
    "privacy",
    "security",
    "compliance",
    "official",
    "legal",
)


def _iter_messages_reversed(messages: Iterable[Any]):
    try:
        for m in reversed(list(messages)):
            yield m
    except Exception:
        for m in messages:
            yield m


def _latest_user_content(messages: Iterable[Any]) -> str:
    for m in _iter_messages_reversed(messages):
        role = getattr(m, "role", None) or getattr(m, "type", None)
        if role in ("user", "human"):
            content = getattr(m, "content", "")
            if isinstance(content, str):
                return content
    try:
        last = list(messages)[-1]
        content = getattr(last, "content", "")
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def _detect_sources(messages: Iterable[Any]) -> tuple[bool, bool]:
    saw_db = False
    saw_docs = False
    for m in _iter_messages_reversed(messages):
        name = (getattr(m, "name", None) or "").lower()
        mtype = (getattr(m, "type", None) or "").lower()
        is_toolish = (mtype == "tool") or bool(name)
        if not is_toolish:
            continue
        if any(db in name for db in _DB_TOOL_NAMES):
            saw_db = True
        if any(doc in name for doc in _DOC_TOOL_HINTS):
            saw_docs = True
        if saw_db and saw_docs:
            break
    return saw_db, saw_docs


def _infer_docs_importance(user_text: str) -> str:
    text = (user_text or "").lower()
    if any(kw in text for kw in _POLICY_HINT_KEYWORDS):
        return "high"
    return "normal"


def source_router(state: MessagesState):
    messages = state["messages"]
    source_db, source_docs = _detect_sources(messages)
    user_text = _latest_user_content(messages)
    importance = _infer_docs_importance(user_text)

    # Minimal rule: verify when DB is involved, or when docs importance is high
    verify_answer = bool(source_db) or (importance == "high" and bool(source_docs))

    return {
        "source_db": bool(source_db),
        "source_docs": bool(source_docs),
        "docs_importance": importance,
        "verify_answer": verify_answer,
    }


__all__ = ["source_router"]
