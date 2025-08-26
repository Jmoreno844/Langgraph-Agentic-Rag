from __future__ import annotations

import json
import re
from typing import Any, Iterable
from langgraph.graph import MessagesState

_DB_TAG_RE = re.compile(r"\[\[DB:([^\]]+)\]\]", re.IGNORECASE)
_PRICE_RE = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")


def _iter_messages_reversed(messages: Iterable[Any]):
    try:
        for m in reversed(list(messages)):
            yield m
    except Exception:
        for m in messages:
            yield m


def _latest_query_products_payload(messages: Iterable[Any]):
    for m in _iter_messages_reversed(messages):
        name = (getattr(m, "name", None) or "").lower()
        mtype = (getattr(m, "type", None) or "").lower()
        if name == "query_products" or (mtype == "tool" and name == "query_products"):
            content = getattr(m, "content", None)
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return None
            return content
    return None


def _index_rows_by_id(rows: Any) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    if not isinstance(rows, list):
        return by_id
    for row in rows:
        data = (
            row if isinstance(row, dict) else getattr(row, "model_dump", lambda: {})()
        )
        if not isinstance(data, dict):
            continue
        pid = data.get("id") or data.get("product_id") or data.get("sku")
        if pid is not None:
            by_id[str(pid)] = data
    return by_id


def verify_groundedness(state: MessagesState):
    # If verification isn't requested, pass through
    if not bool(state.get("verify_answer", False)):
        return {"grounded_ok": True}

    # Only DB checks for now (docs later)
    if bool(state.get("source_db", False)):
        messages = state["messages"]
        final_message = messages[-1]
        final_text = getattr(final_message, "content", "") or ""
        if not isinstance(final_text, str) or not final_text.strip():
            return {
                "grounded_ok": False,
                "blocked_by_guardrail": True,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I couldn't generate a valid answer.",
                    }
                ],
            }

        rows = _latest_query_products_payload(messages)
        if not rows:
            # No rows to verify against: allow pass to avoid false negatives
            return {"grounded_ok": True}

        by_id = _index_rows_by_id(rows)
        tags = set(_DB_TAG_RE.findall(final_text))
        prices = [p.replace(",", "") for p in _PRICE_RE.findall(final_text)]

        # If there are prices in text but no tags, fail
        if prices and not tags:
            return {
                "grounded_ok": False,
                "blocked_by_guardrail": True,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I need to include product IDs for quoted prices. Please ask again or refine the query.",
                    }
                ],
            }

        for tag in tags:
            row = by_id.get(str(tag))
            if not row:
                return {
                    "grounded_ok": False,
                    "blocked_by_guardrail": True,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"I could not verify product {tag} in the latest catalog results.",
                        }
                    ],
                }
            if prices:
                db_price = row.get("price")
                if db_price is not None:
                    db_price_str = str(db_price)
                    if db_price_str not in prices:
                        return {
                            "grounded_ok": False,
                            "blocked_by_guardrail": True,
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": f"The quoted price for {tag} doesn’t match the catalog.",
                                }
                            ],
                        }
            status = row.get("status")
            if status is not None:
                says_unavailable = (
                    any(
                        s in final_text.lower() for s in ["out of stock", "unavailable"]
                    )
                    and "available" not in final_text.lower()
                )
                says_available = (
                    "available" in final_text.lower()
                ) and not says_unavailable
                db_is_active = status == "active"
                if says_available and not db_is_active:
                    return {
                        "grounded_ok": False,
                        "blocked_by_guardrail": True,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"{tag} is not currently available.",
                            }
                        ],
                    }
                if says_unavailable and db_is_active:
                    return {
                        "grounded_ok": False,
                        "blocked_by_guardrail": True,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"{tag} is available per the catalog.",
                            }
                        ],
                    }

    return {"grounded_ok": True}


__all__ = ["verify_groundedness"]
