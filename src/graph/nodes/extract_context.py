from typing import Any, Iterable, List
from src.graph.state import CustomMessagesState


def _iter_messages_reversed(messages: Iterable[Any]):
    """Iterate through messages in reverse order."""
    try:
        for m in reversed(list(messages)):
            yield m
    except Exception:
        for m in messages:
            yield m


def _latest_user_content(messages: Iterable[Any]) -> str:
    """Extract the latest user message content."""
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


def _extract_tool_context_list(messages: Iterable[Any]) -> List[str]:
    """Collect context from tool messages as a list of strings (latest-first)."""
    contexts: List[str] = []
    for m in _iter_messages_reversed(messages):
        mtype = getattr(m, "type", None)
        if mtype == "tool":
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.strip():
                contexts.append(content)
    return contexts


def extract_context(state: CustomMessagesState) -> CustomMessagesState:
    """Extract question and retrieved context from messages and store in state."""
    messages = state["messages"]

    # Extract the current question (latest user message)
    current_question = _latest_user_content(messages)

    # Extract context from tool results as a list
    retrieved_context_list = _extract_tool_context_list(messages)

    return {
        "current_question": current_question,
        "retrieved_context": retrieved_context_list,
    }
