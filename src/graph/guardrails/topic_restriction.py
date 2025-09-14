from typing import List
from guardrails import Guard
from guardrails.hub import RestrictToTopic
from langchain_core.messages import HumanMessage
from src.graph.state import CustomMessagesState
import re

# Configure allowed topics for the assistant
ALLOWED_TOPICS: List[str] = [
    "pc parts support",
    "orders",
    "hardware compatibility",
    "warranty",
    "returns",
    "products",
    "product",
    "product support",
    "product categories",
    "greetings",
]

INVALID_TOPICS: List[str] = [
    "human body",
]

# Initialize a Guard with a topic restriction validator
# Best practice: use deterministic classification; do not rely on LLM to "fix" user input
_guard = Guard().use(
    RestrictToTopic(
        valid_topics=ALLOWED_TOPICS,
        disable_classifier=False,
        disable_llm=True,
        on_fail="exception",
        model_threshold=0.5,
    )
)


def topic_guardrail(state: CustomMessagesState) -> CustomMessagesState:
    """Validate the latest user message is within allowed topics.

    - If valid: pass-through and allow the graph to continue.
    - If invalid: set `blocked_by_guardrail`.
    """
    if not state.get("messages"):
        return {}

    # Prefer the last HumanMessage; fall back to the last message content
    messages = state["messages"]
    user_text = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_text = message.content
            break
    if user_text is None:
        user_text = messages[-1].content

    # Sanitize transcript-like content that includes role markers
    # Extract the portion likely authored by the human user
    lowered = str(user_text).lower()
    if "human" in lowered and "ai" in lowered:
        h_idx = lowered.rfind("human")
        a_idx = lowered.find("ai", h_idx + 5)
        if h_idx != -1 and a_idx != -1 and a_idx > h_idx:
            candidate = str(user_text)[h_idx + len("human") : a_idx].strip()
            if candidate:
                user_text = candidate
    # Remove obvious role markers and extra whitespace
    user_text = re.sub(
        r"\b(human|ai|assistant|system)\b[:\s]*",
        "",
        str(user_text),
        flags=re.IGNORECASE,
    ).strip()

    try:
        _guard.validate(user_text)
        return {"blocked_by_guardrail": False}
    except Exception:
        return {"blocked_by_guardrail": True}
