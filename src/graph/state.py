from langgraph.graph import MessagesState
from typing import List, Any


class CustomMessagesState(MessagesState):
    has_been_rewritten: bool = False
    blocked_by_guardrail: bool = False
    source_db: bool = False
    source_docs: bool = False
    docs_importance: str = "normal"
    verify_answer: bool = False
