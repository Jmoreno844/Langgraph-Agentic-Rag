from langgraph.graph import MessagesState
from typing import List, Any


class CustomMessagesState(MessagesState):
    has_been_rewritten: bool = False
