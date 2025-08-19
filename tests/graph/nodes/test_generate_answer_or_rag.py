import types
import pytest
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


class FakeLLMRecordMessages:
    def __init__(self, response: AIMessage):
        self._response = response
        self.last_messages = None
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        # record exactly what the node passed
        self.last_messages = messages
        return self._response


def test_direct_answer_injects_system_message_when_not_rewritten(monkeypatch):
    # Arrange: fake model returns a plain assistant message (no tool calls)
    fake_ai = AIMessage(content="hello")
    fake_llm = FakeLLMRecordMessages(response=fake_ai)

    # Stub retriever to avoid S3 access during import
    fake_retriever_mod = types.SimpleNamespace()
    fake_retriever_mod.retriever_tool = object()
    monkeypatch.setitem(
        __import__("sys").modules, "src.graph.tools.retriever", fake_retriever_mod
    )

    # Patch the node's model AFTER stubbing retriever
    import importlib
    import src.graph.nodes.generate_answer_or_rag as node

    monkeypatch.setattr(node, "response_model", fake_llm)

    # Minimal user state without rewrite flag
    state = {
        "messages": [HumanMessage(content="hi")],
    }

    # Act
    out_state = node.generate_answer_or_rag(state)

    # Assert: a SystemMessage should be injected when not rewritten
    assert isinstance(fake_llm.last_messages[0], SystemMessage)
    assert out_state["has_been_rewritten"] is False
    assert isinstance(out_state["messages"], list) and len(out_state["messages"]) == 1
    assert isinstance(out_state["messages"][0], AIMessage)
    # Tools were bound for potential routing
    assert fake_llm.bound_tools is not None


def test_tool_call_preserves_rewritten_and_no_system_message_injected(monkeypatch):
    # Arrange: fake model returns an assistant message that requests a tool
    fake_tool_call_ai = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve_rag_docs", "args": {"query": "hi"}, "id": "t1"}],
    )
    fake_llm = FakeLLMRecordMessages(response=fake_tool_call_ai)

    # Stub retriever to avoid S3 access during import
    fake_retriever_mod = types.SimpleNamespace()
    fake_retriever_mod.retriever_tool = object()
    monkeypatch.setitem(
        __import__("sys").modules, "src.graph.tools.retriever", fake_retriever_mod
    )

    import src.graph.nodes.generate_answer_or_rag as node

    monkeypatch.setattr(node, "response_model", fake_llm)

    state = {
        "messages": [HumanMessage(content="Need docs")],
        "has_been_rewritten": True,
    }

    # Act
    out_state = node.generate_answer_or_rag(state)

    # Assert: no SystemMessage prepended when already rewritten
    assert isinstance(fake_llm.last_messages, list)
    assert not isinstance(fake_llm.last_messages[0], SystemMessage)
    # Flag should be preserved
    assert out_state["has_been_rewritten"] is True
    # Returned message should carry a tool call for routing
    assert isinstance(out_state["messages"][0], AIMessage)
    assert out_state["messages"][0].tool_calls
