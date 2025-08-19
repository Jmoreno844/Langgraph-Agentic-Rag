import importlib
import types

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


class FakeLLMPathController:
    def __init__(self, return_tool_call: bool = False):
        self.return_tool_call = return_tool_call
        self.last_messages = None
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.last_messages = messages
        if self.return_tool_call:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_rag_docs", "args": {"query": "q"}, "id": "t1"}
                ],
            )
        return AIMessage(content="direct answer")


def fake_retriever_callable(query: str = None, **kwargs) -> str:
    """fake retriever"""
    return "retrieved context"


def compile_graph_with_fakes(monkeypatch, return_tool_call: bool):
    # Stub retriever module with a simple callable BEFORE importing the graph
    fake_retriever_mod = types.SimpleNamespace()
    fake_retriever_mod.retriever_tool = fake_retriever_callable
    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "src.graph.tools.retriever",
        fake_retriever_mod,
    )

    # Import nodes and set fake LLMs
    import src.graph.nodes.generate_answer_or_rag as gar
    import src.graph.nodes.generate_answer as ga

    fake_llm = FakeLLMPathController(return_tool_call=return_tool_call)
    monkeypatch.setattr(gar, "response_model", fake_llm)
    monkeypatch.setattr(ga, "response_model", fake_llm)

    # Import graph fresh
    import src.graph.graph as g

    importlib.reload(g)

    return g.graph.compile(checkpointer=MemorySaver())


def test_direct_answer_path(monkeypatch):
    app = compile_graph_with_fakes(monkeypatch, return_tool_call=False)
    out = app.invoke(
        {"messages": [HumanMessage(content="hello")], "has_been_rewritten": False},
        config={"configurable": {"thread_id": "t"}},
    )

    # Expect AIMessage present
    assert any(isinstance(m, AIMessage) for m in out["messages"])


def test_rag_path_runs_retriever_and_generate_answer(monkeypatch):
    app = compile_graph_with_fakes(monkeypatch, return_tool_call=True)
    out = app.invoke(
        {"messages": [HumanMessage(content="need docs")], "has_been_rewritten": False},
        config={"configurable": {"thread_id": "t"}},
    )

    # Final state should include an AIMessage produced after tool flow
    assert any(isinstance(m, AIMessage) for m in out["messages"])
