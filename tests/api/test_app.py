import types
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage


class FakeLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="ok")


def _stub_retriever(query: str = None, **kwargs) -> str:
    """Stub retriever for tests; returns a fixed context string."""
    return "ctx"


@pytest.fixture
def app(monkeypatch):
    # Stub retriever to avoid S3 load in tool module
    fake_retriever_mod = types.SimpleNamespace()
    fake_retriever_mod.retriever_tool = _stub_retriever
    monkeypatch.setitem(
        __import__("sys").modules, "src.graph.tools.retriever", fake_retriever_mod
    )

    # Patch runtime.build_app so startup lifespan compiles a local graph
    import src.graph.runtime as runtime

    def fake_build_app():
        # Inject fake models before compiling the graph
        import src.graph.nodes.generate_answer as ga
        import src.graph.nodes.generate_answer_or_rag as gar

        ga.response_model = FakeLLM()
        gar.response_model = FakeLLM()
        import src.graph.graph as g

        return g.graph.compile(checkpointer=MemorySaver())

    monkeypatch.setattr(runtime, "build_app", fake_build_app)

    # Import app after patching
    from src.app.main import app as fastapi_app

    return fastapi_app


def test_root(app):
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World"}


def test_chat_stream_contract(app):
    # Use context manager to ensure startup/shutdown lifespan runs
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/chat",
            json={"session_id": "t1", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            assert resp.status_code == 200
            body = "".join([text for text in resp.iter_text()])
    assert "event: start" in body
    assert "data:" in body
    assert "event: end" in body
