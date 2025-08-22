import types
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import AIMessage

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="ok")

    def _stub_retriever(query: str = None, **kwargs) -> str:
        return "ctx"

    fake_retriever_mod = types.SimpleNamespace()
    fake_retriever_mod.retriever_tool = _stub_retriever
    monkeypatch.setitem(
        __import__("sys").modules, "src.graph.tools.retriever", fake_retriever_mod
    )

    import src.graph.runtime as runtime

    def fake_build_app():
        import src.graph.nodes.generate_answer as ga
        import src.graph.nodes.generate_answer_or_rag as gar
        import src.graph.graph as g

        ga.response_model = FakeLLM()
        gar.response_model = FakeLLM()
        return g.graph.compile(checkpointer=MemorySaver())

    monkeypatch.setattr(runtime, "build_app", fake_build_app)

    # Stub documents service BEFORE importing the app/router
    fake_documents_service = types.SimpleNamespace()

    def document_from_head(key: str, include_url: bool = False):
        return {
            "key": key,
            "name": key.split("/")[-1],
            "size": 10,
            "etag": "abc123",
            "last_modified": "2025-01-01T00:00:00Z",
            "storage_class": None,
            "checksum_algorithm": None,
            "checksum_type": None,
            "url": "https://example.com" if include_url else None,
        }

    async def list_documents(include_url: bool):
        return [document_from_head("file.txt", include_url)]

    def get_sync_status(key: str):
        return ("in_sync", 5, 5, "abc123")

    def list_sync_statuses():
        return [("file.txt", "abc123", 5, 5)]

    def upload_document(file, key, overwrite, include_url):
        return document_from_head(key or "uploaded.txt", include_url)

    def update_document(key, file, create_if_missing, include_url):
        return document_from_head(key, include_url), False

    def delete_document(key):
        return {"key": key, "deleted": True}

    fake_documents_service.document_from_head = document_from_head
    fake_documents_service.list_documents = list_documents
    fake_documents_service.get_document_sync_status = get_sync_status
    fake_documents_service.list_sync_statuses = list_sync_statuses
    fake_documents_service.upload_document = upload_document
    fake_documents_service.update_document = update_document
    fake_documents_service.delete_document = delete_document

    monkeypatch.setitem(
        __import__("sys").modules,
        "src.app.features.documents.service",
        fake_documents_service,
    )

    fake_pinecone_service_mod = types.SimpleNamespace()

    class FakePinecone:
        def get_sync_status(self, key, etag):
            return "in_sync"

    def get_pinecone_service():
        return FakePinecone()

    fake_pinecone_service_mod.get_pinecone_service = get_pinecone_service
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.services.vectorstores.pinecone_service",
        fake_pinecone_service_mod,
    )

    # Ensure a fresh import of the app and router with stubs applied
    sys_modules = __import__("sys").modules
    for mod in [
        "src.app.main",
        "src.app.features.documents.api",
    ]:
        sys_modules.pop(mod, None)

    from src.app.main import app as fastapi_app

    return fastapi_app


def test_list_documents(app):
    client = TestClient(app)
    res = client.get("/documents")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list) and data
    assert data[0]["key"] == "file.txt"


def test_get_document_sync(app):
    client = TestClient(app)
    res = client.get("/documents/file.txt/sync")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "in_sync"
    assert data["vectors_for_doc_id"] == 5


def test_list_documents_sync(app):
    client = TestClient(app)
    res = client.get("/documents/sync")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list) and data
    assert data[0]["key"] == "file.txt"


def test_delete_document(app):
    client = TestClient(app)
    res = client.delete("/documents/file.txt")
    assert res.status_code == 200
    assert res.json() == {"key": "file.txt", "deleted": True}
