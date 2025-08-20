import sys
import types
import pytest


# Pre-stub external modules used at import time to keep tests hermetic
_fake_voyage_module = types.ModuleType("langchain_voyageai")


class _FakeVoyageAIRerank:
    def __init__(self, *args, **kwargs):
        pass


_fake_voyage_module.VoyageAIRerank = _FakeVoyageAIRerank
sys.modules.setdefault("langchain_voyageai", _fake_voyage_module)

# Provide a placeholder loader module so import succeeds; individual tests will monkeypatch
_fake_loader_module = types.ModuleType("src.graph.ingestion.local_loader_splitter")


def _placeholder_loader(_path: str):
    return []


_fake_loader_module.load_local_documents = _placeholder_loader
sys.modules.setdefault("src.graph.ingestion.local_loader_splitter", _fake_loader_module)


def test_create_in_memory_retriever_tool_raises_when_no_docs(monkeypatch):
    from src.graph.vectorstores import in_memory as mod

    # Force loader to return no documents
    monkeypatch.setattr(mod, "load_local_documents", lambda _dir: [], raising=True)

    with pytest.raises(ValueError) as exc:
        mod.create_in_memory_retriever_tool()

    assert "No documents loaded" in str(exc.value)


def test_create_in_memory_retriever_tool_happy_path(monkeypatch):
    from types import SimpleNamespace
    from src.graph.vectorstores import in_memory as mod

    # Build fake docs
    fake_doc = SimpleNamespace(page_content="hello", metadata={})
    monkeypatch.setattr(
        mod, "load_local_documents", lambda _dir: [fake_doc], raising=True
    )

    # Stub embeddings + vector store
    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        def __init__(self, docs, embedding):
            self.docs = docs
            self.embedding = embedding

        @classmethod
        def from_documents(cls, documents, embedding):
            return cls(documents, embedding)

        def as_retriever(self, search_kwargs=None):
            return SimpleNamespace(invoke=lambda _q: [fake_doc])

    # Stub reranker to return a simple identity compressor
    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

    # Fake contextual compression retriever that just stores inputs
    class FakeCCR:
        def __init__(self, base_compressor, base_retriever):
            self.base_compressor = base_compressor
            self.base_retriever = base_retriever

    # Capture the tool creation args
    created = {"name": None, "description": None, "retriever": None}

    def fake_create_retriever_tool(retriever, name, description):
        created["name"] = name
        created["description"] = description
        created["retriever"] = retriever
        return SimpleNamespace(name=name, description=description, retriever=retriever)

    monkeypatch.setattr(mod, "OpenAIEmbeddings", FakeEmbeddings, raising=True)
    monkeypatch.setattr(mod, "InMemoryVectorStore", FakeVectorStore, raising=True)
    monkeypatch.setattr(mod, "VoyageAIRerank", FakeReranker, raising=True)
    monkeypatch.setattr(mod, "ContextualCompressionRetriever", FakeCCR, raising=True)
    monkeypatch.setattr(
        mod, "create_retriever_tool", fake_create_retriever_tool, raising=True
    )

    tool, retriever = mod.create_in_memory_retriever_tool()

    assert tool.name == "retrieve_rag_docs"
    assert "Search Aetherix Dynamics" in tool.description
    assert created["retriever"] is not None
    # sanity check retriever behavior
    assert retriever.base_retriever.invoke("q")[0] is fake_doc
