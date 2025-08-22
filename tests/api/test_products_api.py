import types
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    # Stub graph/runtime to avoid external LLM/OpenAI and retriever tools
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

    # Stub products service BEFORE importing the app/router
    fake_products_service = types.SimpleNamespace()

    product = {
        "id": 1,
        "name": "Geo Drone Pro",
        "category": "Aircraft",
        "price": 9999.99,
        "status": "current",
        "model": "GDP-2025",
        "aliases": ["GDP", "Geo Drone Pro"],
        "tags": ["survey", "inspection"],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": None,
    }

    def list_products(**kwargs):
        return [product]

    def get_product(product_id: int):
        if product_id != 1:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Product not found")
        return product

    def create_product(payload):
        return {**product, "id": 2, **payload.model_dump()}

    def replace_product(product_id: int, payload):
        return {**product, "id": product_id, **payload.model_dump()}

    def update_product(product_id: int, payload):
        return {**product, "id": product_id, **payload.model_dump(exclude_none=True)}

    def delete_product(product_id: int):
        return {"id": product_id, "deleted": True}

    fake_products_service.list_products = list_products
    fake_products_service.get_product = get_product
    fake_products_service.create_product = create_product
    fake_products_service.replace_product = replace_product
    fake_products_service.update_product = update_product
    fake_products_service.delete_product = delete_product

    monkeypatch.setitem(
        __import__("sys").modules,
        "src.app.features.products.service",
        fake_products_service,
    )

    # Ensure a fresh import of the app and router with stubs applied
    sys_modules = __import__("sys").modules
    for mod in [
        "src.app.main",
        "src.app.features.products.api",
    ]:
        sys_modules.pop(mod, None)

    from src.app.main import app as fastapi_app

    return fastapi_app


def test_list_products(app):
    client = TestClient(app)
    res = client.get("/products")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list) and data
    assert data[0]["name"] == "Geo Drone Pro"


def test_get_product_found(app):
    client = TestClient(app)
    res = client.get("/products/1")
    assert res.status_code == 200
    assert res.json()["id"] == 1


def test_get_product_not_found(app):
    client = TestClient(app)
    res = client.get("/products/999")
    assert res.status_code == 404


def test_create_product(app):
    client = TestClient(app)
    payload = {
        "name": "Controller X",
        "category": "Controller",
        "price": 199.5,
    }
    res = client.post("/products", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == payload["name"]


def test_replace_product(app):
    client = TestClient(app)
    payload = {
        "name": "Drone Y",
        "category": "Aircraft",
        "price": 5000.0,
    }
    res = client.put("/products/5", json=payload)
    assert res.status_code == 200
    assert res.json()["id"] == 5


def test_update_product_partial(app):
    client = TestClient(app)
    payload = {"price": 888.0}
    res = client.patch("/products/1", json=payload)
    assert res.status_code == 200
    assert res.json()["price"] == 888.0


def test_delete_product(app):
    client = TestClient(app)
    res = client.delete("/products/3")
    assert res.status_code == 200
    assert res.json() == {"id": 3, "deleted": True}
