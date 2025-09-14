import os

os.environ.setdefault("AWS_ACCESS_KEY_ID", "")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "")
os.environ.setdefault("AWS_S3_RAG_DOCUMENTS_BUCKET", "")
import os
import sys
import asyncio
import types
import pytest
from dotenv import load_dotenv

# Ensure repo root on path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Load environment
load_dotenv()
os.environ.setdefault("RAG_USE_COMPRESSION", "false")  # avoid VoyageAI during tests

from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.tools.retriever import create_retriever_tool

from langgraph.checkpoint.memory import MemorySaver

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

DATASET_PATH = Path("tests/evals/data/synthetic_dataset.csv")
MIN_SCORE_THRESHOLD = 0.5  # start lenient; tighten later


def build_local_retriever(data_dir: Path) -> tuple[BM25Retriever, object]:
    loader = DirectoryLoader(
        str(data_dir), glob="*.txt", loader_cls=TextLoader, show_progress=False
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 4
    tool = create_retriever_tool(
        retriever,
        "retrieve_rag_docs",
        "Search TechForge Components knowledge base for support, policy, and product documentation.",
    )
    return retriever, tool


def load_rows(limit: int | None = 3):
    if not DATASET_PATH.exists():
        pytest.skip(
            f"Dataset not found at {DATASET_PATH}. Run scripts/generate_eval_dataset.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    records = df.to_dict("records")
    return records if limit is None else records[:limit]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", load_rows(3))
async def test_real_graph_rag(row):
    # Build local retriever from tests/data
    data_dir = Path("tests/data")
    retriever, retriever_tool = build_local_retriever(data_dir)

    # Pre-inject a fake pinecone_service module BEFORE importing the graph
    fake_mod_name = "src.services.vectorstores.pinecone_service"
    fake_mod = types.ModuleType(fake_mod_name)

    class _FakePineconeService:
        def get_retriever(self, k: int = 5):
            retriever.k = k
            return retriever

        def get_retriever_tool(self):
            return retriever_tool

    _singleton = _FakePineconeService()

    def get_pinecone_service():
        return _singleton

    fake_mod.get_pinecone_service = get_pinecone_service
    sys.modules[fake_mod_name] = fake_mod

    # Inject fake tool modules to avoid DB dependencies
    tools_mod = types.ModuleType("src.graph.tools.hybrid_retriever")
    tools_mod.retriever_tool = retriever_tool
    sys.modules["src.graph.tools.hybrid_retriever"] = tools_mod

    qp_mod = types.ModuleType("src.graph.tools.query_products")
    qp_mod.query_products_tool = retriever_tool
    sys.modules["src.graph.tools.query_products"] = qp_mod

    cat_mod = types.ModuleType("src.graph.tools.list_product_categories")
    cat_mod.list_product_categories_tool = retriever_tool
    sys.modules["src.graph.tools.list_product_categories"] = cat_mod

    # Now import the real graph which will pick up our injected modules
    from src.graph.graph import graph

    # Compile real graph with in-memory checkpointer
    app = graph.compile(checkpointer=MemorySaver())

    question = row["input"]
    expected_output = str(row.get("expected_output") or "")
    expected_context = [str(row.get("context") or "")]

    # Run the actual graph (async)
    session_id = f"eval-{abs(hash(question)) % 100000}"
    config = {"configurable": {"thread_id": session_id}}
    state = await app.ainvoke({"messages": [("user", question)]}, config)

    actual_output = state["messages"][-1].content

    # Independently get retrieval_context from the SAME retriever instance
    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
    retrieval_context = [d.page_content for d in retrieved_docs]

    # Prepare test case and metrics
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        context=expected_context,
    )

    metrics = [
        FaithfulnessMetric(threshold=MIN_SCORE_THRESHOLD, model="gpt-4o-mini"),
        AnswerRelevancyMetric(threshold=MIN_SCORE_THRESHOLD, model="gpt-4o-mini"),
        ContextualRecallMetric(threshold=MIN_SCORE_THRESHOLD, model="gpt-4o-mini"),
        ContextualRelevancyMetric(threshold=MIN_SCORE_THRESHOLD, model="gpt-4o-mini"),
    ]

    assert_test(test_case, metrics, run_async=False)
