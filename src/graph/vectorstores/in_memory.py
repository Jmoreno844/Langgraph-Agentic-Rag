from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.graph.ingestion.local_loader_splitter import load_local_documents
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from src.settings import settings
from pathlib import Path
import os


def create_in_memory_retriever_tool():
    """
    Build and return a LangChain retriever tool using an in-memory vector store
    over the documents loaded from the local tests/data directory.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "tests" / "data"
    docs = load_local_documents(str(data_dir))
    if not docs:
        raise ValueError("No documents loaded from local tests/data directory.")

    vectorStore = InMemoryVectorStore.from_documents(
        documents=docs, embedding=OpenAIEmbeddings()
    )

    base_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    use_compression = os.getenv("RAG_USE_COMPRESSION", "true").lower() == "true"
    if use_compression:
        compressor = VoyageAIRerank(
            model="rerank-lite-1", voyageai_api_key=settings.VOYAGE_API_KEY, top_k=3
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_rag_docs",
        description=(
            "Search Aetherix Dynamics product, company and support "
            "documentation and return the most relevant snippets."
        ),
    )
    return retriever_tool, retriever


__all__ = ["create_in_memory_retriever_tool"]
