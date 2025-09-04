from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.graph.ingestion.local_loader_splitter import load_local_documents
from pathlib import Path
from src.graph.retrievers.factory import (
    maybe_wrap_with_compression,
    create_standard_retriever_tool,
)


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

    retriever = maybe_wrap_with_compression(base_retriever)

    retriever_tool = create_standard_retriever_tool(retriever)
    return retriever_tool, retriever


__all__ = ["create_in_memory_retriever_tool"]
