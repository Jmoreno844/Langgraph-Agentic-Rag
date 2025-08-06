from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.ingestion.loader_splitter import load_s3_documents
from langchain.tools.retriever import create_retriever_tool


def create_in_memory_retriever_tool():
    """
    Build and return a LangChain retriever tool using an in-memory vector store
    over the documents loaded from S3.
    """
    docs = load_s3_documents()
    if not docs:
        raise ValueError("No documents loaded from S3.")

    vs = InMemoryVectorStore.from_documents(
        documents=docs, embedding=OpenAIEmbeddings()
    )
    retriever = vs.as_retriever()

    return create_retriever_tool(
        retriever,
        "retrieve_rag_docs",
        description=(
            "Search Aetherix Dynamics product, company and support "
            "documentation and return the most relevant snippets."
        ),
    )


__all__ = ["create_in_memory_retriever_tool"]
