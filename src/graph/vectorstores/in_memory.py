from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.graph.ingestion.loader_splitter import load_s3_documents
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from src.settings import settings


def create_in_memory_retriever_tool():
    """
    Build and return a LangChain retriever tool using an in-memory vector store
    over the documents loaded from S3.
    """
    docs = load_s3_documents()
    if not docs:
        raise ValueError("No documents loaded from S3.")

    vectorStore = InMemoryVectorStore.from_documents(
        documents=docs, embedding=OpenAIEmbeddings()
    )

    base_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
    compressor = VoyageAIRerank(
        model="rerank-lite-1", voyageai_api_key=settings.VOYAGE_API_KEY, top_k=3
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

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
