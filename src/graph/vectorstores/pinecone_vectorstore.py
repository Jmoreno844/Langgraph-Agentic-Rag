from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.graph.ingestion.loader_splitter import load_s3_documents
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from src.settings import settings


pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX)


def create_in_memory_retriever_tool():
    """
    Build and return a LangChain retriever tool using an in-memory vector store
    over the documents loaded from S3.
    """
    docs = load_s3_documents()
    if not docs:
        raise ValueError("No documents loaded from S3.")

    vectorStore = PineconeVectorStore.from_documents(
        documents=docs, embedding=PineconeEmbeddings()
    )

    base_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
    compressor = VoyageAIRerank(
        model="rerank-lite-1", voyageai_api_key=settings.VOYAGE_API_KEY, top_k=3
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    print(retriever.invoke("What is the company's mission?"))

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
