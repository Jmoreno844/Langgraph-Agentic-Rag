# src/tools/retriever.py
import os

# Select vector backend: "pinecone" or "in_memory" (default)
_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "RAG_VECTOR_BACKEND").lower()

if _BACKEND == "pinecone":
    from src.services.vectorstores.pinecone_service import get_pinecone_service

    # Create the tool once from the Pinecone-backed service
    retriever_tool = get_pinecone_service().get_retriever_tool()
else:
    from src.graph.vectorstores.in_memory import create_in_memory_retriever_tool

    # Create the tool once from the in-memory vectorstore
    retriever_tool, _ = create_in_memory_retriever_tool()
