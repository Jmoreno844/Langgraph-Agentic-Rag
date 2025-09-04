from typing import Optional
import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIRerank
from src.settings import settings
from .batch_voyage_compressor import BatchVoyageCompressor


def build_batch_voyage_compressor(
    model: str = "rerank-lite-1", top_k: int = 3
) -> BatchVoyageCompressor:
    """Construct a batch VoyageAI compressor for better performance."""
    return BatchVoyageCompressor(
        model=model,
        voyage_api_key=settings.VOYAGE_API_KEY,
        top_k=top_k,
    )


def build_voyage_reranker(
    model: str = "rerank-lite-1", top_k: int = 3
) -> VoyageAIRerank:
    """Construct a VoyageAI reranker using project settings."""
    return VoyageAIRerank(
        model=model,
        voyageai_api_key=settings.VOYAGE_API_KEY,
        top_k=top_k,
    )


def maybe_wrap_with_compression(
    base_retriever, use_compression: Optional[bool] = None, *, top_k: int = 3
):
    """Optionally wrap a base retriever with contextual compression using VoyageAI.

    If use_compression is None, the value is read from RAG_USE_COMPRESSION (default true).
    Set RAG_USE_BATCH_COMPRESSION=true to use the optimized batch compressor (default).
    Set RAG_USE_BATCH_COMPRESSION=false to use individual API calls.
    """
    if use_compression is None:
        use_compression = os.getenv("RAG_USE_COMPRESSION", "true").lower() == "true"
    if not use_compression:
        return base_retriever

    # Choose compressor based on environment variable
    use_batch = os.getenv("RAG_USE_BATCH_COMPRESSION", "true").lower() == "true"

    if use_batch:
        print("🚀 Using optimized BatchVoyageCompressor (single API call)")
        compressor = build_batch_voyage_compressor(top_k=top_k)
    else:
        print("🐌 Using standard VoyageAIRerank (individual API calls)")
        compressor = build_voyage_reranker(top_k=top_k)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def create_standard_retriever_tool(retriever):
    """Create a standardized retriever tool with consistent naming and description."""
    from langchain.tools.retriever import create_retriever_tool

    return create_retriever_tool(
        retriever,
        "retrieve_rag_docs",
        (
            "Search TechForge Components knowledge base: product, company, support, and store policy docs. "
            "Use this when the user asks about policies (refunds, returns, warranty, shipping), FAQs, setup/support steps, or documentation details. "
            "Input: the raw user question. Output: the most relevant snippets for answering."
        ),
    )


def create_timed_retriever_tool(retriever, name: str = "retrieve_rag_docs"):
    """Return a standard retriever tool; timing is handled by LangSmith spans in components."""
    return create_standard_retriever_tool(retriever)
