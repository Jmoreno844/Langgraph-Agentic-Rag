from typing import Optional, Sequence, Tuple

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_voyageai import VoyageAIRerank
from src.settings import settings


class PineconeVectorStoreService:
    """
    Service for managing Pinecone-backed vector search.

    - Initializes a reusable vector store bound to an existing Pinecone index
    - Provides retrieval with contextual compression (VoyageAI reranker)
    - Exposes ingestion helpers for upserting and deleting documents
    """

    def __init__(self):
        self._embedding = PineconeEmbeddings()
        self._vectorstore = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX,
            embedding=self._embedding,
        )
        self._reranker = VoyageAIRerank(
            model="rerank-lite-1",
            voyageai_api_key=settings.VOYAGE_API_KEY,
            top_k=3,
        )

    def get_retriever(self, k: int = 5) -> ContextualCompressionRetriever:
        base_retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        return ContextualCompressionRetriever(
            base_compressor=self._reranker,
            base_retriever=base_retriever,
        )

    def get_retriever_tool(self):
        retriever = self.get_retriever()
        return create_retriever_tool(
            retriever,
            "retrieve_rag_docs",
            (
                "Search Aetherix Dynamics product, company and support "
                "documentation and return the most relevant snippets."
            ),
        )

    def upsert_documents(self, docs: Sequence, *, namespace: Optional[str] = None):
        """Add or update documents in the Pinecone index."""
        self._vectorstore.add_documents(docs, namespace=namespace)

    def delete_by_ids(self, ids: Sequence[str], *, namespace: Optional[str] = None):
        """Delete vectors by their document IDs."""
        self._vectorstore.delete(ids=list(ids), namespace=namespace)

    def delete_by_doc_id(self, doc_id: str, *, namespace: Optional[str] = None):
        """Delete all vectors whose metadata matches the provided doc_id."""
        self._vectorstore.delete(filter={"doc_id": doc_id}, namespace=namespace)

    def get_vector_counts(
        self, doc_id: str, etag: Optional[str] = None
    ) -> Tuple[int, int]:
        """Return (count_for_doc_id, count_for_doc_id_and_etag) using describe_index_stats.

        If any error occurs, returns (0, 0).
        """
        try:
            # Lazy import to avoid hard dependency at module import time
            from pinecone import Pinecone

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX)

            stats_doc = index.describe_index_stats(filter={"doc_id": doc_id})
            count_doc = int(stats_doc.get("total_vector_count", 0))

            if etag is None:
                return count_doc, 0

            stats_both = index.describe_index_stats(
                filter={"doc_id": doc_id, "etag": etag}
            )
            count_both = int(stats_both.get("total_vector_count", 0))
            return count_doc, count_both
        except Exception:
            return 0, 0

    def get_sync_status(self, doc_id: str, etag: Optional[str]) -> str:
        """Compute a simple sync status label based on counts.

        - "in_sync": vectors exist for this doc_id and match the current etag
        - "stale": vectors exist for doc_id, but none match etag
        - "not_indexed": no vectors exist for doc_id
        """
        count_doc, count_both = self.get_vector_counts(doc_id, etag)
        if count_doc == 0:
            return "not_indexed"
        if count_both > 0:
            return "in_sync"
        return "stale"


_service: Optional[PineconeVectorStoreService] = None


def get_pinecone_service() -> PineconeVectorStoreService:
    global _service
    if _service is None:
        _service = PineconeVectorStoreService()
    return _service
