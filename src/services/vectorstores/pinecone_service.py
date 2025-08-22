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
        self._embedding = PineconeEmbeddings(model=settings.PINECONE_EMBEDDINGS_MODEL)
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
        self._vectorstore.delete(
            filter={"doc_id": {"$eq": doc_id}}, namespace=namespace
        )

    def _extract_count(self, stats: dict) -> int:
        """Extract a vector count from describe_index_stats response.

        Falls back to summing per-namespace counts if total_vector_count is absent.
        """
        total = stats.get("total_vector_count")
        if isinstance(total, int):
            return int(total)
        namespaces = stats.get("namespaces") or {}
        try:
            return sum(int(ns.get("vector_count", 0)) for ns in namespaces.values())
        except Exception:
            return 0

    def _count_via_list(self, index, *, doc_id: str, etag: Optional[str]) -> int:
        """Fallback counter using list() + fetch() + client-side metadata filtering.

        Works for Serverless/Starter indexes that do not support metadata filters
        in describe_index_stats.
        """
        try:
            # Default namespace in serverless stats shows as empty string
            ids = []
            for vec in index.list(namespace=""):
                vid = vec.get("id") if isinstance(vec, dict) else vec
                if isinstance(vid, str):
                    ids.append(vid)
            count = 0
            for i in range(0, len(ids), 100):
                batch = ids[i : i + 100]
                resp = index.fetch(ids=batch, namespace="")
                vectors = resp.get("vectors", {}) if isinstance(resp, dict) else {}
                for v in vectors.values():
                    meta = v.get("metadata") if isinstance(v, dict) else None
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("doc_id") != doc_id:
                        continue
                    if etag is not None and meta.get("etag") != etag:
                        continue
                    count += 1
            return count
        except Exception:
            return 0

    def _count_via_query(self, index, *, doc_id: str, etag: Optional[str]) -> int:
        """Fallback counter using a zero-vector query with metadata filter.

        top_k is capped (commonly 1000), so counts above that will be truncated.
        """
        try:
            stats = index.describe_index_stats()
            dim = int(stats.get("dimension") or 0)
            if dim <= 0:
                return 0
            zero = [0.0] * dim
            flt = {"doc_id": {"$eq": doc_id}}
            if etag is not None:
                flt["etag"] = {"$eq": etag}
            res = index.query(
                vector=zero, top_k=1000, include_metadata=True, filter=flt
            )
            matches = res.get("matches") if isinstance(res, dict) else None
            if isinstance(matches, list):
                return len(matches)
            return 0
        except Exception:
            return 0

    def get_vector_counts(
        self, doc_id: str, etag: Optional[str] = None
    ) -> Tuple[int, int]:
        """Return (count_for_doc_id, count_for_doc_id_and_etag) using describe_index_stats.

        Falls back to list() or zero-vector query when serverless/starter indexes do not support
        metadata filters in describe_index_stats.

        If any error occurs, returns (0, 0).
        """
        try:
            # Lazy import to avoid hard dependency at module import time
            from pinecone import Pinecone
            from pinecone.exceptions.exceptions import PineconeApiException

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX)

            # Try metadata-filtered stats (Enterprise)
            try:
                stats_doc = index.describe_index_stats(
                    filter={"doc_id": {"$eq": doc_id}}
                )
                count_doc = self._extract_count(stats_doc)
            except PineconeApiException:
                # Serverless/Starter doesn't support filtered stats; fallback
                count_doc = self._count_via_list(index, doc_id=doc_id, etag=None)
                if count_doc == 0:
                    count_doc = self._count_via_query(index, doc_id=doc_id, etag=None)

            if etag is None:
                return count_doc, 0

            try:
                stats_both = index.describe_index_stats(
                    filter={"doc_id": {"$eq": doc_id}, "etag": {"$eq": etag}}
                )
                count_both = self._extract_count(stats_both)
            except PineconeApiException:
                count_both = self._count_via_list(index, doc_id=doc_id, etag=etag)
                if count_both == 0:
                    count_both = self._count_via_query(index, doc_id=doc_id, etag=etag)

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
