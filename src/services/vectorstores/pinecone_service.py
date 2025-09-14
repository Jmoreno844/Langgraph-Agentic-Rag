from typing import Optional, Sequence, Tuple
import ast

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from src.graph.retrievers.factory import (
    maybe_wrap_with_compression,
    create_standard_retriever_tool,
)
from src.settings import settings


class PineconeVectorStoreService:
    """
    Service for managing Pinecone-backed vector search.

    - Initializes a reusable vector store bound to an existing Pinecone index
    - Provides retrieval with contextual compression (VoyageAI reranker)
    - Exposes ingestion helpers for upserting and deleting documents
    """

    def __init__(self):
        print("--- Initializing PineconeVectorStoreService ---")
        self._embedding = PineconeEmbeddings(model=settings.PINECONE_EMBEDDINGS_MODEL)
        self._vectorstore = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX,
            embedding=self._embedding,
        )

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        """Normalize namespace to empty-string when None is provided."""
        resolved = "" if namespace is None else namespace
        print(f"--- Resolving namespace: input={namespace}, output={resolved!r} ---")
        return resolved

    def _list_all_vector_ids(self, index, namespace: str) -> list[str]:
        """Return a best-effort list of vector IDs for the given namespace."""
        ids: list[str] = []
        try:
            print(f"--- Listing vectors from namespace: {namespace!r} ---")
            item_count = 0
            for item in index.list(namespace=namespace):
                if item_count < 5:  # Log first few items
                    print(
                        f"  - index.list item type: {type(item)}, value: {str(item)[:100]}"
                    )
                item_count += 1
                if isinstance(item, (list, tuple, set)):
                    for elem in item:
                        if isinstance(elem, str):
                            ids.append(elem)
                    continue
                if isinstance(item, dict):
                    candidate = item.get("id")
                    if isinstance(candidate, str) and candidate:
                        ids.append(candidate)
                elif isinstance(item, str):
                    ids.append(item.strip())
                else:
                    candidate = getattr(item, "id", None)
                    if isinstance(candidate, str) and candidate:
                        ids.append(candidate)
        except Exception as e:
            print(f"!!! EXCEPTION in _list_all_vector_ids: {e}")
            return []
        print(f"--- Found {len(ids)} total vector IDs in namespace {namespace!r}. ---")
        seen: set[str] = set()
        unique_ids: list[str] = [x for x in ids if not (x in seen or seen.add(x))]
        return unique_ids

    def get_retriever(self, k: int = 5):
        base_retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        return base_retriever

    def get_retriever_tool(self):
        retriever = self.get_retriever()
        return create_standard_retriever_tool(retriever)

    def upsert_documents(self, docs: Sequence, *, namespace: Optional[str] = None):
        """Add or update documents in the Pinecone index."""
        ns = self._resolve_namespace(namespace)
        self._vectorstore.add_documents(docs, namespace=ns)

    def delete_by_ids(self, ids: Sequence[str], *, namespace: Optional[str] = None):
        """Delete vectors by their document IDs."""
        ns = self._resolve_namespace(namespace)
        self._vectorstore.delete(ids=list(ids), namespace=ns)

    def delete_by_doc_id(self, doc_id: str, *, namespace: Optional[str] = None):
        """Delete all vectors whose metadata matches the provided doc_id.

        Tries server-side filtered delete first; falls back to list+fetch and
        explicit ID deletion when necessary (e.g., serverless limitations).
        """
        ns = self._resolve_namespace(namespace)
        try:
            self._vectorstore.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=ns)
            return
        except Exception:
            pass
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX)

            all_ids: list[str] = self._list_all_vector_ids(index, ns)
            if not all_ids:
                return

            ids_to_delete: list[str] = []
            for i in range(0, len(all_ids), 100):
                batch = all_ids[i : i + 100]
                try:
                    resp = index.fetch(ids=batch, namespace=ns)
                    if not hasattr(resp, "vectors"):
                        continue
                    for vid, v in resp.vectors.items():
                        if (
                            hasattr(v, "metadata")
                            and v.metadata
                            and v.metadata.get("doc_id") == doc_id
                        ):
                            ids_to_delete.append(vid)
                except Exception:
                    continue

            if ids_to_delete:
                index.delete(ids=ids_to_delete, namespace=ns)
        except Exception:
            pass

    def _extract_count(self, stats: dict) -> int:
        total = stats.get("total_vector_count")
        if isinstance(total, int):
            return int(total)
        namespaces = stats.get("namespaces") or {}
        try:
            return sum(int(ns.get("vector_count", 0)) for ns in namespaces.values())
        except Exception:
            return 0

    def _count_via_list(
        self, index, *, doc_id: str, etag: Optional[str], namespace: str
    ) -> int:
        try:
            ids = self._list_all_vector_ids(index, namespace)
            if not ids:
                return 0
            count = 0
            for i in range(0, len(ids), 100):
                batch = ids[i : i + 100]
                resp = index.fetch(ids=batch, namespace=namespace)
                if not hasattr(resp, "vectors"):
                    continue
                for v in resp.vectors.values():
                    if not (hasattr(v, "metadata") and v.metadata):
                        continue
                    meta = v.metadata
                    if meta.get("doc_id") != doc_id:
                        continue
                    if etag is not None and meta.get("etag") != etag:
                        continue
                    count += 1
            return count
        except Exception:
            return 0

    def _count_via_query(
        self, index, *, doc_id: str, etag: Optional[str], namespace: str
    ) -> int:
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
                vector=zero,
                top_k=1000,
                include_metadata=True,
                filter=flt,
                namespace=namespace,
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
        try:
            from pinecone import Pinecone
            from pinecone.exceptions.exceptions import PineconeApiException

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX)

            try:
                stats_doc = index.describe_index_stats(
                    filter={"doc_id": {"$eq": doc_id}}
                )
                count_doc = self._extract_count(stats_doc)
            except PineconeApiException:
                namespace = self._resolve_namespace(None)
                count_doc = self._count_via_list(
                    index, doc_id=doc_id, etag=None, namespace=namespace
                )
                if count_doc == 0:
                    count_doc = self._count_via_query(
                        index, doc_id=doc_id, etag=None, namespace=namespace
                    )

            if etag is None:
                return count_doc, 0

            try:
                stats_both = index.describe_index_stats(
                    filter={"doc_id": {"$eq": doc_id}, "etag": {"$eq": etag}}
                )
                count_both = self._extract_count(stats_both)
            except PineconeApiException:
                namespace = self._resolve_namespace(None)
                count_both = self._count_via_list(
                    index, doc_id=doc_id, etag=etag, namespace=namespace
                )
                if count_both == 0:
                    count_both = self._count_via_query(
                        index, doc_id=doc_id, etag=etag, namespace=namespace
                    )

            return count_doc, count_both
        except Exception:
            return 0, 0

    def compute_index_snapshot(
        self, *, namespace: Optional[str] = None
    ) -> tuple[set[str], dict[str, int], dict[tuple[str, str], int]]:
        """Compute a snapshot of the index in one pass.

        Returns:
            - unique_doc_ids: set[str]
            - doc_id_to_count: dict[doc_id, count]
            - doc_id_etag_to_count: dict[(doc_id, etag), count]
        """
        try:
            from pinecone import Pinecone

            ns = self._resolve_namespace(namespace)
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX)

            vector_ids: list[str] = self._list_all_vector_ids(index, ns)
            unique_doc_ids: set[str] = set()
            doc_id_to_count: dict[str, int] = {}
            doc_id_etag_to_count: dict[tuple[str, str], int] = {}

            for i in range(0, len(vector_ids), 100):
                batch = vector_ids[i : i + 100]
                try:
                    resp = index.fetch(ids=batch, namespace=ns)
                    if not hasattr(resp, "vectors"):
                        continue
                    for v in resp.vectors.values():
                        if not (hasattr(v, "metadata") and v.metadata):
                            continue
                        meta = v.metadata
                        did = meta.get("doc_id")
                        et = meta.get("etag")
                        if isinstance(did, str) and did:
                            unique_doc_ids.add(did)
                            doc_id_to_count[did] = doc_id_to_count.get(did, 0) + 1
                            if isinstance(et, str) and et:
                                key = (did, et)
                                doc_id_etag_to_count[key] = (
                                    doc_id_etag_to_count.get(key, 0) + 1
                                )
                except Exception:
                    continue

            return unique_doc_ids, doc_id_to_count, doc_id_etag_to_count
        except Exception as e:
            print(f"Error computing index snapshot: {e}")
            return set(), {}, {}

    def get_all_indexed_doc_ids(self, *, namespace: Optional[str] = None) -> list[str]:
        try:
            ns = self._resolve_namespace(namespace)
            unique_doc_ids, _, _ = self.compute_index_snapshot(namespace=ns)
            return list(unique_doc_ids)
        except Exception as e:
            print(f"Error getting indexed doc_ids: {e}")
            return []

    def get_sync_status(self, doc_id: str, etag: Optional[str]) -> str:
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
