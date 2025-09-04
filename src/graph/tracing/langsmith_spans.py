from typing import List, Optional

from langchain_core.documents import Document
from langsmith import traceable


def _rrf(
    dense_docs: List[Document], sparse_docs: List[Document], k: int = 60
) -> List[Document]:
    scores = {}
    runs = [dense_docs, sparse_docs]
    for run in runs:
        for rank, doc in enumerate(run):
            doc_id = doc.metadata.get("id") or doc.metadata.get("doc_id") or id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered_docs: List[Document] = []
    seen = set()
    for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        for run in runs:
            for doc in run:
                if (
                    doc.metadata.get("id") or doc.metadata.get("doc_id") or id(doc)
                ) == doc_id and doc_id not in seen:
                    ordered_docs.append(doc)
                    seen.add(doc_id)
                    break
            if doc_id in seen:
                break
    return ordered_docs


@traceable(name="RRF Fusion")
def rrf_fuse(
    dense_docs: List[Document], sparse_docs: List[Document], top_k: int
) -> List[Document]:
    return _rrf(dense_docs[:top_k], sparse_docs[:top_k])


@traceable(name="VoyageAI Rerank API")
def voyage_rerank(client, *, query: str, documents: List[str], model: str, top_k: int):
    return client.rerank(query=query, documents=documents, model=model, top_k=top_k)
