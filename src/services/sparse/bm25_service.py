from typing import Optional, Sequence
from langchain_community.retrievers import BM25Retriever
from src.graph.ingestion.loader_splitter import load_s3_documents


class BM25Service:
    def __init__(self, documents: Optional[Sequence] = None):
        # Load documents from S3 if not provided
        self._documents = (
            list(documents) if documents is not None else load_s3_documents()
        )
        self._retriever = BM25Retriever.from_documents(self._documents)

    def get_retriever(self, k: int = 5) -> BM25Retriever:
        self._retriever.k = k
        return self._retriever
