from typing import Optional, Sequence
from langchain_community.retrievers import BM25Retriever
from src.graph.ingestion.s3_loader import load_s3_documents


class BM25Service:
    def __init__(self, documents: Optional[Sequence] = None):
        # Load documents from S3 if not provided
        self._documents = (
            list(documents) if documents is not None else load_s3_documents()
        )
        
        # Handle empty documents gracefully
        if not self._documents:
            print("Warning: No documents available for BM25 retriever")
            self._retriever = None
        else:
            self._retriever = BM25Retriever.from_documents(self._documents)

    def get_retriever(self, k: int = 5) -> Optional[BM25Retriever]:
        if self._retriever is None:
            return None
        self._retriever.k = k
        return self._retriever
