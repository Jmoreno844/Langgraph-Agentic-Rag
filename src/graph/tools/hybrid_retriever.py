# src/graph/tools/hybrid_retriever.py
from src.services.vectorstores.pinecone_service import get_pinecone_service
from src.services.sparse.bm25_service import BM25Service
from src.graph.retrievers.hybrid import CustomHybridRetriever
from src.graph.retrievers.factory import (
    create_standard_retriever_tool,
    build_batch_voyage_compressor,
)

# 1. Build individual components
dense_retriever = get_pinecone_service().get_retriever()
sparse_retriever = BM25Service().get_retriever()

# Handle case where BM25 is not available (no documents)
if sparse_retriever is None:
    print("Warning: BM25 retriever not available (no documents found)")
    # Fall back to dense-only retrieval
    from langchain.retrievers import MergerRetriever
    _hybrid = MergerRetriever(retrievers=[dense_retriever])
else:
    compressor = build_batch_voyage_compressor()
    # 2. Assemble the all-in-one custom retriever
    _hybrid = CustomHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        compressor=compressor,
    )

# 3. Expose as a standard LangChain tool
retriever_tool = create_standard_retriever_tool(_hybrid)
