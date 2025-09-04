from __future__ import annotations
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
import asyncio
from src.graph.tracing.langsmith_spans import rrf_fuse
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


class CustomHybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    compressor: BaseDocumentCompressor
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Steps 1 & 2: Dense and Sparse Retrieval (auto-traced by LangChain)
        dense_docs = self.dense_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        sparse_docs = self.sparse_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )

        # Step 3: RRF Fusion (custom traceable span)
        fused_docs = rrf_fuse(dense_docs, sparse_docs, self.top_k)

        # Step 4: Compression (auto-traced by LangChain)
        compressed_docs = self.compressor.compress_documents(
            fused_docs, query, callbacks=run_manager.get_child()
        )

        return compressed_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Steps 1 & 2: Dense and Sparse Retrieval (in parallel)
        async def _aget(ret, _callbacks):
            if hasattr(ret, "aget_relevant_documents"):
                return await ret.aget_relevant_documents(query, callbacks=_callbacks)
            return await asyncio.to_thread(
                ret.get_relevant_documents, query, callbacks=_callbacks
            )

        dense_docs, sparse_docs = await asyncio.gather(
            _aget(self.dense_retriever, run_manager.get_child()),
            _aget(self.sparse_retriever, run_manager.get_child()),
        )

        # Step 3: RRF Fusion
        fused_docs = rrf_fuse(dense_docs, sparse_docs, self.top_k)

        # Step 4: Compression
        compressed_docs = await self.compressor.acompress_documents(
            fused_docs, query, callbacks=run_manager.get_child()
        )

        return compressed_docs
