from langchain.retrievers import ContextualCompressionRetriever
from src.graph.vectorstores.in_memory import create_in_memory_retriever_tool
from langchain_voyageai import VoyageAIRerank
from src.settings import settings


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


_, retriever = create_in_memory_retriever_tool()

compressor = VoyageAIRerank(
    model="rerank-lite-1", voyageai_api_key=settings.VOYAGE_API_KEY, top_k=3
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What products does Aetherix Dynamics offer?"
)
pretty_print_docs(compressed_docs)
