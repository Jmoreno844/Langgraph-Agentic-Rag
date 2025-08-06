from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from src.ingestion.loader_splitter import load_s3_documents
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()


def test_memory_vectorstore():
    """
    Test function to verify S3 document loading and vector store creation.
    """
    try:
        docs = load_s3_documents()
        if not docs:
            raise ValueError("No documents loaded from S3.")

        vectorStore = InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
        )

        retriever = vectorStore.as_retriever()

        # Test retrieval
        results = retriever.invoke("What is your warranty policy?")
        print(f"Retrieved {len(results)} documents")
        for i, doc in enumerate(results[:2]):  # Show first 2 results
            print(f"Document {i+1}: {doc.page_content[:200]}...")

        return retriever

    except Exception as e:
        print(f"Error in test: {e}")
        return None


if __name__ == "__main__":
    test_memory_vectorstore()
