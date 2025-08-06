# loaders.py
from langchain_community.document_loaders import S3DirectoryLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_s3_documents():
    load_dotenv()
    AWS_ACCESS_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_S3_RAG_DOCUMENTS_BUCKET = os.getenv("AWS_S3_RAG_DOCUMENTS_BUCKET")

    if not all([AWS_ACCESS_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_RAG_DOCUMENTS_BUCKET]):
        raise ValueError("Missing required AWS environment variables")

    loader = S3DirectoryLoader(
        AWS_S3_RAG_DOCUMENTS_BUCKET,
        aws_access_key_id=AWS_ACCESS_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    docs = loader.load()
    docs_list = []

    # Handle both single documents and lists of documents
    if isinstance(docs, list):
        for document in docs:
            docs_list.append(document)
    else:
        docs_list.append(docs)

    print(f"Number of documents loaded: {len(docs_list)}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


# Export the function
__all__ = ["load_s3_documents"]
