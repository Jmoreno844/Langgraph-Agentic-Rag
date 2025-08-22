from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_local_documents(directory_path: str):
    """
    Load local documents from a directory path (e.g., tests/data), split them,
    and return chunks compatible with the vector store.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

    loader = DirectoryLoader(directory_path, glob="**/*.txt")
    docs = loader.load()

    docs_list = []
    if isinstance(docs, list):
        for document in docs:
            docs_list.append(document)
    else:
        docs_list.append(docs)

    # Use the shared splitter to keep chunking consistent (500 size, 100 overlap)
    from src.graph.ingestion.splitter import get_text_splitter

    text_splitter = get_text_splitter()
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


__all__ = ["load_local_documents"]
