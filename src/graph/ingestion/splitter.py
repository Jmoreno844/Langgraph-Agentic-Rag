from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def get_text_splitter():
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def split_text(text: str, doc_id: str, etag: str):
    splitter = get_text_splitter()
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=c, metadata={"doc_id": doc_id, "etag": etag, "chunk_number": i}
        )
        for i, c in enumerate(chunks, 1)
    ]
