from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class Document(BaseModel):
    key: str = Field(
        ...,
        description="Full S3 object key, e.g., 'folder/file.txt'. Also used as the Pinecone doc_id.",
    )
    name: str = Field(..., description="File name derived from the key")
    size: int = Field(..., description="Object size in bytes")
    etag: str = Field(
        ...,
        description="Entity tag (ETag) identifying the object version; also stored alongside vector chunks for traceability.",
    )
    last_modified: datetime = Field(
        ..., description="Timestamp when the object was last modified (ISO 8601)"
    )
    storage_class: Optional[str] = Field(
        None, description="S3 storage class, e.g., STANDARD"
    )
    checksum_algorithm: Optional[List[str]] = Field(
        None, description="Checksum algorithms associated with the object"
    )
    checksum_type: Optional[str] = Field(
        None, description="Checksum type for the object"
    )
    url: Optional[str] = Field(
        None,
        description="Presigned URL to download the object if requested (time-limited)",
    )


class UploadResult(BaseModel):
    key: str
    etag: str


class DeleteResult(BaseModel):
    key: str
    deleted: bool = Field(
        ...,
        description="Whether the S3 object was deleted. Pinecone vector cleanup is attempted separately and does not affect this flag.",
    )


class SyncStatus(BaseModel):
    key: str
    etag: Optional[str] = None
    status: str = Field(
        ...,
        description="One of: in_sync (vectors present and match etag), stale (vectors present but no match on etag), not_indexed (no vectors).",
    )
    vectors_for_doc_id: int = Field(
        ..., description="Pinecone vector count for this doc_id regardless of etag"
    )
    vectors_for_doc_id_and_etag: int = Field(
        ..., description="Pinecone vector count for this doc_id and current etag"
    )


class SyncResult(BaseModel):
    total_s3_documents: int
    total_pinecone_documents: int
    synced: int
    added: int
    updated: int
    deleted: int
    errors: List[str]
    debug: Optional[List[str]] = Field(
        default=None,
        description="Debug log lines emitted during sync (present when debug=true)",
    )
