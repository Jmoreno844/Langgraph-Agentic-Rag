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
    key: str = Field(..., description="S3 object key for the uploaded file")
    etag: str = Field(..., description="ETag returned by S3 for the upload")


class DeleteResult(BaseModel):
    key: str = Field(..., description="S3 object key that was deleted")
    deleted: bool = Field(
        ...,
        description="Whether the S3 object was deleted. Pinecone vector cleanup is attempted separately and does not affect this flag.",
    )


class SyncStatus(BaseModel):
    key: str = Field(..., description="S3 key (also Pinecone doc_id)")
    etag: Optional[str] = Field(None, description="Current S3 ETag for the object")
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
