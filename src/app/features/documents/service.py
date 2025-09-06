from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile
import asyncio
from src.services.s3_service import (
    get_s3_bucket_contents,
    create_presigned_url,
    upload_fileobj_to_s3,
    head_object_from_s3,
    object_exists_in_s3,
    delete_file_from_s3,
    get_object_bytes_from_s3,
)
from src.settings import settings
from .schemas import Document, DeleteResult
from src.graph.ingestion.splitter import split_text
from src.services.vectorstores.pinecone_service import get_pinecone_service


def document_from_head(key: str, include_url: bool = False) -> Document:
    head = head_object_from_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
    return Document(
        key=key,
        name=key.split("/")[-1],
        size=head.get("ContentLength", 0),
        etag=head.get("ETag", "").strip('"'),
        last_modified=head.get("LastModified"),
        storage_class=head.get("StorageClass"),
        checksum_algorithm=head.get("ChecksumAlgorithm"),
        checksum_type=head.get("ChecksumType"),
        url=(
            create_presigned_url(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
            if include_url
            else None
        ),
    )


def _ingest_key_into_pinecone(key: str, etag: str) -> None:
    raw_bytes = get_object_bytes_from_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return
    docs = split_text(text=text, doc_id=key, etag=etag)
    service = get_pinecone_service()
    service.upsert_documents(docs)


def _delete_key_from_pinecone(key: str) -> None:
    service = get_pinecone_service()
    service.delete_by_doc_id(key)


async def list_documents(include_url: bool) -> list[Document]:
    contents = await asyncio.to_thread(
        get_s3_bucket_contents, settings.AWS_S3_RAG_DOCUMENTS_BUCKET
    )
    return [
        Document(
            key=obj["Key"],
            name=obj["Key"].split("/")[-1],
            size=obj["Size"],
            etag=obj.get("ETag", "").strip('"'),
            last_modified=obj["LastModified"],
            storage_class=obj.get("StorageClass"),
            checksum_algorithm=obj.get("ChecksumAlgorithm"),
            checksum_type=obj.get("ChecksumType"),
            url=(
                create_presigned_url(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, obj["Key"])
                if include_url
                else None
            ),
        )
        for obj in contents
    ]


async def get_document_sync_status(key: str) -> tuple[str, int, int, str]:
    """Return (status, vectors_for_doc_id, vectors_for_doc_id_and_etag, etag)."""
    head = await asyncio.to_thread(
        head_object_from_s3, settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key
    )
    etag = head.get("ETag", "").strip('"')
    service = get_pinecone_service()
    count_doc, count_both = await asyncio.to_thread(
        service.get_vector_counts, key, etag
    )
    status = await asyncio.to_thread(service.get_sync_status, key, etag)
    return status, count_doc, count_both, etag


async def list_sync_statuses() -> list[tuple[str, str, int, int]]:
    """Return list of (key, etag, vectors_for_doc_id, vectors_for_doc_id_and_etag)."""
    contents = await asyncio.to_thread(
        get_s3_bucket_contents, settings.AWS_S3_RAG_DOCUMENTS_BUCKET
    )
    service = get_pinecone_service()
    results: list[tuple[str, str, int, int]] = []
    for obj in contents:
        key = obj["Key"]
        etag = obj.get("ETag", "").strip('"')
        c_doc, c_both = await asyncio.to_thread(service.get_vector_counts, key, etag)
        results.append((key, etag, c_doc, c_both))
    return results


async def upload_document(
    file: UploadFile,
    key: Optional[str],
    overwrite: bool,
    include_url: bool,
) -> Document:
    resolved_key = key or file.filename
    if not resolved_key:
        raise HTTPException(status_code=400, detail="A key or filename is required")

    exists = await asyncio.to_thread(
        object_exists_in_s3, settings.AWS_S3_RAG_DOCUMENTS_BUCKET, resolved_key
    )
    if not overwrite and exists:
        raise HTTPException(
            status_code=409, detail="Object already exists; use overwrite=true"
        )

    await file.seek(0)
    await asyncio.to_thread(
        upload_fileobj_to_s3,
        settings.AWS_S3_RAG_DOCUMENTS_BUCKET,
        resolved_key,
        file.file,
        content_type=file.content_type,
    )
    document = await asyncio.to_thread(document_from_head, resolved_key, include_url)

    # Keep Pinecone in sync
    try:
        if overwrite:
            await asyncio.to_thread(_delete_key_from_pinecone, resolved_key)
        await asyncio.to_thread(_ingest_key_into_pinecone, resolved_key, document.etag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vectorstore ingest failed: {exc}")

    return document


async def update_document(
    key: str,
    file: UploadFile,
    create_if_missing: bool,
    include_url: bool,
) -> Tuple[Document, bool]:
    exists = await asyncio.to_thread(
        object_exists_in_s3, settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key
    )
    if not exists and not create_if_missing:
        raise HTTPException(status_code=404, detail="Object not found")

    await file.seek(0)
    await asyncio.to_thread(
        upload_fileobj_to_s3,
        settings.AWS_S3_RAG_DOCUMENTS_BUCKET,
        key,
        file.file,
        content_type=file.content_type,
    )
    document = await asyncio.to_thread(document_from_head, key, include_url)
    created = not exists

    # Replace vectors for this doc
    try:
        await asyncio.to_thread(_delete_key_from_pinecone, key)
        await asyncio.to_thread(_ingest_key_into_pinecone, key, document.etag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vectorstore update failed: {exc}")

    return document, created


def delete_document(key: str) -> DeleteResult:
    if not object_exists_in_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key):
        raise HTTPException(status_code=404, detail="Object not found")

    deleted = delete_file_from_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
    # Best-effort vectorstore cleanup; do not fail deletion if vectorstore call fails
    try:
        _delete_key_from_pinecone(key)
    except Exception:
        pass
    return DeleteResult(key=key, deleted=bool(deleted))


async def sync_documents() -> dict:
    """Synchronize S3 documents with Pinecone vectorstore.

    Returns a dictionary with sync statistics including counts for synced, added, updated, and deleted documents.
    """
    from src.app.features.documents.schemas import SyncResult

    # Get all S3 documents
    s3_contents = await asyncio.to_thread(
        get_s3_bucket_contents, settings.AWS_S3_RAG_DOCUMENTS_BUCKET
    )
    s3_keys = {obj["Key"] for obj in s3_contents}

    # Get all indexed Pinecone documents
    pinecone_service = get_pinecone_service()
    pinecone_doc_ids = set(
        await asyncio.to_thread(pinecone_service.get_all_indexed_doc_ids)
    )

    # Initialize counters
    synced = 0
    added = 0
    updated = 0
    deleted = 0
    errors = []

    # Process S3 documents
    for obj in s3_contents:
        key = obj["Key"]
        etag = obj.get("ETag", "").strip('"')

        try:
            # Check sync status
            status, _, _, _ = await get_document_sync_status(key)

            if status == "not_indexed":
                # Document not in Pinecone, add it
                await asyncio.to_thread(_ingest_key_into_pinecone, key, etag)
                added += 1
            elif status == "stale":
                # Document exists but etag changed, update it
                await asyncio.to_thread(_delete_key_from_pinecone, key)
                await asyncio.to_thread(_ingest_key_into_pinecone, key, etag)
                updated += 1
            elif status == "in_sync":
                # Document is already synced
                synced += 1
        except Exception as e:
            errors.append(f"Error processing {key}: {str(e)}")

    # Remove orphaned vectors (in Pinecone but not in S3)
    orphaned_doc_ids = pinecone_doc_ids - s3_keys
    for doc_id in orphaned_doc_ids:
        try:
            await asyncio.to_thread(_delete_key_from_pinecone, doc_id)
            deleted += 1
        except Exception as e:
            errors.append(f"Error deleting orphaned {doc_id}: {str(e)}")

    return {
        "total_s3_documents": len(s3_keys),
        "total_pinecone_documents": len(pinecone_doc_ids),
        "synced": synced,
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "errors": errors,
    }
