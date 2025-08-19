from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile
from src.services.s3_service import (
    get_s3_bucket_contents,
    create_presigned_url,
    upload_fileobj_to_s3,
    head_object_from_s3,
    object_exists_in_s3,
    delete_file_from_s3,
)
from src.settings import settings
from .schemas import Document, DeleteResult


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


async def list_documents(include_url: bool) -> list[Document]:
    contents = get_s3_bucket_contents(settings.AWS_S3_RAG_DOCUMENTS_BUCKET)
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


async def upload_document(
    file: UploadFile,
    key: Optional[str],
    overwrite: bool,
    include_url: bool,
) -> Document:
    resolved_key = key or file.filename
    if not resolved_key:
        raise HTTPException(status_code=400, detail="A key or filename is required")

    if not overwrite and object_exists_in_s3(
        settings.AWS_S3_RAG_DOCUMENTS_BUCKET, resolved_key
    ):
        raise HTTPException(
            status_code=409, detail="Object already exists; use overwrite=true"
        )

    await file.seek(0)
    upload_fileobj_to_s3(
        settings.AWS_S3_RAG_DOCUMENTS_BUCKET,
        resolved_key,
        file.file,
        content_type=file.content_type,
    )
    return document_from_head(resolved_key, include_url=include_url)


async def update_document(
    key: str,
    file: UploadFile,
    create_if_missing: bool,
    include_url: bool,
) -> Tuple[Document, bool]:
    exists = object_exists_in_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
    if not exists and not create_if_missing:
        raise HTTPException(status_code=404, detail="Object not found")

    await file.seek(0)
    upload_fileobj_to_s3(
        settings.AWS_S3_RAG_DOCUMENTS_BUCKET,
        key,
        file.file,
        content_type=file.content_type,
    )
    document = document_from_head(key, include_url=include_url)
    created = not exists
    return document, created


def delete_document(key: str) -> DeleteResult:
    if not object_exists_in_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key):
        raise HTTPException(status_code=404, detail="Object not found")

    deleted = delete_file_from_s3(settings.AWS_S3_RAG_DOCUMENTS_BUCKET, key)
    return DeleteResult(key=key, deleted=bool(deleted))
