from fastapi import (
    APIRouter,
    Query,
    UploadFile,
    File,
    status,
    Path,
    Form,
    Response,
)
from typing import Optional
from src.app.features.documents.schemas import Document, DeleteResult, SyncStatus
from .service import (
    list_documents as svc_list_documents,
    upload_document as svc_upload_document,
    update_document as svc_update_document,
    delete_document as svc_delete_document,
    document_from_head as svc_document_from_head,
    get_document_sync_status as svc_get_sync_status,
    list_sync_statuses as svc_list_sync_statuses,
)

router = APIRouter()


def _document_from_head(key: str, include_url: bool = False) -> Document:
    return svc_document_from_head(key, include_url=include_url)


@router.get(
    "/documents",
    response_model=list[Document],
    response_model_exclude_none=True,
    summary="List documents from S3",
    tags=["documents"],
)
async def get_documents(
    include_url: bool = Query(
        False, description="If true, include a presigned download URL for each object"
    ),
):
    """Returns S3 object metadata from the configured bucket.

    Note: This endpoint only reads from S3. Pinecone is not queried here.
    Optionally includes a presigned URL for download.
    """
    return await svc_list_documents(include_url=include_url)


@router.get(
    "/documents/{key}/sync",
    response_model=SyncStatus,
    summary="Get sync status between S3 and Pinecone for a single document",
    tags=["documents"],
)
async def get_document_sync(
    key: str = Path(..., description="S3 key of the object (also Pinecone doc_id)"),
):
    """Returns whether the S3 object has corresponding vectors in Pinecone that match the current ETag.

    Status values:
    - in_sync: vectors exist for this doc_id and match the current etag
    - stale: vectors exist for doc_id, but none match the current etag
    - not_indexed: no vectors exist for doc_id
    """
    status, count_doc, count_both, etag = svc_get_sync_status(key)
    return SyncStatus(
        key=key,
        etag=etag,
        status=status,
        vectors_for_doc_id=count_doc,
        vectors_for_doc_id_and_etag=count_both,
    )


@router.get(
    "/documents/sync",
    response_model=list[SyncStatus],
    summary="List sync status for all S3 documents",
    tags=["documents"],
)
async def list_documents_sync():
    """Returns sync information for every object in the S3 bucket.

    This performs a Pinecone stats check per object, so call volume scales with number of items.
    """
    statuses = []
    for key, etag, count_doc, count_both in svc_list_sync_statuses():
        from src.services.vectorstores.pinecone_service import get_pinecone_service

        status = get_pinecone_service().get_sync_status(key, etag)
        statuses.append(
            SyncStatus(
                key=key,
                etag=etag,
                status=status,
                vectors_for_doc_id=count_doc,
                vectors_for_doc_id_and_etag=count_both,
            )
        )
    return statuses


@router.post(
    "/documents",
    response_model=Document,
    status_code=status.HTTP_201_CREATED,
    response_model_exclude_none=True,
    summary="Upload a new document to S3",
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="File to upload"),
    key: Optional[str] = Form(
        None,
        description="Destination S3 key (path/filename). Defaults to the uploaded filename. This key is also used as the Pinecone doc_id.",
    ),
    overwrite: bool = Form(
        False,
        description="If true, overwrite an existing object at the same key. When overwriting, existing Pinecone vectors for this key are replaced.",
    ),
    include_url: bool = Form(
        False,
        description="If true, include a presigned URL in the response.",
    ),
):
    """Uploads a file to S3 and synchronizes the Pinecone vectorstore.

    - If no key is provided, the original filename is used.
    - On success, the file contents are split and upserted into Pinecone under `doc_id = key`.
    - If `overwrite=true`, any existing vectors for the same key are removed before re-ingesting.
    - Non-text files may be ignored or partially ingested (best-effort UTF-8 decode).
    """
    return await svc_upload_document(
        file=file,
        key=key,
        overwrite=overwrite,
        include_url=include_url,
    )


@router.put(
    "/documents/{key}",
    response_model=Document,
    response_model_exclude_none=True,
    summary="Update or replace an existing document in S3",
    tags=["documents"],
)
async def update_document(
    response: Response,
    key: str = Path(
        ..., description="S3 key of the object to update (also used as Pinecone doc_id)"
    ),
    file: UploadFile = File(..., description="Replacement file contents"),
    create_if_missing: bool = Query(
        False, description="If true, create the object when it does not exist"
    ),
    include_url: bool = Query(
        False, description="If true, include a presigned URL in the response"
    ),
):
    """Overwrites the object stored at the provided key. Returns updated metadata.

    Pinecone synchronization: existing vectors for this key are deleted and replaced with
    vectors derived from the new file contents.

    Set create_if_missing=true to create the object if it does not exist.
    """
    document, created = await svc_update_document(
        key=key,
        file=file,
        create_if_missing=create_if_missing,
        include_url=include_url,
    )
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return document


@router.delete(
    "/documents/{key}",
    response_model=DeleteResult,
    summary="Delete a document from S3",
    tags=["documents"],
)
async def delete_document(
    key: str = Path(
        ..., description="S3 key of the object to delete (also used as Pinecone doc_id)"
    ),
):
    """Removes the object at the specified key.

    Also attempts to delete all Pinecone vectors associated with this key (best-effort).
    S3 deletion status is reported in the response; Pinecone cleanup is not included in
    this status and does not block the S3 deletion.
    """
    return svc_delete_document(key)
