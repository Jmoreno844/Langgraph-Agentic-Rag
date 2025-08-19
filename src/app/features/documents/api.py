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
from src.app.features.documents.schemas import Document, DeleteResult
from .service import (
    list_documents as svc_list_documents,
    upload_document as svc_upload_document,
    update_document as svc_update_document,
    delete_document as svc_delete_document,
    document_from_head as svc_document_from_head,
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

    Optionally includes a presigned URL for download.
    """
    return await svc_list_documents(include_url=include_url)


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
        description="Destination S3 key (path/filename). Defaults to the uploaded filename.",
    ),
    overwrite: bool = Form(
        False,
        description="If true, overwrite an existing object at the same key.",
    ),
    include_url: bool = Form(
        False,
        description="If true, include a presigned URL in the response.",
    ),
):
    """Uploads a file to S3.

    If no key is provided, the original filename is used. By default, this endpoint
    fails if the object already exists; set overwrite=true to replace.
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
    key: str = Path(..., description="S3 key of the object to update"),
    file: UploadFile = File(..., description="Replacement file contents"),
    create_if_missing: bool = Query(
        False, description="If true, create the object when it does not exist"
    ),
    include_url: bool = Query(
        False, description="If true, include a presigned URL in the response"
    ),
):
    """Overwrites the object stored at the provided key. Returns updated metadata.

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
    key: str = Path(..., description="S3 key of the object to delete"),
):
    """Removes the object at the specified key."""
    return svc_delete_document(key)
