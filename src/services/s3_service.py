import boto3
from src.settings import settings
from typing import Optional
from botocore.exceptions import ClientError

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)


def get_s3_client():
    return s3_client


def get_s3_bucket_contents(bucket_name: str):
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    return response.get("Contents", [])


def create_presigned_url(bucket_name: str, key: str, expires_in: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )


def upload_file_to_s3(bucket_name: str, key: str, file_path: str) -> str:
    s3_client.upload_file(file_path, bucket_name, key)
    return True


def delete_file_from_s3(bucket_name: str, key: str) -> bool:
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        return True
    except Exception as e:
        print(f"Error deleting file from S3: {e}")
        return False


# --- New helpers for API-friendly operations ---


def upload_fileobj_to_s3(
    bucket_name: str,
    key: str,
    file_obj,
    *,
    content_type: Optional[str] = None,
) -> None:
    """Upload a file-like object to S3 using managed multipart uploads.

    The provided file object must be positioned at the beginning (seeked to 0).
    """
    extra_args = {"ContentType": content_type} if content_type else None
    if extra_args:
        s3_client.upload_fileobj(file_obj, bucket_name, key, ExtraArgs=extra_args)
    else:
        s3_client.upload_fileobj(file_obj, bucket_name, key)


def head_object_from_s3(bucket_name: str, key: str) -> dict:
    """Return object metadata via HEAD request. Raises if not found."""
    return s3_client.head_object(Bucket=bucket_name, Key=key)


def object_exists_in_s3(bucket_name: str, key: str) -> bool:
    """Check whether an object exists without downloading it."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as exc:
        # 404 means not found; other errors bubble up to caller
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 or (
            exc.response.get("Error", {}).get("Code")
            in {"404", "NotFound", "NoSuchKey"}
        ):
            return False
        raise
