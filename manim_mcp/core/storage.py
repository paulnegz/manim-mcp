"""S3/MinIO storage for rendered files."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from manim_mcp.exceptions import StorageConnectionError, StorageUploadError

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)

CONTENT_TYPES = {
    "mp4": "video/mp4",
    "gif": "image/gif",
    "webm": "video/webm",
    "png": "image/png",
    "mov": "video/quicktime",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


class S3Storage:
    def __init__(self, config: ManimMCPConfig) -> None:
        self.config = config
        self.available = False
        self._client = None

    async def initialize(self) -> None:
        try:
            self._client = await asyncio.to_thread(self._create_client)
            await asyncio.to_thread(self._ensure_bucket)
            self.available = True
            logger.info("S3 storage initialized (bucket: %s)", self.config.s3_bucket)
        except (EndpointConnectionError, ClientError, NoCredentialsError, Exception) as e:
            logger.warning("S3 storage unavailable, running in degraded mode: %s", e)
            self.available = False

    def _create_client(self):
        scheme = "https" if self.config.s3_secure else "http"
        endpoint_url = f"{scheme}://{self.config.s3_endpoint}"
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.config.s3_access_key,
            aws_secret_access_key=self.config.s3_secret_key,
        )

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.config.s3_bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.config.s3_bucket)
            logger.info("Created S3 bucket: %s", self.config.s3_bucket)

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        if not self.available:
            raise StorageConnectionError("S3 storage is not available")

        if content_type is None:
            ext = local_path.rsplit(".", 1)[-1].lower() if "." in local_path else ""
            content_type = CONTENT_TYPES.get(ext, "application/octet-stream")

        last_error: Exception | None = None
        for attempt in range(1, self.config.s3_upload_retry_attempts + 1):
            try:
                await asyncio.to_thread(
                    self._client.upload_file,
                    local_path,
                    self.config.s3_bucket,
                    object_key,
                    ExtraArgs={"ContentType": content_type},
                )
                s3_url = f"s3://{self.config.s3_bucket}/{object_key}"
                logger.info("Uploaded %s to %s", local_path, s3_url)
                return s3_url
            except (ClientError, EndpointConnectionError) as e:
                last_error = e
                if attempt < self.config.s3_upload_retry_attempts:
                    delay = 2 ** (attempt - 1)
                    logger.warning("Upload attempt %d failed, retrying in %ds: %s", attempt, delay, e)
                    await asyncio.sleep(delay)

        raise StorageUploadError(f"Upload failed after {self.config.s3_upload_retry_attempts} attempts: {last_error}")

    async def generate_presigned_url(self, object_key: str, expiry: int | None = None) -> str | None:
        if not self.available:
            return None
        try:
            url = await asyncio.to_thread(
                self._client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self.config.s3_bucket, "Key": object_key},
                ExpiresIn=expiry or self.config.s3_presigned_expiry,
            )
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL for %s: %s", object_key, e)
            return None

    async def delete_object(self, object_key: str) -> None:
        if not self.available:
            return
        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=self.config.s3_bucket,
                Key=object_key,
            )
        except ClientError as e:
            logger.error("Failed to delete %s: %s", object_key, e)

    async def object_exists(self, object_key: str) -> bool:
        if not self.available:
            return False
        try:
            await asyncio.to_thread(
                self._client.head_object,
                Bucket=self.config.s3_bucket,
                Key=object_key,
            )
            return True
        except ClientError:
            return False
