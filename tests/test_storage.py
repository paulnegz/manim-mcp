"""Tests for S3 storage using moto mocks."""

from __future__ import annotations

import os
import tempfile

import boto3
import pytest
from moto import mock_aws

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.storage import S3Storage
from manim_mcp.exceptions import StorageConnectionError


@pytest.fixture
def s3_config() -> ManimMCPConfig:
    return ManimMCPConfig(
        s3_endpoint="localhost:9000",
        s3_access_key="testing",
        s3_secret_key="testing",
        s3_bucket="test-bucket",
        s3_prefix="renders/",
    )


@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(b"fake video content")
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _make_mocked_storage(s3_config: ManimMCPConfig) -> S3Storage:
    storage = S3Storage(s3_config)
    storage._client = boto3.client("s3", region_name="us-east-1")
    storage._client.create_bucket(Bucket=s3_config.s3_bucket)
    storage.available = True
    return storage


class TestS3Storage:
    async def test_initialize_creates_bucket(self, s3_config: ManimMCPConfig):
        with mock_aws():
            storage = S3Storage(s3_config)
            storage._client = boto3.client("s3", region_name="us-east-1")
            storage._ensure_bucket()
            storage.available = True

            assert storage.available is True

    async def test_upload_and_exists(self, s3_config: ManimMCPConfig, temp_file: str):
        with mock_aws():
            storage = _make_mocked_storage(s3_config)

            s3_url = await storage.upload_file(temp_file, "renders/test/video.mp4")
            assert s3_url.startswith("s3://")

            exists = await storage.object_exists("renders/test/video.mp4")
            assert exists is True

    async def test_delete_object(self, s3_config: ManimMCPConfig, temp_file: str):
        with mock_aws():
            storage = _make_mocked_storage(s3_config)

            await storage.upload_file(temp_file, "renders/test/video.mp4")
            await storage.delete_object("renders/test/video.mp4")

            exists = await storage.object_exists("renders/test/video.mp4")
            assert exists is False

    async def test_presigned_url(self, s3_config: ManimMCPConfig, temp_file: str):
        with mock_aws():
            storage = _make_mocked_storage(s3_config)

            await storage.upload_file(temp_file, "renders/test/video.mp4")
            url = await storage.generate_presigned_url("renders/test/video.mp4")
            assert url is not None
            assert "renders/test/video.mp4" in url

    async def test_upload_when_unavailable(self, s3_config: ManimMCPConfig, temp_file: str):
        storage = S3Storage(s3_config)
        storage.available = False

        with pytest.raises(StorageConnectionError):
            await storage.upload_file(temp_file, "test/file.mp4")

    async def test_presigned_url_when_unavailable(self, s3_config: ManimMCPConfig):
        storage = S3Storage(s3_config)
        storage.available = False

        result = await storage.generate_presigned_url("test/file.mp4")
        assert result is None

    async def test_object_exists_when_unavailable(self, s3_config: ManimMCPConfig):
        storage = S3Storage(s3_config)
        storage.available = False

        result = await storage.object_exists("test/file.mp4")
        assert result is False
