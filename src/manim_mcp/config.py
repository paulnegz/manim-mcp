"""Configuration for manim-mcp server, loaded from environment variables."""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class DefaultQuality(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    production = "production"
    fourk = "fourk"


class DefaultFormat(str, Enum):
    mp4 = "mp4"
    gif = "gif"
    webm = "webm"
    mov = "mov"
    png = "png"


class ManimMCPConfig(BaseSettings):
    model_config = {"env_prefix": "MANIM_MCP_"}

    server_name: str = "manim-mcp"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "manim-renders"
    s3_prefix: str = "renders/"
    s3_secure: bool = False
    s3_presigned_expiry: int = 3600
    s3_upload_retry_attempts: int = 3
    s3_max_upload_size_mb: int = 500

    manim_executable: str = "manim"
    default_quality: DefaultQuality = DefaultQuality.medium
    default_format: DefaultFormat = DefaultFormat.mp4
    render_timeout: int = 180
    max_concurrent_renders: int = 4

    tracker_db_path: str = "renders.db"
    max_code_length: int = 50000
    log_level: str = "INFO"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_retries: int = 3

    latex_available: bool = False
