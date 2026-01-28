"""Exception hierarchy for manim-mcp."""


class ManimMCPError(Exception):
    """Base exception for all manim-mcp errors."""


class CodeValidationError(ManimMCPError):
    """Syntax errors or invalid code."""


class DangerousCodeError(CodeValidationError):
    """Blocked imports or operations detected."""


class RenderError(ManimMCPError):
    """Manim execution failure."""


class RenderTimeoutError(RenderError):
    """Render exceeded timeout."""


class OutputNotFoundError(RenderError):
    """Render succeeded but no output file found."""


class StorageError(ManimMCPError):
    """S3/MinIO storage failure."""


class StorageUploadError(StorageError):
    """Upload failed after retries."""


class StorageConnectionError(StorageError):
    """S3 endpoint unreachable."""


class RenderNotFoundError(ManimMCPError):
    """Unknown render ID."""


class ConcurrencyLimitError(ManimMCPError):
    """Too many concurrent renders."""


class ManimNotInstalledError(ManimMCPError):
    """Manim binary not found on PATH."""


class FileSizeLimitError(ManimMCPError):
    """Output file exceeds maximum upload size."""
