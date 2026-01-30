"""Exception hierarchy for manim-mcp."""


class ManimMCPError(Exception):
    """Base exception for all manim-mcp errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class CodeValidationError(ManimMCPError):
    """Syntax errors or invalid code."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message, details={"errors": errors or []})
        self.errors = errors or []


class DangerousCodeError(CodeValidationError):
    """Blocked imports or operations detected."""

    def __init__(self, message: str, blocked_items: list[str] | None = None):
        super().__init__(message, blocked_items)
        self.blocked_items = blocked_items or []


class RenderError(ManimMCPError):
    """Manim execution failure."""

    def __init__(
        self,
        message: str,
        exit_code: int | None = None,
        stderr: str | None = None,
    ):
        details = {}
        if exit_code is not None:
            details["exit_code"] = exit_code
        if stderr:
            details["stderr"] = stderr[:2000]  # Truncate long error output
        super().__init__(message, details)
        self.exit_code = exit_code
        self.stderr = stderr


class RenderTimeoutError(RenderError):
    """Render exceeded timeout."""

    def __init__(self, message: str, timeout_seconds: int):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.details["timeout_seconds"] = timeout_seconds


class OutputNotFoundError(RenderError):
    """Render succeeded but no output file found."""


class StorageError(ManimMCPError):
    """S3/MinIO storage failure."""


class StorageUploadError(StorageError):
    """Upload failed after retries."""

    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message, details={"attempts": attempts})
        self.attempts = attempts


class StorageConnectionError(StorageError):
    """S3 endpoint unreachable."""


class RenderNotFoundError(ManimMCPError):
    """Unknown render ID."""

    def __init__(self, render_id: str):
        super().__init__(f"Render '{render_id}' not found", details={"render_id": render_id})
        self.render_id = render_id


class ConcurrencyLimitError(ManimMCPError):
    """Too many concurrent renders."""

    def __init__(self, message: str, current: int, limit: int):
        super().__init__(message, details={"current": current, "limit": limit})
        self.current = current
        self.limit = limit


class ManimNotInstalledError(ManimMCPError):
    """Manim binary not found on PATH."""


class FileSizeLimitError(ManimMCPError):
    """Output file exceeds maximum upload size."""

    def __init__(self, message: str, size_bytes: int, limit_bytes: int):
        super().__init__(
            message,
            details={"size_bytes": size_bytes, "limit_bytes": limit_bytes},
        )
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


# ── LLM Errors ────────────────────────────────────────────────────────

class LLMError(ManimMCPError):
    """Base class for LLM-related errors."""


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, details={"provider": provider})
        self.provider = provider


class LLMRateLimitError(LLMError):
    """LLM provider rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """LLM returned invalid or empty response."""


class LLMMaxRetriesError(LLMError):
    """LLM call failed after maximum retries."""

    def __init__(self, message: str, attempts: int, last_error: str | None = None):
        super().__init__(
            message,
            details={"attempts": attempts, "last_error": last_error},
        )
        self.attempts = attempts
        self.last_error = last_error


# ── RAG Errors ────────────────────────────────────────────────────────

class RAGError(ManimMCPError):
    """ChromaDB or RAG system failure."""


class RAGConnectionError(RAGError):
    """ChromaDB endpoint unreachable."""


class RAGIndexError(RAGError):
    """Failed to index content into ChromaDB."""


class RAGSearchError(RAGError):
    """Failed to search RAG database."""


# ── Agent Errors ──────────────────────────────────────────────────────

class AgentError(ManimMCPError):
    """Multi-agent pipeline failure."""


class AgentTimeoutError(AgentError):
    """Agent exceeded maximum processing time."""


class AgentPipelineError(AgentError):
    """Pipeline failed to complete all stages."""

    def __init__(self, message: str, failed_stage: str):
        super().__init__(message, details={"failed_stage": failed_stage})
        self.failed_stage = failed_stage
