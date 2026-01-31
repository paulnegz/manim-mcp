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

    manim_executable: str = "manimgl"
    default_quality: DefaultQuality = DefaultQuality.medium
    default_format: DefaultFormat = DefaultFormat.mp4
    render_timeout: int = 360
    max_concurrent_renders: int = 4

    tracker_db_path: str = "renders.db"
    max_code_length: int = 50000
    log_level: str = "INFO"

    # LLM Provider settings
    llm_provider: str = "gemini"  # gemini or claude
    llm_max_retries: int = 3
    llm_timeout: int = 60  # seconds

    # Gemini settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"

    # Claude settings
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    latex_available: bool = False

    # Legacy alias for backwards compatibility
    @property
    def gemini_max_retries(self) -> int:
        return self.llm_max_retries

    # ChromaDB / RAG settings
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_path: str | None = None  # If set, use local persistent DB instead of HTTP
    rag_enabled: bool = True
    rag_results_limit: int = 5
    rag_collection_scenes: str = "manim_scenes"
    rag_collection_docs: str = "manim_docs"
    rag_collection_errors: str = "error_patterns"
    rag_collection_api: str = "manim_api"
    rag_collection_patterns: str = "animation_patterns"

    # Enhanced RAG settings (Anthropic Contextual Retrieval approach)
    rag_use_enhanced: bool = True  # Use enhanced RAG with hybrid search
    rag_hybrid_search: bool = True  # Enable BM25 + Dense hybrid search
    rag_use_reranking: bool = True  # Enable Cohere reranking (requires COHERE_API_KEY)
    rag_contextual_embeddings: bool = True  # Generate contextual prefixes

    # Voyage AI settings (code-specific embeddings)
    voyage_api_key: str = ""  # VOYAGE_API_KEY for voyage-code-3
    voyage_model: str = "voyage-code-3"
    voyage_dimension: int = 1024  # Matryoshka: 256, 512, 1024, 2048

    # Cohere settings (reranking)
    cohere_api_key: str = ""  # COHERE_API_KEY for reranking
    cohere_rerank_model: str = "rerank-english-v3.0"

    # Agent mode: simple (direct LLM) or advanced (multi-agent pipeline)
    agent_mode: str = "simple"

    # Audio / TTS settings
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"  # Clear, professional voice for education
    tts_pause_ms: int = 1500  # Pause between sentences (1.5 seconds)
    tts_max_concurrent: int = 5  # Max parallel TTS requests
