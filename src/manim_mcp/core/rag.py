"""ChromaDB-powered RAG service for Manim code generation."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)


class ChromaDBService:
    """RAG service using ChromaDB for similarity search.

    Provides graceful degradation - if ChromaDB is unavailable, all methods
    return empty results and the pipeline falls back to simple generation.
    """

    def __init__(self, config: ManimMCPConfig) -> None:
        self.config = config
        self.available = False
        self._client = None
        self._scenes_collection = None
        self._docs_collection = None
        self._errors_collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB connection with graceful degradation."""
        if not self.config.rag_enabled:
            logger.info("RAG disabled via configuration")
            return

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.HttpClient(
                host=self.config.chromadb_host,
                port=self.config.chromadb_port,
                settings=Settings(anonymized_telemetry=False),
            )

            # Test connection by getting heartbeat
            self._client.heartbeat()

            # Get or create collections
            self._scenes_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_scenes,
                metadata={"description": "Production Manim scene code"},
            )
            self._docs_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_docs,
                metadata={"description": "Manim API documentation and examples"},
            )
            self._errors_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_errors,
                metadata={"description": "Error patterns and fixes"},
            )

            self.available = True
            logger.info(
                "ChromaDB connected: %s:%d (scenes=%d, docs=%d, errors=%d)",
                self.config.chromadb_host,
                self.config.chromadb_port,
                self._scenes_collection.count(),
                self._docs_collection.count(),
                self._errors_collection.count(),
            )

        except ImportError:
            logger.warning(
                "chromadb not installed - RAG unavailable. "
                "Install with: pip install 'manim-mcp[rag]'"
            )
        except Exception as e:
            logger.warning("ChromaDB connection failed - RAG unavailable: %s", e)

    async def index_manim_code(
        self,
        code: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index Manim scene code for future retrieval.

        Args:
            code: The Manim Python code to index
            metadata: Optional metadata (prompt, source, etc.)

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._scenes_collection:
            return None

        try:
            doc_id = self._hash_content(code)
            meta = metadata or {}
            meta["code_length"] = len(code)

            self._scenes_collection.upsert(
                ids=[doc_id],
                documents=[code],
                metadatas=[meta],
            )
            logger.debug("Indexed scene code: %s", doc_id[:8])
            return doc_id

        except Exception as e:
            logger.warning("Failed to index scene code: %s", e)
            return None

    async def index_documentation(
        self,
        content: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index Manim documentation or examples.

        Args:
            content: Documentation text or example code
            metadata: Optional metadata (source, api_class, etc.)

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._docs_collection:
            return None

        try:
            doc_id = self._hash_content(content)
            meta = metadata or {}

            self._docs_collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
            logger.debug("Indexed documentation: %s", doc_id[:8])
            return doc_id

        except Exception as e:
            logger.warning("Failed to index documentation: %s", e)
            return None

    async def index_error_pattern(
        self,
        error: str,
        fix: str,
        original_code: str | None = None,
    ) -> str | None:
        """Index an error pattern and its fix for self-learning.

        Args:
            error: The error message or pattern
            fix: The fixed code or fix description
            original_code: Optional original code that caused the error

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._errors_collection:
            return None

        try:
            # Combine error and fix for the document
            document = f"ERROR:\n{error}\n\nFIX:\n{fix}"
            doc_id = self._hash_content(document)

            metadata = {
                "error_type": self._classify_error(error),
            }
            if original_code:
                metadata["original_code_hash"] = self._hash_content(original_code)

            self._errors_collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )
            logger.debug("Indexed error pattern: %s", doc_id[:8])
            return doc_id

        except Exception as e:
            logger.warning("Failed to index error pattern: %s", e)
            return None

    async def search_similar_scenes(
        self,
        query: str,
        n_results: int | None = None,
    ) -> list[dict]:
        """Search for similar Manim scenes.

        Args:
            query: Natural language description or code snippet
            n_results: Number of results (default from config)

        Returns:
            List of results with content, metadata, and similarity score
        """
        if not self.available or not self._scenes_collection:
            return []

        n = n_results or self.config.rag_results_limit

        try:
            results = self._scenes_collection.query(
                query_texts=[query],
                n_results=n,
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Scene search failed: %s", e)
            return []

    async def search_documentation(
        self,
        query: str,
        n_results: int | None = None,
    ) -> list[dict]:
        """Search Manim documentation and examples.

        Args:
            query: Natural language query about Manim API
            n_results: Number of results (default from config)

        Returns:
            List of results with content and metadata
        """
        if not self.available or not self._docs_collection:
            return []

        n = n_results or self.config.rag_results_limit

        try:
            results = self._docs_collection.query(
                query_texts=[query],
                n_results=n,
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Documentation search failed: %s", e)
            return []

    async def search_error_patterns(
        self,
        error: str,
        n_results: int = 3,
    ) -> list[dict]:
        """Search for similar error patterns and their fixes.

        Args:
            error: The error message to search for
            n_results: Number of results

        Returns:
            List of error/fix pairs
        """
        if not self.available or not self._errors_collection:
            return []

        try:
            results = self._errors_collection.query(
                query_texts=[error],
                n_results=n_results,
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Error pattern search failed: %s", e)
            return []

    async def get_collection_stats(self) -> dict:
        """Get statistics about all collections.

        Returns:
            Dict with collection counts and availability status
        """
        stats = {
            "available": self.available,
            "scenes_count": 0,
            "docs_count": 0,
            "errors_count": 0,
        }

        if not self.available:
            return stats

        try:
            if self._scenes_collection:
                stats["scenes_count"] = self._scenes_collection.count()
            if self._docs_collection:
                stats["docs_count"] = self._docs_collection.count()
            if self._errors_collection:
                stats["errors_count"] = self._errors_collection.count()
        except Exception as e:
            logger.warning("Failed to get collection stats: %s", e)

        return stats

    async def clear_collection(self, collection_name: str) -> bool:
        """Clear all documents from a collection.

        Args:
            collection_name: Name of collection to clear

        Returns:
            True if cleared, False on error or unavailable
        """
        if not self.available or not self._client:
            return False

        try:
            self._client.delete_collection(collection_name)
            # Recreate empty collection
            if collection_name == self.config.rag_collection_scenes:
                self._scenes_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Production Manim scene code"},
                )
            elif collection_name == self.config.rag_collection_docs:
                self._docs_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Manim API documentation and examples"},
                )
            elif collection_name == self.config.rag_collection_errors:
                self._errors_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Error patterns and fixes"},
                )
            return True
        except Exception as e:
            logger.warning("Failed to clear collection %s: %s", collection_name, e)
            return False

    def _hash_content(self, content: str) -> str:
        """Generate a deterministic hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _classify_error(self, error: str) -> str:
        """Classify error type for metadata."""
        error_lower = error.lower()
        if "syntax" in error_lower:
            return "syntax"
        if "import" in error_lower or "module" in error_lower:
            return "import"
        if "attribute" in error_lower or "no attribute" in error_lower:
            return "attribute"
        if "type" in error_lower:
            return "type"
        if "scene" in error_lower:
            return "scene"
        if "latex" in error_lower or "tex" in error_lower:
            return "latex"
        return "runtime"

    def _format_results(self, raw_results: dict) -> list[dict]:
        """Format ChromaDB results into a consistent structure."""
        results = []

        if not raw_results or not raw_results.get("ids"):
            return results

        ids = raw_results["ids"][0] if raw_results["ids"] else []
        documents = raw_results["documents"][0] if raw_results.get("documents") else []
        metadatas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
        distances = raw_results["distances"][0] if raw_results.get("distances") else []

        for i, doc_id in enumerate(ids):
            result = {
                "document_id": doc_id,
                "content": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "similarity_score": 1.0 - (distances[i] if i < len(distances) else 0.0),
            }
            results.append(result)

        return results
