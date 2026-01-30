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
        self._api_collection = None
        self._patterns_collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB connection with graceful degradation.

        Supports two modes:
        1. HTTP client: connects to a ChromaDB server (chromadb_host:chromadb_port)
        2. Persistent client: uses local storage (chromadb_path)

        Falls back to persistent mode if HTTP fails or chromadb_path is set.
        """
        if not self.config.rag_enabled:
            logger.info("RAG disabled via configuration")
            return

        try:
            import os
            import warnings

            # Suppress pydantic v1 warnings for Python 3.14+
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
                import chromadb

            # Priority: 1) local data/chromadb, 2) explicit path, 3) HTTP server
            local_db_path = self.config.chromadb_path or os.path.join(os.getcwd(), "data", "chromadb")

            if os.path.exists(local_db_path):
                # Use local persistent DB
                try:
                    self._client = chromadb.PersistentClient(path=local_db_path)
                    logger.info("ChromaDB using local DB: %s", local_db_path)
                except Exception as local_err:
                    logger.warning("PersistentClient failed: %s, trying HTTP", local_err)
                    self._client = chromadb.HttpClient(
                        host=self.config.chromadb_host,
                        port=self.config.chromadb_port,
                    )
                    self._client.heartbeat()
            else:
                # Try HTTP client
                self._client = chromadb.HttpClient(
                    host=self.config.chromadb_host,
                    port=self.config.chromadb_port,
                )
                self._client.heartbeat()
                logger.info("ChromaDB HTTP connected: %s:%d",
                           self.config.chromadb_host, self.config.chromadb_port)

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
            self._api_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_api,
                metadata={"description": "Manimgl API signatures and parameters"},
            )
            self._patterns_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_patterns,
                metadata={"description": "3b1b animation patterns and templates"},
            )

            self.available = True
            logger.info(
                "ChromaDB ready (scenes=%d, docs=%d, errors=%d, api=%d, patterns=%d)",
                self._scenes_collection.count(),
                self._docs_collection.count(),
                self._errors_collection.count(),
                self._api_collection.count(),
                self._patterns_collection.count(),
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

    async def index_api_signature(
        self,
        signature_doc: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index a manimgl API signature for parameter validation.

        Args:
            signature_doc: Document describing the API (signature, params, docstring)
            metadata: Metadata including full_name, class_name, parameters, etc.

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._api_collection:
            return None

        try:
            meta = metadata or {}
            doc_id = meta.get("id") or self._hash_content(signature_doc)

            self._api_collection.upsert(
                ids=[doc_id],
                documents=[signature_doc],
                metadatas=[meta],
            )
            logger.debug("Indexed API signature: %s", doc_id)
            return doc_id

        except Exception as e:
            logger.warning("Failed to index API signature: %s", e)
            return None

    async def search_api_signatures(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Search for API signatures matching a query.

        Args:
            query: Method name, class name, or description
            n_results: Number of results to return

        Returns:
            List of matching API signatures with metadata
        """
        if not self.available or not self._api_collection:
            return []

        try:
            results = self._api_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("API signature search failed: %s", e)
            return []

    async def get_api_signature(
        self,
        full_name: str,
    ) -> dict | None:
        """Get a specific API signature by its full name (e.g., 'Axes.get_riemann_rectangles').

        Args:
            full_name: Fully qualified name like 'ClassName.method_name'

        Returns:
            API signature dict or None if not found
        """
        if not self.available or not self._api_collection:
            return None

        try:
            results = self._api_collection.get(
                ids=[full_name],
                include=["documents", "metadatas"],
            )

            if results and results.get("ids"):
                return {
                    "document_id": results["ids"][0],
                    "content": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {},
                }
            return None

        except Exception as e:
            logger.warning("API signature lookup failed for %s: %s", full_name, e)
            return None

    async def index_animation_pattern(
        self,
        pattern_doc: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index a 3b1b animation pattern for code generation guidance.

        Args:
            pattern_doc: Document describing the pattern (name, code template, etc.)
            metadata: Metadata including name, category, math_concepts, keywords

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._patterns_collection:
            return None

        try:
            meta = metadata or {}
            doc_id = meta.get("id") or self._hash_content(pattern_doc)

            self._patterns_collection.upsert(
                ids=[doc_id],
                documents=[pattern_doc],
                metadatas=[meta],
            )
            logger.debug("Indexed animation pattern: %s", doc_id)
            return doc_id

        except Exception as e:
            logger.warning("Failed to index animation pattern: %s", e)
            return None

    async def search_animation_patterns(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """Search for animation patterns matching a query.

        Args:
            query: Math concept, animation type, or description
            n_results: Number of results to return

        Returns:
            List of matching patterns with metadata and code templates
        """
        if not self.available or not self._patterns_collection:
            return []

        try:
            results = self._patterns_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Animation pattern search failed: %s", e)
            return []

    async def search_similar_scenes(
        self,
        query: str,
        n_results: int | None = None,
        prioritize_3b1b: bool = True,
    ) -> list[dict]:
        """Search for similar Manim scenes.

        Args:
            query: Natural language description or code snippet
            n_results: Number of results (default from config)
            prioritize_3b1b: If True, boost 3b1b original code and filter out generated code

        Returns:
            List of results with content, metadata, and similarity score
        """
        if not self.available or not self._scenes_collection:
            return []

        # Fetch more results so we can filter/re-rank
        fetch_n = (n_results or self.config.rag_results_limit) * 3 if prioritize_3b1b else (n_results or self.config.rag_results_limit)

        try:
            results = self._scenes_collection.query(
                query_texts=[query],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
            )

            formatted = self._format_results(results)

            if prioritize_3b1b and formatted:
                formatted = self._prioritize_3b1b_results(formatted, n_results or self.config.rag_results_limit)

            return formatted

        except Exception as e:
            logger.warning("Scene search failed: %s", e)
            return []

    def _prioritize_3b1b_results(self, results: list[dict], n_results: int) -> list[dict]:
        """Re-rank results to prioritize 3b1b original code over generated code.

        3b1b code is identified by:
        - Has 'manim_imports_ext' in code (3b1b custom library)
        - Has source metadata indicating 3b1b
        - Does NOT have 'source': 'generated' in metadata
        """
        # Separate into 3b1b original vs generated
        original_3b1b = []
        other_results = []

        for r in results:
            content = r.get("content", "")
            meta = r.get("metadata", {})
            source = meta.get("source", "")

            # Check if it's generated code (skip it)
            if source in ("generated", "self_indexed"):
                logger.debug("[RAG] Filtering out generated code (source=%s)", source)
                continue

            # Check if it's 3b1b original code (boost it)
            is_3b1b = (
                "manim_imports_ext" in content or
                "OldTex" in content or
                "from manim_imports_ext" in content or
                source == "3b1b" or
                "3blue1brown" in source.lower()
            )

            if is_3b1b:
                original_3b1b.append(r)
                logger.debug("[RAG] Found 3b1b original (score=%.3f)", r.get("similarity_score", 0))
            else:
                other_results.append(r)

        # Prioritize: 3b1b first, then others
        prioritized = original_3b1b + other_results
        return prioritized[:n_results]

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

    async def store_error_pattern(
        self,
        error_message: str,
        code: str,
        fix: str | None = None,
        prompt: str | None = None,
        transformed_code: str | None = None,
    ) -> bool:
        """Store an error pattern for future learning.

        Args:
            error_message: The error message that occurred
            code: The original code (what LLM generated)
            fix: The fixed code (if available)
            prompt: The original prompt (for context)
            transformed_code: The code after bridge transformation (what actually ran)

        Returns:
            True if stored successfully
        """
        if not self.available or not self._errors_collection:
            return False

        try:
            import hashlib
            # Create unique ID from error + code hash
            content_hash = hashlib.sha256(f"{error_message}{code[:500]}".encode()).hexdigest()[:16]
            doc_id = f"err_{content_hash}"

            # Parse error reason from message
            error_reason = self._parse_error_reason(error_message)

            # Format document for retrieval with both original and transformed code
            parts = [
                f"ERROR: {error_message}",
                f"\nREASON: {error_reason}",
                f"\nORIGINAL_CODE (what LLM generated):\n```python\n{code[:2000]}\n```",
            ]

            if transformed_code and transformed_code != code:
                parts.append(
                    f"\nTRANSFORMED_CODE (what actually ran after CE→manimgl bridge):\n"
                    f"```python\n{transformed_code[:2000]}\n```"
                )

            if fix:
                parts.append(f"\nFIX:\n```python\n{fix[:2000]}\n```")

            document = "\n".join(parts)

            metadata = {
                "error_type": self._classify_error(error_message),
                "error_reason": error_reason[:200],
                "has_fix": fix is not None,
                "has_transformed": transformed_code is not None,
                "code_length": len(code),
                "prompt": (prompt or "")[:200],
            }

            self._errors_collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )

            logger.info("[RAG] Stored error pattern: %s (reason=%s, has_fix=%s)",
                       error_message[:50], error_reason[:30], fix is not None)
            return True

        except Exception as e:
            logger.warning("Failed to store error pattern: %s", e)
            return False

    def _parse_error_reason(self, error_message: str) -> str:
        """Extract a human-readable reason from error message."""
        import re

        # Common patterns
        patterns = [
            (r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
             lambda m: f"{m.group(1)} doesn't have method/attribute '{m.group(2)}' in manimgl"),
            (r"TypeError: .*unexpected keyword argument '(\w+)'",
             lambda m: f"Parameter '{m.group(1)}' doesn't exist in manimgl"),
            (r"ImportError: cannot import name '(\w+)'",
             lambda m: f"'{m.group(1)}' doesn't exist in manimgl"),
            (r"NameError: name '(\w+)' is not defined",
             lambda m: f"'{m.group(1)}' is not defined - might be CE-only"),
            (r"FileNotFoundError:.*'(\w+)'",
             lambda m: f"Missing dependency: '{m.group(1)}' not installed"),
            (r"SyntaxError: (.+)",
             lambda m: f"Syntax error: {m.group(1)}"),
        ]

        for pattern, formatter in patterns:
            match = re.search(pattern, error_message)
            if match:
                return formatter(match)

        # Fallback: extract first line after "Error" or "Exception"
        lines = error_message.split('\n')
        for line in lines:
            if 'Error' in line or 'Exception' in line:
                return line.strip()[:100]

        return "Unknown error"

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
            "api_count": 0,
            "patterns_count": 0,
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
            if self._api_collection:
                stats["api_count"] = self._api_collection.count()
            if self._patterns_collection:
                stats["patterns_count"] = self._patterns_collection.count()
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
            elif collection_name == self.config.rag_collection_api:
                self._api_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Manimgl API signatures and parameters"},
                )
            elif collection_name == self.config.rag_collection_patterns:
                self._patterns_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "3b1b animation patterns and templates"},
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
