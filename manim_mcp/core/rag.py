"""ChromaDB-powered RAG service for Manim code generation."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from manim_mcp.core.embeddings import get_embedding_function_for_collection
from manim_mcp.core.probe_search import get_probe_searcher, ProbeSearcher

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
        self._intro_outro_collection = None
        self._characters_collection = None
        self._legacy_collection = None
        self._probe_searcher: ProbeSearcher | None = None

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

            # Get or create collections with appropriate embedding functions
            # Code collections (scenes, api) use UniXcoder when enabled
            # Text collections (docs, errors, patterns) use default embeddings
            scenes_embedding_fn = get_embedding_function_for_collection(
                self.config.rag_collection_scenes
            )
            api_embedding_fn = get_embedding_function_for_collection(
                self.config.rag_collection_api
            )

            self._scenes_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_scenes,
                metadata={"description": "Production Manim scene code"},
                embedding_function=scenes_embedding_fn,
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
                embedding_function=api_embedding_fn,
            )
            self._patterns_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_patterns,
                metadata={"description": "3b1b animation patterns and templates"},
            )
            self._intro_outro_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_intro_outro,
                metadata={"description": "3b1b video intro/outro patterns (quotes, end screens, banners)"},
            )
            self._characters_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_characters,
                metadata={"description": "Pi creature character patterns and animations"},
            )
            self._legacy_collection = self._client.get_or_create_collection(
                name=self.config.rag_collection_legacy,
                metadata={"description": "Legacy/archived 3b1b constructs for reference"},
            )

            self.available = True
            logger.info(
                "ChromaDB ready (scenes=%d, docs=%d, errors=%d, api=%d, patterns=%d, intro_outro=%d, characters=%d, legacy=%d)",
                self._scenes_collection.count(),
                self._docs_collection.count(),
                self._errors_collection.count(),
                self._api_collection.count(),
                self._patterns_collection.count(),
                self._intro_outro_collection.count(),
                self._characters_collection.count(),
                self._legacy_collection.count(),
            )

            # Initialize Probe searcher for AST-aware code search
            self._probe_searcher = get_probe_searcher()
            if self._probe_searcher.available:
                logger.info("Probe searcher available for AST-aware code search")
            else:
                logger.debug("Probe searcher not available (probe CLI not installed or no paths configured)")

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
        exact_match_boost: float = 2.0,
    ) -> list[dict]:
        """Search for API signatures using hybrid search (keyword + semantic).

        Combines exact keyword matching on method names with semantic similarity
        for better retrieval of API methods like 'FadeIn', 'Transform', etc.

        Args:
            query: Method name, class name, or description
            n_results: Number of results to return
            exact_match_boost: Multiplier for exact match scores (default 2.0)

        Returns:
            List of matching API signatures with metadata, ranked by hybrid score
        """
        if not self.available or not self._api_collection:
            return []

        try:
            return await self._search_api_hybrid(
                query=query,
                n_results=n_results,
                exact_match_boost=exact_match_boost,
            )
        except Exception as e:
            logger.warning("API signature search failed: %s", e)
            return []

    async def _search_api_hybrid(
        self,
        query: str,
        n_results: int = 5,
        exact_match_boost: float = 2.0,
    ) -> list[dict]:
        """Perform hybrid search combining exact keyword matching with semantic search.

        Strategy:
        1. Extract potential method/class names from query
        2. Search for exact matches on method_name and class_name metadata
        3. Perform semantic search for broader matches
        4. Merge results with boosted scores for exact matches
        5. Deduplicate and return top n_results

        Args:
            query: Search query (method name, class name, or description)
            n_results: Number of results to return
            exact_match_boost: Score multiplier for exact matches

        Returns:
            List of API signatures ranked by hybrid score
        """
        import re

        results_map: dict[str, dict] = {}  # doc_id -> result with score

        # Extract potential identifiers from query (CamelCase, snake_case, dotted)
        # Examples: "FadeIn", "move_to", "Axes.get_graph", "Transform animation"
        identifiers = re.findall(r'[A-Z][a-zA-Z0-9]*|[a-z_][a-z0-9_]*', query)
        # Also try the full query for dotted names like "Axes.get_graph"
        if '.' in query:
            identifiers.append(query.split()[0] if ' ' in query else query)

        # Phase 1: Exact matches on metadata fields
        for identifier in identifiers:
            if len(identifier) < 2:
                continue

            # Try exact match on method_name
            try:
                exact_results = self._api_collection.get(
                    where={"method_name": identifier},
                    include=["documents", "metadatas"],
                )
                if exact_results and exact_results.get("ids"):
                    for i, doc_id in enumerate(exact_results["ids"]):
                        if doc_id not in results_map:
                            results_map[doc_id] = {
                                "document_id": doc_id,
                                "content": exact_results["documents"][i] if exact_results.get("documents") else "",
                                "metadata": exact_results["metadatas"][i] if exact_results.get("metadatas") else {},
                                "similarity_score": 1.0 * exact_match_boost,  # Boosted exact match
                                "match_type": "exact_method",
                            }
                            logger.debug("[RAG] Exact method match: %s (boosted score=%.2f)",
                                       doc_id, 1.0 * exact_match_boost)
            except Exception as e:
                logger.debug("Exact method_name search failed for '%s': %s", identifier, e)

            # Try exact match on class_name
            try:
                class_results = self._api_collection.get(
                    where={"class_name": identifier},
                    include=["documents", "metadatas"],
                )
                if class_results and class_results.get("ids"):
                    for i, doc_id in enumerate(class_results["ids"]):
                        if doc_id not in results_map:
                            results_map[doc_id] = {
                                "document_id": doc_id,
                                "content": class_results["documents"][i] if class_results.get("documents") else "",
                                "metadata": class_results["metadatas"][i] if class_results.get("metadatas") else {},
                                "similarity_score": 0.9 * exact_match_boost,  # Slightly lower than method match
                                "match_type": "exact_class",
                            }
                            logger.debug("[RAG] Exact class match: %s (boosted score=%.2f)",
                                       doc_id, 0.9 * exact_match_boost)
            except Exception as e:
                logger.debug("Exact class_name search failed for '%s': %s", identifier, e)

        # Phase 2: Semantic search for broader matches
        semantic_results = self._api_collection.query(
            query_texts=[query],
            n_results=n_results * 2,  # Fetch more to allow for deduplication
            include=["documents", "metadatas", "distances"],
        )

        if semantic_results and semantic_results.get("ids"):
            ids = semantic_results["ids"][0] if semantic_results["ids"] else []
            documents = semantic_results["documents"][0] if semantic_results.get("documents") else []
            metadatas = semantic_results["metadatas"][0] if semantic_results.get("metadatas") else []
            distances = semantic_results["distances"][0] if semantic_results.get("distances") else []

            for i, doc_id in enumerate(ids):
                semantic_score = 1.0 - (distances[i] if i < len(distances) else 0.0)

                if doc_id in results_map:
                    # Already have exact match - keep the higher boosted score
                    # but note that semantic also matched (good signal)
                    existing = results_map[doc_id]
                    existing["match_type"] = f"{existing.get('match_type', 'unknown')}+semantic"
                    logger.debug("[RAG] Semantic also matched exact: %s (keeping boosted score=%.2f)",
                               doc_id, existing["similarity_score"])
                else:
                    # New result from semantic search only
                    results_map[doc_id] = {
                        "document_id": doc_id,
                        "content": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "similarity_score": semantic_score,
                        "match_type": "semantic",
                    }

        # Phase 3: Sort by score and return top n_results
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        # Clean up internal match_type field before returning (optional: keep for debugging)
        final_results = []
        for r in sorted_results[:n_results]:
            result = {
                "document_id": r["document_id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "similarity_score": min(r["similarity_score"], 1.0),  # Cap at 1.0 for display
            }
            # Preserve match_type in metadata for debugging/analytics
            if "match_type" in r:
                result["metadata"] = {**result["metadata"], "_match_type": r["match_type"]}
            final_results.append(result)

        logger.debug("[RAG] Hybrid API search for '%s': %d exact + %d semantic -> %d results",
                    query,
                    sum(1 for r in results_map.values() if "exact" in r.get("match_type", "")),
                    sum(1 for r in results_map.values() if r.get("match_type") == "semantic"),
                    len(final_results))

        return final_results

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

    async def index_intro_outro_pattern(
        self,
        pattern_doc: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index a video intro/outro pattern (opening quote, end screen, banner).

        Args:
            pattern_doc: Document describing the pattern (code, description)
            metadata: Metadata including type (intro/outro/banner), source_file

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._intro_outro_collection:
            return None

        try:
            meta = metadata or {}
            doc_id = meta.get("id") or self._hash_content(pattern_doc)

            self._intro_outro_collection.upsert(
                ids=[doc_id],
                documents=[pattern_doc],
                metadatas=[meta],
            )
            logger.debug("Indexed intro/outro pattern: %s", doc_id)
            return doc_id

        except Exception as e:
            logger.warning("Failed to index intro/outro pattern: %s", e)
            return None

    async def index_character_pattern(
        self,
        pattern_doc: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index a pi creature character pattern.

        Args:
            pattern_doc: Document describing the character pattern
            metadata: Metadata including character_type, animation_type

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._characters_collection:
            return None

        try:
            meta = metadata or {}
            doc_id = meta.get("id") or self._hash_content(pattern_doc)

            self._characters_collection.upsert(
                ids=[doc_id],
                documents=[pattern_doc],
                metadatas=[meta],
            )
            logger.debug("Indexed character pattern: %s", doc_id)
            return doc_id

        except Exception as e:
            logger.warning("Failed to index character pattern: %s", e)
            return None

    async def index_legacy_pattern(
        self,
        pattern_doc: str,
        metadata: dict | None = None,
    ) -> str | None:
        """Index a legacy/archived construct pattern.

        Args:
            pattern_doc: Document describing the legacy pattern
            metadata: Metadata including original_module, deprecated_reason

        Returns:
            Document ID if indexed, None if RAG unavailable
        """
        if not self.available or not self._legacy_collection:
            return None

        try:
            meta = metadata or {}
            doc_id = meta.get("id") or self._hash_content(pattern_doc)

            self._legacy_collection.upsert(
                ids=[doc_id],
                documents=[pattern_doc],
                metadatas=[meta],
            )
            logger.debug("Indexed legacy pattern: %s", doc_id)
            return doc_id

        except Exception as e:
            logger.warning("Failed to index legacy pattern: %s", e)
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

    async def search_intro_outro_patterns(
        self,
        query: str,
        n_results: int = 3,
        pattern_type: str | None = None,
    ) -> list[dict]:
        """Search for intro/outro patterns matching a query.

        Args:
            query: Description of desired intro/outro effect
            n_results: Number of results to return
            pattern_type: Filter by type (intro, outro, banner)

        Returns:
            List of matching patterns with metadata and code
        """
        if not self.available or not self._intro_outro_collection:
            return []

        try:
            where_filter = {"pattern_type": pattern_type} if pattern_type else None
            results = self._intro_outro_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Intro/outro pattern search failed: %s", e)
            return []

    async def search_character_patterns(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """Search for pi creature character patterns.

        Args:
            query: Description of desired character animation
            n_results: Number of results to return

        Returns:
            List of matching character patterns
        """
        if not self.available or not self._characters_collection:
            return []

        try:
            results = self._characters_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Character pattern search failed: %s", e)
            return []

    async def search_legacy_patterns(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[dict]:
        """Search for legacy/archived construct patterns.

        Args:
            query: Description of desired construct
            n_results: Number of results to return

        Returns:
            List of matching legacy patterns
        """
        if not self.available or not self._legacy_collection:
            return []

        try:
            results = self._legacy_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            return self._format_results(results)

        except Exception as e:
            logger.warning("Legacy pattern search failed: %s", e)
            return []

    async def search_similar_scenes(
        self,
        query: str,
        n_results: int | None = None,
        prioritize_3b1b: bool = True,
    ) -> list[dict]:
        """Search for similar Manim scenes using hybrid ChromaDB + Probe search.

        Args:
            query: Natural language description or code snippet
            n_results: Number of results (default from config)
            prioritize_3b1b: If True, boost 3b1b original code and filter out generated code

        Returns:
            List of results with content, metadata, and similarity score
        """
        import asyncio

        target_n = n_results or self.config.rag_results_limit
        chromadb_results = []
        probe_results = []

        # Run ChromaDB and Probe searches in parallel
        async def chromadb_search():
            if not self.available or not self._scenes_collection:
                return []
            fetch_n = target_n * 3 if prioritize_3b1b else target_n
            try:
                results = self._scenes_collection.query(
                    query_texts=[query],
                    n_results=fetch_n,
                    include=["documents", "metadatas", "distances"],
                )
                return self._format_results(results)
            except Exception as e:
                logger.warning("ChromaDB scene search failed: %s", e)
                return []

        async def probe_search():
            if not self._probe_searcher or not self._probe_searcher.available:
                return []
            try:
                result = await self._probe_searcher.search_scenes(
                    query=query,
                    max_tokens=4000,
                    max_results=target_n,
                )
                return result.to_rag_format()
            except Exception as e:
                logger.warning("Probe scene search failed: %s", e)
                return []

        # Execute both searches concurrently
        chromadb_results, probe_results = await asyncio.gather(
            chromadb_search(),
            probe_search(),
        )

        # Merge results: Probe results are AST-aware complete code blocks
        merged = self._merge_search_results(
            chromadb_results,
            probe_results,
            probe_boost=0.15,  # Boost Probe results for complete code blocks
        )

        if prioritize_3b1b and merged:
            merged = self._prioritize_3b1b_results(
                merged,
                target_n,
                query=query,
            )

        return merged[:target_n]

    def _merge_search_results(
        self,
        chromadb_results: list[dict],
        probe_results: list[dict],
        probe_boost: float = 0.15,
    ) -> list[dict]:
        """Merge ChromaDB and Probe search results with deduplication.

        Strategy:
        - Use content hash for deduplication
        - Probe results get a score boost (complete AST-aware blocks)
        - Items found in both get an additional overlap boost
        - Sort by final score

        Args:
            chromadb_results: Results from ChromaDB vector search
            probe_results: Results from Probe AST-aware search
            probe_boost: Score boost for Probe results (0.0-1.0)

        Returns:
            Merged and deduplicated results sorted by score
        """
        results_map: dict[str, dict] = {}  # content_hash -> result

        # Add ChromaDB results
        for r in chromadb_results:
            content = r.get("content", "")
            content_hash = self._hash_content(content[:500])  # Hash prefix for dedup
            if content_hash not in results_map:
                results_map[content_hash] = {
                    **r,
                    "source": "chromadb",
                    "_final_score": r.get("similarity_score", 0.5),
                }

        # Add/merge Probe results
        for r in probe_results:
            content = r.get("content", "")
            content_hash = self._hash_content(content[:500])

            if content_hash in results_map:
                # Found in both - boost the existing result
                existing = results_map[content_hash]
                existing["_final_score"] = min(1.0, existing["_final_score"] + 0.1)
                existing["source"] = "chromadb+probe"
                logger.debug("[RAG] Overlap boost for result found in both sources")
            else:
                # New from Probe - add with boost
                probe_score = r.get("similarity_score", 0.5) + probe_boost
                results_map[content_hash] = {
                    **r,
                    "source": "probe",
                    "_final_score": min(1.0, probe_score),
                }

        # Sort by final score and clean up
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x.get("_final_score", 0),
            reverse=True,
        )

        # Remove internal fields before returning
        for r in sorted_results:
            r["similarity_score"] = r.pop("_final_score", r.get("similarity_score", 0))
            r.pop("source", None)

        if probe_results:
            logger.info(
                "[RAG] Merged %d ChromaDB + %d Probe results -> %d unique",
                len(chromadb_results), len(probe_results), len(sorted_results)
            )

        return sorted_results

    def _prioritize_3b1b_results(self, results: list[dict], n_results: int, query: str = "") -> list[dict]:
        """Re-rank results to prioritize 3b1b original code and method-containing results.

        Two-stage re-ranking:
        1. Filter out generated code, boost 3b1b original
        2. Boost results containing actual method calls related to query

        3b1b code is identified by:
        - Has 'manim_imports_ext' in code (3b1b custom library)
        - Has source metadata indicating 3b1b
        - Does NOT have 'source': 'generated' in metadata
        """
        import re

        # Stage 1: Filter generated, identify 3b1b
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

        # Stage 2: Boost results containing relevant method calls
        # Extract method patterns from query (scalable - no hardcoded domains)
        method_patterns = self._extract_method_patterns(query)

        if method_patterns:
            # Score each result by method presence
            def method_score(r):
                content = r.get("content", "")
                score = 0
                for pattern in method_patterns:
                    if pattern in content:
                        score += 1
                        # Extra boost if it's a method CALL (has parentheses after)
                        if re.search(rf"{re.escape(pattern)}\s*\(", content):
                            score += 2
                return score

            # Sort 3b1b results by method score (higher first)
            original_3b1b.sort(key=method_score, reverse=True)
            other_results.sort(key=method_score, reverse=True)

            # Log what we boosted
            if original_3b1b and method_patterns:
                top_score = method_score(original_3b1b[0])
                if top_score > 0:
                    logger.info("[RAG] Boosted result with %d method matches for patterns: %s",
                               top_score, method_patterns[:3])

        # Prioritize: 3b1b first, then others
        prioritized = original_3b1b + other_results
        return prioritized[:n_results]

    def _extract_method_patterns(self, query: str) -> list[str]:
        """Extract likely manimgl method names from query.

        Scalable approach - maps common words to method patterns:
        - "riemann" -> get_riemann_rectangles
        - "graph" -> get_graph
        - "area" -> get_area_under_graph
        - etc.

        Returns method name patterns to search for in results.
        """
        if not query:
            return []

        query_lower = query.lower()
        patterns = []

        # Map concept words to manimgl method names
        # This is more scalable than hardcoding in query building
        concept_to_methods = {
            "riemann": ["get_riemann_rectangles"],
            "rectangle": ["get_riemann_rectangles", "Rectangle"],
            "graph": ["get_graph", "get_area_under_graph"],
            "area": ["get_area_under_graph", "get_riemann_rectangles"],
            "integral": ["get_area_under_graph", "get_riemann_rectangles"],
            "derivative": ["get_derivative_graph", "get_secant_slope_group"],
            "tangent": ["get_tangent_line", "TangentLine"],
            "secant": ["get_secant_slope_group"],
            "axes": ["Axes", "NumberPlane", "get_graph"],
            "transform": ["Transform", "ReplacementTransform", "TransformMatchingShapes"],
            "fade": ["FadeIn", "FadeOut", "FadeTransform"],
            "write": ["Write", "ShowCreation"],
            "animate": [".animate."],
            "vector": ["Vector", "Arrow"],
            "matrix": ["Matrix", "IntegerMatrix"],
            "circle": ["Circle", "Arc", "Annulus"],
            "line": ["Line", "DashedLine", "NumberLine"],
            "text": ["Text", "Tex", "TexText"],
            "label": ["get_graph_label", "DecimalNumber"],
            "dot": ["Dot", "SmallDot"],
            "arrow": ["Arrow", "Vector", "DoubleArrow"],
            "brace": ["Brace", "BraceLabel"],
            "updater": ["add_updater", "always_redraw"],
            "3d": ["ThreeDAxes", "Surface", "ParametricSurface"],
            "surface": ["Surface", "ParametricSurface"],
            "camera": ["frame", "set_camera_orientation"],
        }

        for concept, methods in concept_to_methods.items():
            if concept in query_lower:
                patterns.extend(methods)

        # Deduplicate while preserving order
        seen = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        return unique_patterns

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
            from datetime import datetime

            # Create unique ID from error + code hash
            content_hash = hashlib.sha256(f"{error_message}{code[:500]}".encode()).hexdigest()[:16]
            doc_id = f"err_{content_hash}"

            # Parse error reason from message
            error_reason = self._parse_error_reason(error_message)

            # Normalize error for better semantic matching
            normalized_error = self._normalize_error_pattern(error_message)

            # Format document for retrieval with both original and transformed code
            # Include normalized version for better similarity search
            parts = [
                f"ERROR: {error_message}",
                f"\nNORMALIZED: {normalized_error}",
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

            # Check if this pattern already exists (to preserve success tracking)
            existing = None
            try:
                existing_result = self._errors_collection.get(ids=[doc_id], include=["metadatas"])
                if existing_result and existing_result.get("metadatas"):
                    existing = existing_result["metadatas"][0]
            except Exception:
                pass  # Pattern doesn't exist yet

            # Build metadata with success tracking
            metadata = {
                "error_type": self._classify_error(error_message),
                "error_reason": error_reason[:200],
                "normalized_error": normalized_error[:300],
                "has_fix": fix is not None,
                "has_transformed": transformed_code is not None,
                "code_length": len(code),
                "prompt": (prompt or "")[:200],
                # Success tracking - preserve existing or initialize
                "success_count": existing.get("success_count", 1 if fix else 0) if existing else (1 if fix else 0),
                "attempt_count": existing.get("attempt_count", 1) if existing else 1,
                "created_at": existing.get("created_at", datetime.now().isoformat()) if existing else datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # Calculate success rate
            if metadata["attempt_count"] > 0:
                metadata["success_rate"] = metadata["success_count"] / metadata["attempt_count"]
            else:
                metadata["success_rate"] = 0.0

            self._errors_collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )

            logger.info("[RAG] Stored error pattern: %s (reason=%s, has_fix=%s, success_rate=%.2f)",
                       error_message[:50], error_reason[:30], fix is not None, metadata["success_rate"])
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
            (r"TypeError: Object .* cannot be converted to an animation",
             lambda m: "Method used in self.play() is not animatable - use .animate.method()"),
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

    def _normalize_error_pattern(self, error_message: str) -> str:
        """Normalize error message for better semantic matching.

        Generalizes specific details (class names, line numbers, variable names)
        into placeholders so similar errors can match even with different specifics.

        Example:
            "'MyClass' object has no attribute 'foo'"
            becomes "'<CLASS>' object has no attribute '<ATTR>'"

        This improves retrieval of similar error patterns.
        """
        import re

        normalized = error_message

        # Replace specific class names with placeholder
        normalized = re.sub(r"'(\w+)' object", "'<CLASS>' object", normalized)

        # Replace specific attribute/method names in AttributeError
        normalized = re.sub(
            r"has no attribute '(\w+)'",
            "has no attribute '<ATTR>'",
            normalized
        )

        # Replace specific parameter names in TypeError
        normalized = re.sub(
            r"unexpected keyword argument '(\w+)'",
            "unexpected keyword argument '<PARAM>'",
            normalized
        )

        # Replace line numbers
        normalized = re.sub(r"line \d+", "line <N>", normalized)
        normalized = re.sub(r"Line \d+", "Line <N>", normalized)

        # Replace column numbers
        normalized = re.sub(r"column \d+", "column <N>", normalized)
        normalized = re.sub(r"col \d+", "col <N>", normalized)

        # Replace file paths (keep just filename pattern)
        normalized = re.sub(r'/tmp/[^/]+/', '/tmp/<TMP>/', normalized)

        # Replace hex addresses
        normalized = re.sub(r'0x[0-9a-fA-F]+', '0x<ADDR>', normalized)

        # Replace specific variable/function names in NameError
        normalized = re.sub(
            r"name '(\w+)' is not defined",
            "name '<NAME>' is not defined",
            normalized
        )

        return normalized

    async def update_error_pattern_success(
        self,
        error_message: str,
        code: str,
        success: bool = True,
    ) -> bool:
        """Update success tracking for an error pattern.

        Called when a fix from RAG was applied and we know if it worked.
        This helps prioritize patterns with higher success rates in retrieval.

        Args:
            error_message: The original error message
            code: The code that had the error
            success: Whether the fix worked

        Returns:
            True if updated successfully
        """
        if not self.available or not self._errors_collection:
            return False

        try:
            import hashlib
            from datetime import datetime

            # Find the pattern by ID
            content_hash = hashlib.sha256(f"{error_message}{code[:500]}".encode()).hexdigest()[:16]
            doc_id = f"err_{content_hash}"

            # Get existing document
            result = self._errors_collection.get(ids=[doc_id], include=["metadatas"])

            if not result or not result.get("metadatas"):
                return False

            metadata = result["metadatas"][0] if result["metadatas"] else {}

            # Update success tracking
            success_count = metadata.get("success_count", 0)
            attempt_count = metadata.get("attempt_count", 0)

            if success:
                success_count += 1
            attempt_count += 1

            # Update metadata
            metadata["success_count"] = success_count
            metadata["attempt_count"] = attempt_count
            metadata["success_rate"] = success_count / attempt_count if attempt_count > 0 else 0
            metadata["last_used"] = datetime.now().isoformat()

            self._errors_collection.update(
                ids=[doc_id],
                metadatas=[metadata],
            )

            logger.debug(
                "[RAG] Updated error pattern success: %s (success_rate=%.2f, attempts=%d)",
                doc_id[:8], metadata["success_rate"], attempt_count
            )
            return True

        except Exception as e:
            logger.warning("Failed to update error pattern success: %s", e)
            return False

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
            "intro_outro_count": 0,
            "characters_count": 0,
            "legacy_count": 0,
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
            if self._intro_outro_collection:
                stats["intro_outro_count"] = self._intro_outro_collection.count()
            if self._characters_collection:
                stats["characters_count"] = self._characters_collection.count()
            if self._legacy_collection:
                stats["legacy_count"] = self._legacy_collection.count()
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
            # Recreate empty collection with appropriate embedding function
            embedding_fn = get_embedding_function_for_collection(collection_name)

            if collection_name == self.config.rag_collection_scenes:
                self._scenes_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Production Manim scene code"},
                    embedding_function=embedding_fn,
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
                    embedding_function=embedding_fn,
                )
            elif collection_name == self.config.rag_collection_patterns:
                self._patterns_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "3b1b animation patterns and templates"},
                )
            elif collection_name == self.config.rag_collection_intro_outro:
                self._intro_outro_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "3b1b video intro/outro patterns"},
                )
            elif collection_name == self.config.rag_collection_characters:
                self._characters_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Pi creature character patterns"},
                )
            elif collection_name == self.config.rag_collection_legacy:
                self._legacy_collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Legacy/archived constructs"},
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
