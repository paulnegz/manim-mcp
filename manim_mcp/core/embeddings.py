"""Code-aware embeddings using UniXcoder for AST-aware code understanding.

UniXcoder is a 125M parameter model from Microsoft that understands code structure
and AST semantics, making it superior to general text embeddings for code retrieval.

Reference: https://huggingface.co/microsoft/unixcoder-base
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Environment variable to enable code embeddings
CODE_EMBEDDINGS_ENV_VAR = "MANIM_MCP_USE_CODE_EMBEDDINGS"


def is_code_embeddings_enabled() -> bool:
    """Check if code-aware embeddings are enabled via environment variable."""
    return os.environ.get(CODE_EMBEDDINGS_ENV_VAR, "").lower() in ("true", "1", "yes")


class CodeEmbeddingFunction:
    """ChromaDB-compatible embedding function using UniXcoder for code understanding.

    This class implements ChromaDB's EmbeddingFunction protocol, providing
    code-aware embeddings that understand AST structure and code semantics.

    Features:
    - Uses microsoft/unixcoder-base for code-aware embeddings
    - Implements caching for repeated embeddings
    - Supports batching for efficiency
    - Falls back gracefully if model loading fails

    Usage:
        embedding_fn = CodeEmbeddingFunction()
        embeddings = embedding_fn(["def hello(): pass", "class Foo: ..."])
    """

    # Model configuration
    MODEL_NAME = "microsoft/unixcoder-base"
    MAX_LENGTH = 512  # UniXcoder's max token length
    EMBEDDING_DIM = 768  # UniXcoder output dimension

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_size: int = 1000,
    ) -> None:
        """Initialize the CodeEmbeddingFunction.

        Args:
            model_name: HuggingFace model name (default: microsoft/unixcoder-base)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_size: Number of embeddings to cache (default: 1000)
        """
        self._model_name = model_name or self.MODEL_NAME
        self._device = device
        self._cache_size = cache_size

        # Lazy initialization
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._initialization_error: str | None = None

        # Embedding cache (content hash -> embedding)
        self._cache: dict[str, list[float]] = {}

    def _initialize(self) -> bool:
        """Lazily initialize the model and tokenizer.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return self._initialization_error is None

        self._initialized = True

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Auto-detect device
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "Loading UniXcoder model: %s (device=%s)",
                self._model_name, self._device
            )

            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()

            logger.info("UniXcoder model loaded successfully")
            return True

        except ImportError as e:
            self._initialization_error = (
                f"transformers or torch not installed: {e}. "
                "Install with: pip install transformers torch"
            )
            logger.warning("CodeEmbeddingFunction init failed: %s", self._initialization_error)
            return False

        except Exception as e:
            self._initialization_error = f"Failed to load UniXcoder model: {e}"
            logger.warning("CodeEmbeddingFunction init failed: %s", self._initialization_error)
            return False

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for a single text/code snippet.

        Args:
            text: The code or text to embed

        Returns:
            List of floats representing the embedding vector
        """
        import torch

        # Tokenize with truncation
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get embeddings (no gradient computation needed)
        with torch.no_grad():
            outputs = self._model(**inputs)

            # UniXcoder uses the [CLS] token embedding (first token)
            # Shape: (batch_size, sequence_length, hidden_size)
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize the embedding
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list and return
        return embeddings[0].cpu().tolist()

    def _compute_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of texts.

        Args:
            texts: List of code/text snippets to embed

        Returns:
            List of embedding vectors
        """
        import torch

        # Tokenize batch with truncation and padding
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)

            # Get [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists
        return embeddings.cpu().tolist()

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text snippet."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_from_cache(self, text: str) -> list[float] | None:
        """Get embedding from cache if available."""
        key = self._get_cache_key(text)
        return self._cache.get(key)

    def _add_to_cache(self, text: str, embedding: list[float]) -> None:
        """Add embedding to cache, evicting oldest if necessary."""
        if len(self._cache) >= self._cache_size:
            # Simple eviction: remove first item (oldest in insertion order)
            # Python 3.7+ dicts maintain insertion order
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        key = self._get_cache_key(text)
        self._cache[key] = embedding

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of documents (ChromaDB EmbeddingFunction protocol).

        Args:
            input: List of documents/code to embed

        Returns:
            List of embedding vectors (list of floats)

        Note:
            If initialization fails, returns zero vectors to allow graceful degradation.
        """
        # Initialize on first call
        if not self._initialize():
            logger.warning(
                "UniXcoder not available, returning zero vectors. Error: %s",
                self._initialization_error
            )
            # Return zero vectors for graceful degradation
            return [[0.0] * self.EMBEDDING_DIM for _ in input]

        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each input
        for i, text in enumerate(input):
            cached = self._get_from_cache(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append([])  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute embeddings for uncached texts in batch
        if uncached_texts:
            try:
                # Process in batches of 32 for memory efficiency
                batch_size = 32
                computed_embeddings: list[list[float]] = []

                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_texts))
                    batch = uncached_texts[batch_start:batch_end]
                    batch_embeddings = self._compute_embeddings_batch(batch)
                    computed_embeddings.extend(batch_embeddings)

                # Fill in results and update cache
                for i, idx in enumerate(uncached_indices):
                    embedding = computed_embeddings[i]
                    results[idx] = embedding
                    self._add_to_cache(uncached_texts[i], embedding)

            except Exception as e:
                logger.error("Failed to compute embeddings: %s", e)
                # Return zero vectors for failed computations
                for idx in uncached_indices:
                    results[idx] = [0.0] * self.EMBEDDING_DIM

        return results

    def embed_code(self, code: str) -> list[float]:
        """Convenience method to embed a single code snippet.

        Args:
            code: Python code to embed

        Returns:
            Embedding vector
        """
        return self([code])[0]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (same as embed_code for UniXcoder).

        Args:
            query: Search query

        Returns:
            Embedding vector
        """
        return self.embed_code(query)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug("Embedding cache cleared")

    @property
    def is_available(self) -> bool:
        """Check if the embedding function is available and working."""
        return self._initialize()


# Singleton instance for reuse
_code_embedding_instance: CodeEmbeddingFunction | None = None


def get_code_embedding_function() -> CodeEmbeddingFunction:
    """Get a shared instance of the CodeEmbeddingFunction.

    Returns:
        Shared CodeEmbeddingFunction instance
    """
    global _code_embedding_instance
    if _code_embedding_instance is None:
        _code_embedding_instance = CodeEmbeddingFunction()
    return _code_embedding_instance


def get_embedding_function_for_collection(collection_name: str) -> Any:
    """Get the appropriate embedding function for a collection.

    Uses CodeEmbeddingFunction for code collections (scenes, api) when enabled,
    and returns None (ChromaDB default) for text collections (docs, errors, patterns).

    Args:
        collection_name: Name of the ChromaDB collection

    Returns:
        EmbeddingFunction instance or None for default
    """
    # Code collections that benefit from code-aware embeddings
    code_collections = {"manim_scenes", "manim_api"}

    # Check if code embeddings are enabled
    if not is_code_embeddings_enabled():
        logger.debug(
            "Code embeddings disabled (set %s=true to enable)",
            CODE_EMBEDDINGS_ENV_VAR
        )
        return None

    # Use code embeddings for code collections
    if collection_name in code_collections:
        logger.info(
            "Using UniXcoder code embeddings for collection: %s",
            collection_name
        )
        return get_code_embedding_function()

    # Use default embeddings for text collections
    logger.debug(
        "Using default embeddings for text collection: %s",
        collection_name
    )
    return None
