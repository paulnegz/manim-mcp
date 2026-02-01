"""Probe integration for AST-aware semantic code search.

Probe (https://github.com/probelabs/probe) provides:
- Tree-sitter based code parsing (understands Python AST)
- Hybrid BM25 + TF-IDF ranking for better keyword matching
- Token-limited output for LLM context windows
- Complete code block extraction (no truncation)

This module integrates Probe with manim-mcp's RAG pipeline to improve
code search quality for 3Blue1Brown scene examples and API lookups.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """A single result from Probe search."""

    file_path: str
    line_start: int
    line_end: int
    code: str
    score: float = 0.0
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for RAG context building."""
        return {
            "content": self.code,
            "metadata": {
                "file": self.file_path,
                "line_start": self.line_start,
                "line_end": self.line_end,
                "source": "probe",
            },
            "similarity_score": self.score,
        }


@dataclass
class ProbeSearchResult:
    """Complete result from a Probe search."""

    query: str
    results: list[ProbeResult] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False

    def to_rag_format(self) -> list[dict[str, Any]]:
        """Convert to RAG-compatible format for use in prompts."""
        return [r.to_dict() for r in self.results]


class ProbeSearcher:
    """Semantic code search using Probe.

    Probe combines ripgrep (fast file scanning) with tree-sitter (AST parsing)
    and NLP ranking (BM25/TF-IDF) for high-quality code search.

    Usage:
        searcher = ProbeSearcher(["/path/to/3b1b-videos"])
        results = await searcher.search("Riemann sum animation", max_tokens=2000)
    """

    def __init__(
        self,
        search_paths: list[str] | None = None,
        default_max_tokens: int = 2000,
        default_max_results: int = 5,
        default_reranker: str = "hybrid",
    ) -> None:
        """Initialize Probe searcher.

        Args:
            search_paths: Directories to search. Defaults to MANIM_MCP_PROBE_PATHS env var.
            default_max_tokens: Default token limit for results.
            default_max_results: Default max number of results.
            default_reranker: Ranking algorithm (hybrid, bm25, tfidf).
        """
        self.search_paths = search_paths or self._get_default_paths()
        self.default_max_tokens = default_max_tokens
        self.default_max_results = default_max_results
        self.default_reranker = default_reranker
        self._probe_available: bool | None = None

    def _get_default_paths(self) -> list[str]:
        """Get search paths from environment."""
        paths_str = os.environ.get("MANIM_MCP_PROBE_PATHS", "")
        if paths_str:
            return [p.strip() for p in paths_str.split(":") if p.strip()]
        return []

    @property
    def available(self) -> bool:
        """Check if Probe is available and paths are configured."""
        if self._probe_available is None:
            self._probe_available = (
                shutil.which("probe") is not None
                and len(self.search_paths) > 0
            )
            if self._probe_available:
                logger.info("[PROBE] Available with paths: %s", self.search_paths)
            else:
                if not shutil.which("probe"):
                    logger.debug("[PROBE] Not available: probe CLI not found")
                else:
                    logger.debug("[PROBE] Not available: no search paths configured")
        return self._probe_available

    async def search(
        self,
        query: str,
        max_tokens: int | None = None,
        max_results: int | None = None,
        reranker: str | None = None,
        file_extensions: list[str] | None = None,
        include_tests: bool = False,
    ) -> ProbeSearchResult:
        """Search codebase with Probe.

        Args:
            query: Search query (supports AND, OR, NOT operators).
            max_tokens: Max tokens in output (for LLM context limits).
            max_results: Max number of results to return.
            reranker: Ranking algorithm (hybrid, bm25, tfidf).
            file_extensions: Filter by extensions (e.g., ["py"]).
            include_tests: Include test files in results.

        Returns:
            ProbeSearchResult with matching code blocks.
        """
        if not self.available:
            logger.debug("[PROBE] Search skipped: not available")
            return ProbeSearchResult(query=query)

        max_tokens = max_tokens or self.default_max_tokens
        max_results = max_results or self.default_max_results
        reranker = reranker or self.default_reranker

        # Build query with file filters
        full_query = query
        if file_extensions:
            ext_filter = " OR ".join(f"ext:{ext}" for ext in file_extensions)
            full_query = f"({query}) AND ({ext_filter})"

        # Build command
        cmd = [
            "probe", "search", full_query,
            *self.search_paths,
            "--max-tokens", str(max_tokens),
            "--max-results", str(max_results),
            "--reranker", reranker,
        ]

        if include_tests:
            cmd.append("--allow-tests")

        logger.debug("[PROBE] Running: %s", " ".join(cmd))

        try:
            # Run Probe asynchronously
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0  # 30 second timeout
            )

            if proc.returncode != 0:
                logger.warning("[PROBE] Search failed: %s", stderr.decode()[:500])
                return ProbeSearchResult(query=query)

            # Parse output
            return self._parse_output(query, stdout.decode())

        except asyncio.TimeoutError:
            logger.warning("[PROBE] Search timed out after 30s")
            return ProbeSearchResult(query=query)
        except Exception as e:
            logger.warning("[PROBE] Search error: %s", e)
            return ProbeSearchResult(query=query)

    async def search_scenes(
        self,
        query: str,
        max_tokens: int = 3000,
        max_results: int = 3,
    ) -> ProbeSearchResult:
        """Search for Manim scene implementations.

        Optimized for finding Scene class definitions with construct() methods.

        Args:
            query: Description of the animation to find.
            max_tokens: Max tokens in output.
            max_results: Max number of results.

        Returns:
            ProbeSearchResult with matching scene code.
        """
        # Enhance query to target Scene classes
        enhanced_query = f"({query}) AND (class AND Scene AND def construct)"
        return await self.search(
            query=enhanced_query,
            max_tokens=max_tokens,
            max_results=max_results,
            file_extensions=["py"],
            reranker="hybrid",
        )

    async def search_api(
        self,
        method_name: str,
        class_name: str | None = None,
        max_tokens: int = 1000,
    ) -> ProbeSearchResult:
        """Search for API method definitions.

        Finds exact method definitions with signatures and docstrings.

        Args:
            method_name: Name of the method to find.
            class_name: Optional class to scope the search.
            max_tokens: Max tokens in output.

        Returns:
            ProbeSearchResult with method definitions.
        """
        if class_name:
            query = f"def {method_name} AND class:{class_name}"
        else:
            query = f"def {method_name}"

        return await self.search(
            query=query,
            max_tokens=max_tokens,
            max_results=3,
            file_extensions=["py"],
            reranker="bm25",  # Exact matching for API lookups
        )

    async def search_patterns(
        self,
        pattern_type: str,
        max_tokens: int = 2000,
        max_results: int = 3,
    ) -> ProbeSearchResult:
        """Search for animation pattern implementations.

        Args:
            pattern_type: Type of pattern (e.g., "Riemann", "Transform", "3D").
            max_tokens: Max tokens in output.
            max_results: Max number of results.

        Returns:
            ProbeSearchResult with matching patterns.
        """
        # Common animation patterns to search for
        pattern_queries = {
            "riemann": "get_riemann_rectangles OR Riemann AND self.play",
            "transform": "Transform AND self.play AND animate",
            "3d": "ThreeDScene OR frame.reorient OR Surface",
            "graph": "Axes AND get_graph AND self.play",
            "matrix": "Matrix AND Transform AND self.play",
            "geometry": "Polygon OR Triangle OR Circle AND self.play",
            "text": "Tex OR TexText OR Write AND self.play",
            "arrow": "Arrow OR Vector AND GrowArrow OR ShowCreation",
        }

        query = pattern_queries.get(pattern_type.lower(), pattern_type)

        return await self.search(
            query=query,
            max_tokens=max_tokens,
            max_results=max_results,
            file_extensions=["py"],
            reranker="hybrid",
        )

    def _parse_output(self, query: str, output: str) -> ProbeSearchResult:
        """Parse Probe output into structured results.

        Probe outputs code blocks with file/line annotations.
        Format varies but typically includes file paths and line numbers.
        """
        results = []

        # Try to parse as structured output first
        # Probe may output JSON with --json flag (if supported)
        try:
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    results.append(ProbeResult(
                        file_path=item.get("file", ""),
                        line_start=item.get("line_start", 0),
                        line_end=item.get("line_end", 0),
                        code=item.get("code", item.get("content", "")),
                        score=item.get("score", 0.0),
                    ))
                return ProbeSearchResult(query=query, results=results)
        except json.JSONDecodeError:
            pass

        # Parse text output
        # Probe typically outputs:
        # === /path/to/file.py:42-100 ===
        # <code content>
        # === /path/to/file2.py:10-30 ===
        # <code content>

        current_file = ""
        current_lines = (0, 0)
        current_code_lines = []

        for line in output.split("\n"):
            # Check for file header
            if line.startswith("===") or line.startswith("──"):
                # Save previous result if any
                if current_code_lines and current_file:
                    results.append(ProbeResult(
                        file_path=current_file,
                        line_start=current_lines[0],
                        line_end=current_lines[1],
                        code="\n".join(current_code_lines),
                    ))
                    current_code_lines = []

                # Parse file path and lines
                # Format: === /path/file.py:42-100 === or similar
                import re
                match = re.search(r"([^\s:]+\.py):(\d+)(?:-(\d+))?", line)
                if match:
                    current_file = match.group(1)
                    start = int(match.group(2))
                    end = int(match.group(3)) if match.group(3) else start
                    current_lines = (start, end)
            elif current_file:
                # Accumulate code lines
                current_code_lines.append(line)

        # Don't forget the last result
        if current_code_lines and current_file:
            results.append(ProbeResult(
                file_path=current_file,
                line_start=current_lines[0],
                line_end=current_lines[1],
                code="\n".join(current_code_lines),
            ))

        # If no structured results, treat entire output as single result
        if not results and output.strip():
            results.append(ProbeResult(
                file_path="",
                line_start=0,
                line_end=0,
                code=output.strip(),
            ))

        return ProbeSearchResult(
            query=query,
            results=results,
            total_tokens=len(output.split()),  # Rough estimate
        )

    async def extract(
        self,
        file_path: str,
        line: int,
    ) -> ProbeResult | None:
        """Extract a complete code block at a specific location.

        Uses Probe's extract command to get the full function/class
        containing the specified line.

        Args:
            file_path: Path to the file.
            line: Line number to extract around.

        Returns:
            ProbeResult with the extracted code block, or None if failed.
        """
        if not self.available:
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                "probe", "extract", f"{file_path}:{line}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=10.0
            )

            if proc.returncode != 0:
                return None

            code = stdout.decode().strip()
            if code:
                return ProbeResult(
                    file_path=file_path,
                    line_start=line,
                    line_end=line + code.count("\n"),
                    code=code,
                )
            return None

        except Exception as e:
            logger.debug("[PROBE] Extract failed: %s", e)
            return None


# Singleton instance for convenience
_default_searcher: ProbeSearcher | None = None


def get_probe_searcher() -> ProbeSearcher:
    """Get the default Probe searcher instance."""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = ProbeSearcher()
    return _default_searcher
