"""MCP tools for RAG search and index management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from manim_mcp.bootstrap import AppContext


def register_rag_tools(mcp: FastMCP) -> None:
    """Register RAG-related MCP tools."""

    @mcp.tool()
    async def search_similar_scenes(
        query: str,
        limit: int = 5,
    ) -> dict:
        """Search for similar Manim scenes in the RAG database.

        Use this to find examples of animations similar to what you want to create.
        Results include code snippets and metadata from indexed scenes.

        Args:
            query: Natural language description of the animation you're looking for
            limit: Maximum number of results to return (1-10, default 5)

        Returns:
            Dictionary with 'results' list containing similar scenes with
            document_id, content (code), metadata, and similarity_score.
            Returns empty list if RAG is unavailable.
        """
        ctx: AppContext = mcp.state["ctx"]

        if not ctx.rag.available:
            return {
                "available": False,
                "results": [],
                "message": "RAG service is not available. Start ChromaDB to enable.",
            }

        limit = max(1, min(limit, 10))
        results = await ctx.rag.search_similar_scenes(query, n_results=limit)

        return {
            "available": True,
            "results": results,
            "count": len(results),
        }

    @mcp.tool()
    async def search_manim_docs(
        query: str,
        limit: int = 5,
    ) -> dict:
        """Search Manim documentation and API examples.

        Use this to find documentation about Manim classes, methods, or examples.

        Args:
            query: What you're looking for (e.g., "how to create axes",
                   "VGroup usage", "animation timing")
            limit: Maximum number of results to return (1-10, default 5)

        Returns:
            Dictionary with 'results' list containing documentation snippets.
            Returns empty list if RAG is unavailable.
        """
        ctx: AppContext = mcp.state["ctx"]

        if not ctx.rag.available:
            return {
                "available": False,
                "results": [],
                "message": "RAG service is not available. Start ChromaDB to enable.",
            }

        limit = max(1, min(limit, 10))
        results = await ctx.rag.search_documentation(query, n_results=limit)

        return {
            "available": True,
            "results": results,
            "count": len(results),
        }

    @mcp.tool()
    async def search_error_fixes(
        error_message: str,
        limit: int = 3,
    ) -> dict:
        """Search for fixes to common Manim errors.

        Use this when you encounter an error to find similar errors and their fixes.

        Args:
            error_message: The error message you're trying to fix
            limit: Maximum number of results to return (1-5, default 3)

        Returns:
            Dictionary with 'results' list containing error patterns and fixes.
            Returns empty list if RAG is unavailable.
        """
        ctx: AppContext = mcp.state["ctx"]

        if not ctx.rag.available:
            return {
                "available": False,
                "results": [],
                "message": "RAG service is not available. Start ChromaDB to enable.",
            }

        limit = max(1, min(limit, 5))
        results = await ctx.rag.search_error_patterns(error_message, n_results=limit)

        return {
            "available": True,
            "results": results,
            "count": len(results),
        }

    @mcp.tool()
    async def rag_index_status() -> dict:
        """Get the status of the RAG index.

        Returns statistics about indexed content including:
        - Whether RAG is available
        - Number of scenes indexed
        - Number of documentation entries
        - Number of error patterns

        Returns:
            Dictionary with availability status and collection counts.
        """
        ctx: AppContext = mcp.state["ctx"]

        stats = await ctx.rag.get_collection_stats()

        return {
            "available": stats["available"],
            "collections": {
                "scenes": stats["scenes_count"],
                "docs": stats["docs_count"],
                "errors": stats["errors_count"],
            },
            "total_documents": (
                stats["scenes_count"] +
                stats["docs_count"] +
                stats["errors_count"]
            ),
        }
