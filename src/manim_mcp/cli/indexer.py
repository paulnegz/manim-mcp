"""CLI commands for indexing content into ChromaDB."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.bootstrap import AppContext
    from manim_mcp.cli.output import Printer

logger = logging.getLogger(__name__)

# Repository URLs
REPO_3B1B_VIDEOS = "https://github.com/3b1b/videos.git"
REPO_MANIM_DOCS = "https://github.com/ManimCommunity/manim.git"


async def cmd_index(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Route to the appropriate index subcommand."""
    from manim_mcp.cli.output import spinner

    if args.index_command == "status":
        return await cmd_index_status(args, ctx, printer)
    elif args.index_command == "3b1b-videos":
        return await cmd_index_3b1b(args, ctx, printer)
    elif args.index_command == "manim-docs":
        return await cmd_index_docs(args, ctx, printer)
    elif args.index_command == "custom":
        return await cmd_index_custom(args, ctx, printer)
    elif args.index_command == "clear":
        return await cmd_index_clear(args, ctx, printer)
    else:
        printer.error(f"Unknown index command: {args.index_command}")
        return 1


async def cmd_index_status(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Show RAG index status and statistics."""
    from manim_mcp.cli.output import spinner

    with spinner("Checking index status"):
        stats = await ctx.rag.get_collection_stats()

    if not stats["available"]:
        printer.warn("ChromaDB is not available")
        printer.info("Start ChromaDB with: docker compose up chromadb")
        printer.info("Or install chromadb: pip install 'manim-mcp[rag]'")
        return 1

    printer.success("RAG Status: Available")
    printer.info(f"  Scenes indexed: {stats['scenes_count']}")
    printer.info(f"  Docs indexed: {stats['docs_count']}")
    printer.info(f"  Error patterns: {stats['errors_count']}")

    return 0


async def cmd_index_3b1b(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Clone and index 3b1b/videos repository."""
    import asyncio
    from manim_mcp.cli.output import spinner

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    # Determine source path
    if hasattr(args, "path") and args.path:
        repo_path = Path(args.path)
        if not repo_path.exists():
            printer.error(f"Path does not exist: {repo_path}")
            return 1
        printer.info(f"Using local path: {repo_path}")
    else:
        # Clone to temp directory
        printer.info("Cloning 3b1b/videos repository (this may take a while)...")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "videos"
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", REPO_3B1B_VIDEOS, str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                printer.error(f"Failed to clone repository: {stderr.decode()}")
                return 1

            return await _index_python_files(repo_path, ctx, printer, source="3b1b-videos")

    return await _index_python_files(repo_path, ctx, printer, source="3b1b-videos")


async def cmd_index_docs(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index Manim documentation and examples."""
    import asyncio
    from manim_mcp.cli.output import spinner

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    printer.info("Cloning Manim repository for documentation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "manim"
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", REPO_MANIM_DOCS, str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            printer.error(f"Failed to clone repository: {stderr.decode()}")
            return 1

        # Index example files
        examples_path = repo_path / "docs" / "source" / "examples"
        if examples_path.exists():
            count = await _index_docs_directory(examples_path, ctx, printer)
            printer.success(f"Indexed {count} documentation examples")
        else:
            # Fallback to example_scenes
            examples_path = repo_path / "example_scenes"
            if examples_path.exists():
                count = await _index_docs_directory(examples_path, ctx, printer)
                printer.success(f"Indexed {count} example scenes")
            else:
                printer.warn("No example directories found in repository")

    return 0


async def cmd_index_custom(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index a custom directory of Manim scenes."""
    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    directory = Path(args.directory)
    if not directory.exists():
        printer.error(f"Directory does not exist: {directory}")
        return 1

    return await _index_python_files(directory, ctx, printer, source="custom")


async def cmd_index_clear(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Clear a RAG collection."""
    from manim_mcp.cli.output import spinner

    if not ctx.rag.available:
        printer.error("ChromaDB not available")
        return 1

    collection = args.collection
    valid_collections = ["scenes", "docs", "errors", "all"]
    if collection not in valid_collections:
        printer.error(f"Invalid collection. Choose from: {', '.join(valid_collections)}")
        return 1

    if not args.yes:
        try:
            answer = input(f"Clear {collection} collection? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            printer.info("Cancelled.")
            return 0
        if answer not in ("y", "yes"):
            printer.info("Cancelled.")
            return 0

    collections_to_clear = []
    if collection == "all":
        collections_to_clear = [
            ctx.config.rag_collection_scenes,
            ctx.config.rag_collection_docs,
            ctx.config.rag_collection_errors,
        ]
    elif collection == "scenes":
        collections_to_clear = [ctx.config.rag_collection_scenes]
    elif collection == "docs":
        collections_to_clear = [ctx.config.rag_collection_docs]
    elif collection == "errors":
        collections_to_clear = [ctx.config.rag_collection_errors]

    with spinner(f"Clearing {collection} collection"):
        for coll_name in collections_to_clear:
            success = await ctx.rag.clear_collection(coll_name)
            if not success:
                printer.warn(f"Failed to clear {coll_name}")

    printer.success(f"Cleared {collection} collection")
    return 0


async def _index_python_files(
    directory: Path,
    ctx: AppContext,
    printer: Printer,
    source: str,
) -> int:
    """Index all Python files containing Manim scenes."""
    from manim_mcp.cli.output import spinner

    indexed = 0
    skipped = 0

    # Find all .py files
    py_files = list(directory.rglob("*.py"))
    printer.info(f"Found {len(py_files)} Python files")

    with spinner(f"Indexing {len(py_files)} files"):
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")

                # Only index files that look like Manim scenes
                if not _is_manim_scene(content):
                    skipped += 1
                    continue

                # Index the file
                doc_id = await ctx.rag.index_manim_code(
                    content,
                    metadata={
                        "source": source,
                        "filename": py_file.name,
                        "path": str(py_file.relative_to(directory)),
                    },
                )

                if doc_id:
                    indexed += 1

            except Exception as e:
                logger.debug("Failed to index %s: %s", py_file, e)
                skipped += 1

    printer.success(f"Indexed {indexed} Manim scenes (skipped {skipped} non-scene files)")
    return 0


async def _index_docs_directory(
    directory: Path,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index documentation files (rst, md, py)."""
    indexed = 0

    for ext in ["*.py", "*.rst", "*.md"]:
        for doc_file in directory.rglob(ext):
            try:
                content = doc_file.read_text(encoding="utf-8", errors="ignore")

                # Skip very short files
                if len(content) < 100:
                    continue

                doc_id = await ctx.rag.index_documentation(
                    content,
                    metadata={
                        "source": "manim-docs",
                        "filename": doc_file.name,
                        "type": doc_file.suffix[1:],  # py, rst, md
                    },
                )

                if doc_id:
                    indexed += 1

            except Exception as e:
                logger.debug("Failed to index %s: %s", doc_file, e)

    return indexed


def _is_manim_scene(content: str) -> bool:
    """Check if Python content looks like a Manim scene."""
    # Must have manim import
    if "from manim import" not in content and "import manim" not in content:
        return False

    # Must have a Scene class
    if "(Scene)" not in content and "(ThreeDScene)" not in content:
        if "(MovingCameraScene)" not in content:
            return False

    return True
