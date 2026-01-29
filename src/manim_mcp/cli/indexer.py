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
# Use manimgl (3b1b's fork), NOT Manim Community Edition!
REPO_MANIMGL_DOCS = "https://github.com/3b1b/manim.git"


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
    elif args.index_command == "huggingface":
        return await cmd_index_huggingface(args, ctx, printer)
    elif args.index_command == "custom":
        return await cmd_index_custom(args, ctx, printer)
    elif args.index_command == "clear":
        return await cmd_index_clear(args, ctx, printer)
    elif args.index_command == "api":
        return await cmd_index_api(args, ctx, printer)
    elif args.index_command == "errors":
        return await cmd_index_errors(args, ctx, printer)
    elif args.index_command == "patterns":
        return await cmd_index_patterns(args, ctx, printer)
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
    printer.info(f"  API signatures: {stats['api_count']}")
    printer.info(f"  Animation patterns: {stats['patterns_count']}")

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
    """Index manimgl documentation and examples (NOT Manim Community Edition!)."""
    import asyncio
    from manim_mcp.cli.output import spinner

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    printer.info("Cloning manimgl (3b1b/manim) repository for documentation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "manimgl"
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", REPO_MANIMGL_DOCS, str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            printer.error(f"Failed to clone repository: {stderr.decode()}")
            return 1

        # Index manimgl source code as documentation (it has good docstrings)
        manimlib_path = repo_path / "manimlib"
        if manimlib_path.exists():
            count = await _index_docs_directory(manimlib_path, ctx, printer)
            printer.success(f"Indexed {count} manimgl source files as documentation")

        # Also index example scenes if they exist
        examples_path = repo_path / "example_scenes"
        if examples_path.exists():
            count = await _index_docs_directory(examples_path, ctx, printer)
            printer.success(f"Indexed {count} manimgl example scenes")

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
    valid_collections = ["scenes", "docs", "errors", "api", "patterns", "all"]
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
            ctx.config.rag_collection_api,
            ctx.config.rag_collection_patterns,
        ]
    elif collection == "scenes":
        collections_to_clear = [ctx.config.rag_collection_scenes]
    elif collection == "docs":
        collections_to_clear = [ctx.config.rag_collection_docs]
    elif collection == "errors":
        collections_to_clear = [ctx.config.rag_collection_errors]
    elif collection == "api":
        collections_to_clear = [ctx.config.rag_collection_api]
    elif collection == "patterns":
        collections_to_clear = [ctx.config.rag_collection_patterns]

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
    """Index all Python files containing Manim scenes using AST-based chunking."""
    from manim_mcp.cli.output import spinner

    indexed = 0
    skipped = 0
    chunks_indexed = 0

    # Find all .py files
    py_files = list(directory.rglob("*.py"))
    printer.info(f"Found {len(py_files)} Python files")

    with spinner(f"Indexing {len(py_files)} files with AST chunking"):
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")

                # Only index files that look like Manim scenes
                if not _is_manim_scene(content):
                    skipped += 1
                    continue

                # Extract scene chunks using AST
                chunks = _extract_scene_chunks(content, py_file.name)

                for chunk in chunks:
                    doc_id = await ctx.rag.index_manim_code(
                        chunk["code"],
                        metadata={
                            "source": source,
                            "filename": py_file.name,
                            "path": str(py_file.relative_to(directory)),
                            "chunk_type": chunk["type"],
                            "scene_name": chunk.get("name", ""),
                            "methods": ",".join(chunk.get("methods", [])),
                        },
                    )

                    if doc_id:
                        chunks_indexed += 1

                indexed += 1

            except Exception as e:
                logger.debug("Failed to index %s: %s", py_file, e)
                skipped += 1

    printer.success(
        f"Indexed {indexed} files → {chunks_indexed} scene chunks "
        f"(skipped {skipped} non-scene files)"
    )
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


async def cmd_index_huggingface(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index the 3blue1brown-manim HuggingFace dataset."""
    from manim_mcp.cli.output import spinner

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    try:
        from datasets import load_dataset
    except ImportError:
        printer.error("datasets library not installed. Run: pip install datasets")
        return 1

    printer.info("Loading BibbyResearch/3blue1brown-manim dataset...")

    try:
        dataset = load_dataset("BibbyResearch/3blue1brown-manim", split="train")
    except Exception as e:
        printer.error(f"Failed to load dataset: {e}")
        return 1

    printer.info(f"Dataset loaded: {len(dataset)} examples")

    indexed = 0
    with spinner(f"Indexing {len(dataset)} prompt-code pairs"):
        for i, item in enumerate(dataset):
            try:
                # Dataset has 'prompt' and 'code' fields
                prompt = item.get("prompt", item.get("instruction", ""))
                code = item.get("code", item.get("output", item.get("response", "")))

                if not code or len(code) < 50:
                    continue

                doc_id = await ctx.rag.index_manim_code(
                    code,
                    metadata={
                        "source": "huggingface-3b1b",
                        "prompt": prompt[:200] if prompt else "",
                        "index": i,
                    },
                )

                if doc_id:
                    indexed += 1

            except Exception as e:
                logger.debug("Failed to index item %d: %s", i, e)

    printer.success(f"Indexed {indexed} examples from HuggingFace dataset")
    return 0


def _extract_scene_chunks(content: str, filename: str) -> list[dict]:
    """Extract scene classes using AST for better chunking.

    Instead of indexing whole files, extract individual Scene classes
    with their imports and helper functions for better retrieval.
    """
    import ast

    chunks = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # If AST fails, fall back to whole file
        return [{"code": content, "type": "file", "name": filename}]

    # Extract imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(content, node) or "")

    import_block = "\n".join(imports)

    # Extract Scene classes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's a Scene subclass
            base_names = [
                getattr(base, 'id', getattr(base, 'attr', ''))
                for base in node.bases
            ]
            is_scene = any(
                name in base_names for name in
                ["Scene", "ThreeDScene", "MovingCameraScene", "InteractiveScene", "ZoomedScene"]
            )

            if is_scene:
                # Get the full class source
                class_source = ast.get_source_segment(content, node) or ""

                # Combine imports + class for complete context
                full_chunk = f"{import_block}\n\n{class_source}"

                chunks.append({
                    "code": full_chunk,
                    "type": "scene_class",
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                })

    # If no scenes found, fall back to whole file
    if not chunks:
        chunks.append({"code": content, "type": "file", "name": filename})

    return chunks


def _is_manim_scene(content: str) -> bool:
    """Check if Python content looks like a Manim scene."""
    # Check for manim imports (community or 3b1b/manimgl)
    has_import = any(pattern in content for pattern in [
        "from manim import",
        "import manim",
        "from manimlib",
        "import manimlib",
        "manim_imports_ext",  # 3b1b's import pattern
    ])
    if not has_import:
        return False

    # Must have a Scene class (various types)
    has_scene = any(pattern in content for pattern in [
        "(Scene)",
        "(ThreeDScene)",
        "(MovingCameraScene)",
        "(InteractiveScene)",  # 3b1b's scene type
        "(ZoomedScene)",
        "class.*Scene):",  # Regex-like pattern won't work, keep explicit
    ])
    if not has_scene:
        return False

    return True


async def cmd_index_api(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index manimgl API signatures for parameter validation."""
    import asyncio
    from manim_mcp.cli.output import spinner
    from manim_mcp.cli.api_extractor import extract_api_signatures, get_known_manimgl_parameters

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    printer.info("Cloning manimgl (3b1b/manim) repository for API extraction...")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "manimgl"
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", REPO_MANIMGL_DOCS, str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            printer.error(f"Failed to clone repository: {stderr.decode()}")
            return 1

        manimlib_path = repo_path / "manimlib"
        if not manimlib_path.exists():
            printer.error("manimlib directory not found in repository")
            return 1

        # Extract API signatures using AST
        with spinner("Extracting API signatures from manimlib source"):
            signatures = extract_api_signatures(manimlib_path)

        printer.info(f"Extracted {len(signatures)} API signatures")

        # Index each signature
        indexed = 0
        with spinner(f"Indexing {len(signatures)} API signatures"):
            for sig in signatures:
                doc = sig.to_document()
                meta = sig.to_dict()

                # Flatten parameters for metadata (ChromaDB doesn't support nested dicts)
                param_names = [p.name for p in sig.parameters if p.name not in ("self", "cls")]
                meta["parameter_names"] = ",".join(param_names)
                # Remove nested parameters dict for ChromaDB compatibility
                if "parameters" in meta:
                    del meta["parameters"]

                doc_id = await ctx.rag.index_api_signature(doc, metadata=meta)
                if doc_id:
                    indexed += 1

        # Also index known parameter mappings (hardcoded critical methods)
        known_params = get_known_manimgl_parameters()
        printer.info(f"Indexing {len(known_params)} known critical method signatures")

        for full_name, param_info in known_params.items():
            parts = full_name.split(".")
            class_name = parts[0] if len(parts) > 1 else None
            method_name = parts[-1]

            doc = f"""# {full_name}

## Valid Parameters
{', '.join(param_info['valid'])}

## Invalid Parameters (CE-only, don't use)
{', '.join(param_info['invalid'])}

## Notes
These parameters are verified to work with manimgl. Do NOT use the invalid parameters - they will cause errors.
"""
            meta = {
                "id": full_name,
                "name": method_name,
                "class_name": class_name,
                "valid_params": ",".join(param_info["valid"]),
                "invalid_params": ",".join(param_info["invalid"]),
                "is_verified": True,
            }

            doc_id = await ctx.rag.index_api_signature(doc, metadata=meta)
            if doc_id:
                indexed += 1

    printer.success(f"Indexed {indexed} API signatures")
    return 0


async def cmd_index_errors(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Pre-populate error patterns from code_bridge.py knowledge."""
    from manim_mcp.cli.output import spinner
    from manim_mcp.cli.api_extractor import get_error_patterns

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    error_patterns = get_error_patterns()
    printer.info(f"Indexing {len(error_patterns)} known error patterns")

    indexed = 0
    with spinner(f"Indexing {len(error_patterns)} error patterns"):
        for pattern in error_patterns:
            error_msg = f"Pattern: {pattern['pattern']}"
            if pattern.get("method"):
                error_msg += f" (in {pattern['method']})"

            fix = pattern["fix"]
            if pattern.get("example"):
                fix += f"\n\nExample:\n{pattern['example']}"

            doc_id = await ctx.rag.index_error_pattern(
                error=error_msg,
                fix=fix,
            )

            if doc_id:
                indexed += 1

    printer.success(f"Indexed {indexed} error patterns")
    return 0


async def cmd_index_patterns(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    """Index 3b1b animation patterns for code generation guidance."""
    from manim_mcp.cli.output import spinner
    from manim_mcp.cli.pattern_extractor import get_all_patterns

    if not ctx.rag.available:
        printer.error("ChromaDB not available - cannot index")
        return 1

    patterns = get_all_patterns()
    printer.info(f"Indexing {len(patterns)} animation patterns")

    indexed = 0
    with spinner(f"Indexing {len(patterns)} animation patterns"):
        for pattern in patterns:
            doc = pattern.to_document()
            meta = pattern.to_metadata()

            doc_id = await ctx.rag.index_animation_pattern(doc, metadata=meta)

            if doc_id:
                indexed += 1

    printer.success(f"Indexed {indexed} animation patterns")
    return 0
