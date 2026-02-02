#!/usr/bin/env python3
"""Index 3b1b reusable components and helper functions into ChromaDB."""

import ast
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import chromadb

# Base paths
VIDEOS_PATH = Path("/mnt/volume_sfo2_01/PROD/manim-mcp/data/external/3b1b-videos")
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", 8000))


def extract_class_code(content: str, class_name: str) -> str:
    """Extract full class definition from file content."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Get line range
                start_line = node.lineno - 1
                end_line = node.end_lineno
                lines = content.split('\n')
                return '\n'.join(lines[start_line:end_line])
    except:
        pass
    return ""


def extract_function_code(content: str, func_name: str) -> str:
    """Extract full function definition from file content."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                lines = content.split('\n')
                return '\n'.join(lines[start_line:end_line])
    except:
        pass
    return ""


def get_base_classes(content: str, class_name: str) -> list[str]:
    """Get base classes for a class definition."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                return bases
    except:
        pass
    return []


def extract_config(code: str) -> dict:
    """Extract CONFIG dictionary from class code."""
    config = {}
    match = re.search(r'CONFIG\s*=\s*\{([^}]+)\}', code, re.DOTALL)
    if match:
        try:
            # Simple extraction of key names
            config_str = match.group(1)
            keys = re.findall(r'"(\w+)"\s*:|\'(\w+)\'\s*:', config_str)
            config = {k[0] or k[1]: True for k in keys}
        except:
            pass
    return config


def get_video_info(file_path: Path) -> dict:
    """Extract video/series info from file path."""
    parts = file_path.parts
    info = {"year": None, "series": None, "part": None}

    for i, part in enumerate(parts):
        if part.startswith("_20"):
            info["year"] = part[1:]
        elif part.startswith("part"):
            info["part"] = part
        elif i > 0 and parts[i-1].startswith("_20"):
            info["series"] = part

    return info


def find_components(videos_path: Path) -> list[dict]:
    """Find all VGroup/VMobject subclasses."""
    components = []

    # Component base classes to look for
    base_classes = {"VGroup", "VMobject", "Mobject", "SVGMobject", "ImageMobject",
                    "MobjectMatrix", "Group", "AnimatedBoundary"}

    for py_file in videos_path.rglob("*.py"):
        try:
            content = py_file.read_text(errors='ignore')
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = get_base_classes(content, node.name)

                    # Check if it's a component (inherits from mobject types)
                    # but NOT a Scene
                    if any(b in base_classes for b in bases) and "Scene" not in node.name:
                        code = extract_class_code(content, node.name)
                        if code and len(code) > 50:  # Skip trivial classes
                            video_info = get_video_info(py_file)
                            config = extract_config(code)

                            components.append({
                                "name": node.name,
                                "type": "component",
                                "base_classes": bases,
                                "code": code,
                                "file": str(py_file.relative_to(videos_path)),
                                "config_keys": list(config.keys()),
                                "year": video_info["year"],
                                "series": video_info["series"],
                                "part": video_info["part"],
                                "line_count": len(code.split('\n')),
                            })
        except Exception as e:
            print(f"Error processing {py_file}: {e}", file=sys.stderr)

    return components


def find_helpers(videos_path: Path) -> list[dict]:
    """Find all helper functions (get_*, create_*, make_*)."""
    helpers = []

    # Helper function prefixes
    prefixes = ("get_", "create_", "make_", "build_")

    for py_file in videos_path.rglob("*.py"):
        # Prioritize shared/helper files
        is_shared = any(x in py_file.name.lower() for x in ["shared", "helper", "common", "construct"])

        try:
            content = py_file.read_text(errors='ignore')
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith(prefixes):
                        code = extract_function_code(content, node.name)
                        if code and len(code) > 30:
                            video_info = get_video_info(py_file)

                            # Extract return type hint if present
                            return_type = None
                            if node.returns:
                                if isinstance(node.returns, ast.Name):
                                    return_type = node.returns.id
                                elif isinstance(node.returns, ast.Constant):
                                    return_type = str(node.returns.value)

                            # Extract parameter names
                            params = [arg.arg for arg in node.args.args]

                            helpers.append({
                                "name": node.name,
                                "type": "helper",
                                "params": params,
                                "return_type": return_type,
                                "code": code,
                                "file": str(py_file.relative_to(videos_path)),
                                "is_shared_file": is_shared,
                                "year": video_info["year"],
                                "series": video_info["series"],
                                "part": video_info["part"],
                                "line_count": len(code.split('\n')),
                            })
        except Exception as e:
            print(f"Error processing {py_file}: {e}", file=sys.stderr)

    return helpers


def index_to_chromadb(components: list[dict], helpers: list[dict]):
    """Index components and helpers to ChromaDB."""
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Create/get collections
    comp_collection = client.get_or_create_collection(
        name="manim_components",
        metadata={"description": "Reusable VGroup/VMobject components from 3b1b"}
    )

    helper_collection = client.get_or_create_collection(
        name="manim_helpers",
        metadata={"description": "Helper functions from 3b1b videos"}
    )

    # Index components
    print(f"Indexing {len(components)} components...")
    for comp in components:
        doc_id = f"comp_{comp['name']}_{comp['file'].replace('/', '_')}"

        document = f"""# Component: {comp['name']}
Base classes: {', '.join(comp['base_classes'])}
File: {comp['file']}
Series: {comp['series'] or 'standalone'}

```python
{comp['code']}
```
"""

        metadata = {
            "name": comp["name"],
            "type": "component",
            "base_classes": ",".join(comp["base_classes"]),
            "file": comp["file"],
            "config_keys": ",".join(comp["config_keys"]),
            "year": comp["year"] or "",
            "series": comp["series"] or "",
            "part": comp["part"] or "",
            "line_count": comp["line_count"],
            "indexed_at": datetime.now().isoformat(),
        }

        try:
            comp_collection.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])
        except Exception as e:
            print(f"Error indexing component {comp['name']}: {e}", file=sys.stderr)

    # Index helpers
    print(f"Indexing {len(helpers)} helper functions...")
    for helper in helpers:
        doc_id = f"helper_{helper['name']}_{helper['file'].replace('/', '_')}"

        document = f"""# Helper Function: {helper['name']}
Parameters: {', '.join(helper['params'])}
Returns: {helper['return_type'] or 'unspecified'}
File: {helper['file']}
Shared file: {helper['is_shared_file']}

```python
{helper['code']}
```
"""

        metadata = {
            "name": helper["name"],
            "type": "helper",
            "params": ",".join(helper["params"]),
            "return_type": helper["return_type"] or "",
            "file": helper["file"],
            "is_shared_file": helper["is_shared_file"],
            "year": helper["year"] or "",
            "series": helper["series"] or "",
            "part": helper["part"] or "",
            "line_count": helper["line_count"],
            "indexed_at": datetime.now().isoformat(),
        }

        try:
            helper_collection.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])
        except Exception as e:
            print(f"Error indexing helper {helper['name']}: {e}", file=sys.stderr)

    print(f"\nIndexed to ChromaDB:")
    print(f"  Components: {comp_collection.count()}")
    print(f"  Helpers: {helper_collection.count()}")


def main():
    print(f"Scanning {VIDEOS_PATH}...")

    # Find components and helpers
    components = find_components(VIDEOS_PATH)
    print(f"Found {len(components)} components")

    helpers = find_helpers(VIDEOS_PATH)
    print(f"Found {len(helpers)} helper functions")

    # Index to ChromaDB
    index_to_chromadb(components, helpers)

    print("\nDone!")


if __name__ == "__main__":
    main()
