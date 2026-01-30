"""AST-based Scene subclass discovery."""

from __future__ import annotations

import ast

from manim_mcp.models import SceneInfo

KNOWN_SCENE_BASES = frozenset({
    "Scene", "ThreeDScene", "MovingCameraScene", "ZoomedScene", "VectorScene",
})


class SceneParser:
    def parse_scenes(self, code: str) -> list[SceneInfo]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        scenes: list[SceneInfo] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            base_names = [_get_base_name(b) for b in node.bases]
            base_names = [n for n in base_names if n is not None]

            is_scene = any(name in KNOWN_SCENE_BASES or "Scene" in name for name in base_names)
            if not is_scene:
                continue

            has_construct = any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "construct"
                for item in node.body
            )

            scenes.append(SceneInfo(
                name=node.name,
                has_construct=has_construct,
                base_classes=base_names,
                line_number=node.lineno,
            ))

        return scenes


def _get_base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
