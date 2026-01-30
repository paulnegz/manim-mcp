"""Code Bridge: Transforms Manim Community Edition code to manimgl (3Blue1Brown's fork).

LLMs often generate Manim CE code because it's more common in training data.
This module transforms CE patterns to manimgl equivalents before rendering.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class CEToManimglTransformer(ast.NodeTransformer):
    """AST transformer that converts Manim CE patterns to manimgl."""

    # CE class names → manimgl equivalents
    CLASS_MAPPINGS = {
        "MathTex": "Tex",
        "SingleStringMathTex": "Tex",
        "OldTex": "Tex",  # Legacy 3b1b alias
        "OldTexText": "TexText",  # Legacy 3b1b alias
        "SurroundRectangle": "SurroundingRectangle",  # CE uses SurroundRectangle, manimgl uses SurroundingRectangle
        # Text stays as Text in manimgl, but TexText is preferred for math labels
    }

    # CE function/animation names → manimgl equivalents
    FUNCTION_MAPPINGS = {
        "Create": "ShowCreation",
        # Most animations have same names
    }

    # CE method names → manimgl equivalents
    METHOD_MAPPINGS = {
        "add_coordinates": "add_coordinate_labels",
        "plot": "get_graph",  # axes.plot() → axes.get_graph()
        "get_area": "get_area_under_graph",  # CE uses get_area, manimgl uses get_area_under_graph
        "get_riemann_rectangles": "get_riemann_rectangles",  # same in both
        "arrange_submobjects": "arrange",  # CE uses arrange_submobjects, manimgl uses arrange
        # Note: set_color_by_tex exists in manimgl with same signature, don't transform
        # set_color_by_tex_to_color_map takes a DICT, set_color_by_tex takes (tex, color) args
    }

    # Parameters to remove entirely (not supported in manimgl)
    PARAMS_TO_REMOVE = {
        "Axes": {"tips", "x_length", "y_length", "include_numbers", "label_direction"},
        "NumberPlane": {"tips", "x_length", "y_length", "include_numbers", "label_direction"},
        "NumberLine": {"include_tip", "tip_width", "tip_height", "numbers_with_elongated_ticks", "label_direction", "include_numbers"},
        "ThreeDAxes": {"tips", "x_length", "y_length", "include_numbers", "label_direction"},
        "Arrow": {"tip_length", "tip_width", "max_tip_length_to_length_ratio", "max_stroke_width_to_length_ratio"},
        "Vector": {"tip_length", "tip_width"},
    }

    # Parameter name mappings (CE name → manimgl name)
    PARAM_MAPPINGS = {
        "Axes": {
            "x_length": "width",
            "y_length": "height",
        },
        "NumberPlane": {
            "x_length": "width",
            "y_length": "height",
        },
        "ThreeDAxes": {
            "x_length": "width",
            "y_length": "height",
        },
        "NumberLine": {
            "numbers_with_elongated_ticks": "big_tick_numbers",
        },
        # Method parameter mappings (for transformed methods)
        "get_area_under_graph": {
            "color": "fill_color",
            "opacity": "fill_opacity",
        },
        "get_riemann_rectangles": {
            "opacity": "fill_opacity",
            "sample_type": "input_sample_type",
            # Note: color/fill_color need special handling -> colors=(X,)
            # n_rects doesn't exist in manimgl - it uses dx
        },
    }

    def __init__(self) -> None:
        self.transformations_made: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Transform CE imports to manimgl."""
        if node.module == "manim":
            self.transformations_made.append("import: from manim → from manimlib")
            node.module = "manimlib"
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Transform CE imports to manimgl."""
        for alias in node.names:
            if alias.name == "manim":
                self.transformations_made.append("import: manim → manimlib")
                alias.name = "manimlib"
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Transform CE class names to manimgl equivalents."""
        if node.id in self.CLASS_MAPPINGS:
            new_name = self.CLASS_MAPPINGS[node.id]
            self.transformations_made.append(f"class: {node.id} → {new_name}")
            node.id = new_name
        elif node.id in self.FUNCTION_MAPPINGS:
            new_name = self.FUNCTION_MAPPINGS[node.id]
            self.transformations_made.append(f"function: {node.id} → {new_name}")
            node.id = new_name
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Transform CE method names to manimgl equivalents."""
        # First visit the value (handles chained attributes)
        node.value = self.visit(node.value)

        # Transform method names
        if node.attr in self.METHOD_MAPPINGS:
            new_name = self.METHOD_MAPPINGS[node.attr]
            self.transformations_made.append(f"method: .{node.attr}() → .{new_name}()")
            node.attr = new_name

        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Transform CE function calls, handling parameter mappings."""
        # First, visit children (handles nested calls, class name changes)
        node = self.generic_visit(node)

        # Get the function name being called
        func_name = self._get_call_name(node)
        if not func_name:
            return node

        # Handle parameter transformations for specific classes
        if func_name in self.PARAMS_TO_REMOVE or func_name in self.PARAM_MAPPINGS:
            node.keywords = self._transform_keywords(func_name, node.keywords)

        return node

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the function/class name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _transform_keywords(
        self, func_name: str, keywords: list[ast.keyword]
    ) -> list[ast.keyword]:
        """Transform keyword arguments for a specific function."""
        params_to_remove = self.PARAMS_TO_REMOVE.get(func_name, set())
        param_mappings = self.PARAM_MAPPINGS.get(func_name, {})

        new_keywords = []
        for kw in keywords:
            if kw.arg is None:
                # **kwargs - keep as is
                new_keywords.append(kw)
                continue

            if kw.arg in params_to_remove:
                # Remove this parameter entirely
                self.transformations_made.append(
                    f"param: {func_name}({kw.arg}=...) → removed"
                )
                continue

            if kw.arg in param_mappings:
                # Rename this parameter
                old_name = kw.arg
                new_name = param_mappings[old_name]
                self.transformations_made.append(
                    f"param: {func_name}({old_name}=...) → {func_name}({new_name}=...)"
                )
                kw.arg = new_name

            new_keywords.append(kw)

        return new_keywords


def transform_ce_to_manimgl(code: str) -> tuple[str, list[str]]:
    """Transform Manim CE code to manimgl.

    Args:
        code: Python code that may contain Manim CE patterns

    Returns:
        Tuple of (transformed_code, list_of_transformations_made)
    """
    transformations = []

    # Step 1: Simple string replacements for imports (handles edge cases AST might miss)
    import_replacements = [
        (r"from\s+manim\s+import", "from manimlib import"),
        (r"import\s+manim\b", "import manimlib"),
    ]
    for pattern, replacement in import_replacements:
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            transformations.append(f"regex: {pattern} → {replacement}")

    # Step 2: AST-based transformations for more complex patterns
    try:
        tree = ast.parse(code)
        transformer = CEToManimglTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Convert back to code
        code = ast.unparse(new_tree)
        transformations.extend(transformer.transformations_made)

    except SyntaxError as e:
        logger.warning("AST parse failed, falling back to regex-only: %s", e)
        # Fall back to regex-only transformations
        code = _regex_fallback_transform(code, transformations)

    # Step 3: Post-AST string replacements for patterns AST can't handle well
    post_replacements = [
        # MathTex that might have been missed
        (r"\bMathTex\s*\(", "Tex("),
        (r"\bSingleStringMathTex\s*\(", "Tex("),
        # Legacy 3b1b aliases
        (r"\bOldTex\s*\(", "Tex("),
        (r"\bOldTexText\s*\(", "TexText("),
        # Create animation (CE) → ShowCreation (manimgl)
        (r"\bCreate\s*\(", "ShowCreation("),
        # Method name differences
        (r"\.add_coordinates\s*\(", ".add_coordinate_labels("),
        (r"\.add_coordinates_labels\s*\(", ".add_coordinate_labels("),  # Fix common typo
        (r"\.plot\s*\(", ".get_graph("),
        (r"\.get_area\s*\(", ".get_area_under_graph("),
        (r"\.arrange_submobjects\s*\(", ".arrange("),
        # 3b1b class attribute patterns - replace with actual colors
        (r"self\.axes_color", "GREY"),
        (r"self\.graph_color", "BLUE"),
        (r"self\.area_color", "BLUE_E"),
        (r"self\.rect_color", "YELLOW"),
        (r"self\.label_color", "WHITE"),
        (r"self\.highlight_color", "YELLOW"),
        # Remove invalid parameters from get_riemann_rectangles (manimgl uses dx, not n_rects)
        (r",\s*n_rects\s*=\s*\d+", ""),
        (r"n_rects\s*=\s*\d+\s*,\s*", ""),
        (r",\s*riemann_sum_type\s*=\s*['\"][^'\"]*['\"]", ""),
        (r"riemann_sum_type\s*=\s*['\"][^'\"]*['\"]\s*,\s*", ""),
        # 3D Scene class: CE uses ThreeDScene, manimgl uses Scene (3D is automatic with 3D objects)
        (r"class\s+(\w+)\s*\(\s*ThreeDScene\s*\)", r"class \1(Scene)"),
        # Remove CE-only methods that don't exist in manimgl
        (r"\s*\w+\.add_tips\s*\([^)]*\)\s*\n?", "\n"),  # axes.add_tips() doesn't exist
        (r"\s*self\.add_tips\s*\([^)]*\)\s*\n?", "\n"),
        # Remove Circumscribe (CE only) - replace with Indicate
        (r"\bCircumscribe\s*\(", "Indicate("),
        # Color spelling: CE uses American GRAY, manimgl uses British GREY
        (r"\bGRAY\b", "GREY"),
        (r"\bLIGHT_GRAY\b", "LIGHT_GREY"),
        (r"\bDARK_GRAY\b", "DARK_GREY"),
    ]
    for pattern, replacement in post_replacements:
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            transformations.append(f"post-regex: {pattern}")

    # Step 4: Special handling for get_riemann_rectangles color params
    # manimgl uses colors=(COLOR,) not color= or fill_color=
    code = _fix_riemann_colors(code, transformations)

    # Step 5: Handle 3D scene camera methods (CE → manimgl)
    code = _fix_3d_camera(code, transformations)

    # Step 6: Fix LaTeX issues - mixed text+math in Tex should use TexText
    code = _fix_latex_issues(code, transformations)

    return code, transformations


def _fix_3d_camera(code: str, transformations: list[str]) -> str:
    """Transform CE ThreeDScene camera methods to manimgl equivalents.

    CE uses:
        self.set_camera_orientation(phi=75*DEGREES, theta=30*DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

    manimgl uses:
        self.frame.reorient(phi, theta)  # or just remove, 3D auto-works
    """
    # Remove set_camera_orientation calls (manimgl handles 3D automatically)
    # or transform to frame.reorient if we can parse the args
    pattern = r"self\.set_camera_orientation\s*\(\s*phi\s*=\s*([^,]+)\s*,\s*theta\s*=\s*([^,\)]+)[^)]*\)"
    if re.search(pattern, code):
        # Transform to frame.reorient (simpler manimgl equivalent)
        code = re.sub(pattern, r"self.frame.reorient(\1, \2)", code)
        transformations.append("3d-camera: set_camera_orientation → frame.reorient")

    # Remove other camera methods that don't have direct equivalents
    remove_patterns = [
        (r"self\.begin_ambient_camera_rotation\s*\([^)]*\)\s*\n?", ""),
        (r"self\.stop_ambient_camera_rotation\s*\([^)]*\)\s*\n?", ""),
        (r"self\.move_camera\s*\([^)]*\)\s*\n?", ""),
        # Simple set_camera_orientation without named args
        (r"self\.set_camera_orientation\s*\([^)]*\)\s*\n?", ""),
    ]

    for pattern, replacement in remove_patterns:
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            transformations.append(f"3d-camera: removed unsupported method")

    return code


def _fix_riemann_colors(code: str, transformations: list[str]) -> str:
    """Transform color=/fill_color= to colors=(X,) in get_riemann_rectangles calls.

    manimgl's get_riemann_rectangles uses `colors` parameter (iterable for gradient),
    not `color` or `fill_color`.
    """
    # Pattern to find get_riemann_rectangles calls with color= or fill_color=
    # Transform color=X to colors=(X,)
    patterns = [
        (r"(get_riemann_rectangles\s*\([^)]*),\s*fill_color\s*=\s*([A-Z_]+)", r"\1, colors=(\2,)"),
        (r"(get_riemann_rectangles\s*\([^)]*),\s*color\s*=\s*([A-Z_]+)", r"\1, colors=(\2,)"),
        (r"(get_riemann_rectangles\s*\([^)]*)fill_color\s*=\s*([A-Z_]+)\s*,", r"\1colors=(\2,), "),
        (r"(get_riemann_rectangles\s*\([^)]*)color\s*=\s*([A-Z_]+)\s*,", r"\1colors=(\2,), "),
    ]

    for pattern, replacement in patterns:
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            transformations.append(f"riemann-color-fix: color/fill_color → colors tuple")
            break  # Only apply one transformation

    return code


def _fix_latex_issues(code: str, transformations: list[str]) -> str:
    """Fix common LaTeX issues in manimgl code.

    Common issues:
    - Tex() with mixed text+math should be TexText() or split
    - Unescaped special characters in text mode
    - Missing braces around \to, \infty, etc.
    """
    # Fix Tex with mixed content containing text words + math
    # Pattern: Tex('word word $math$') -> TexText('word word $math$')
    # Look for Tex( with text that contains spaces AND $...$
    mixed_content_pattern = r"Tex\s*\(\s*['\"]([^'\"]*\s+[^'\"]*\$[^'\"]+\$[^'\"]*)['\"]"
    if re.search(mixed_content_pattern, code):
        code = re.sub(
            r"Tex\s*\(\s*(['\"])([^'\"]*\s+[^'\"]*\$[^'\"]+\$[^'\"]*)\1",
            r"TexText(\1\2\1",
            code
        )
        transformations.append("latex-fix: Tex with mixed content → TexText")

    # Fix common ellipsis issue: ... should be \ldots in math mode
    # But only in Tex(), not TexText()
    if "Tex(" in code and "..." in code:
        # Replace ... with \ldots only inside Tex() calls, not TexText()
        def fix_ellipsis(match):
            content = match.group(0)
            if "TexText" not in content:
                return content.replace("...", r"\\ldots ")
            return content
        code = re.sub(r"Tex\s*\([^)]+\)", fix_ellipsis, code)
        transformations.append("latex-fix: ... → \\ldots")

    # Fix: Single $ in Tex should probably be $$ or removed
    # Tex('text $x$ more') is problematic - use TexText instead
    single_dollar_pattern = r"Tex\s*\(\s*['\"]([^'\"]*[^\\]\$[^$]+\$[^'\"]*)['\"]"
    if re.search(single_dollar_pattern, code):
        code = re.sub(
            r"(Tex)\s*\(\s*(['\"])([^'\"]*[^\\]\$[^$]+\$[^'\"]*)\2",
            r"TexText(\2\3\2",
            code
        )
        transformations.append("latex-fix: Tex with inline math → TexText")

    return code


def _regex_fallback_transform(code: str, transformations: list[str]) -> str:
    """Regex-based fallback when AST parsing fails."""
    replacements = [
        # Class names
        (r"\bMathTex\b", "Tex"),
        (r"\bSingleStringMathTex\b", "Tex"),
        (r"\bCreate\b", "ShowCreation"),

        # Axes parameters (risky - might break valid code, but better than failing)
        (r",\s*tips\s*=\s*(True|False)", ""),
        (r"tips\s*=\s*(True|False)\s*,", ""),
        (r",\s*x_length\s*=", ", width="),
        (r",\s*y_length\s*=", ", height="),
        (r"x_length\s*=", "width="),
        (r"y_length\s*=", "height="),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, code):
            code = re.sub(pattern, replacement, code)
            transformations.append(f"fallback-regex: {pattern}")

    return code


def ensure_manimgl_import(code: str) -> str:
    """Ensure code has manimlib import, add if missing."""
    has_import = any(pattern in code for pattern in [
        "from manimlib import",
        "import manimlib",
        "from manim_imports_ext import",
        "from big_ol_pile_of_manim_imports import",
    ])

    if not has_import:
        code = "from manimlib import *\n\n" + code

    return code


def bridge_code(code: str, validate_params: bool = True) -> str:
    """Main entry point: Transform CE code to manimgl and ensure proper imports.

    This is the function that should be called before rendering.

    Args:
        code: Python code that may contain Manim CE patterns
        validate_params: Whether to validate and fix parameters (default True)

    Returns:
        Transformed code ready for manimgl execution
    """
    # Transform CE patterns to manimgl
    code, transformations = transform_ce_to_manimgl(code)

    if transformations:
        logger.info(
            "[CODE-BRIDGE] Made %d transformations: %s",
            len(transformations),
            "; ".join(transformations[:5]) + ("..." if len(transformations) > 5 else "")
        )

    # Validate and fix parameters against manimgl API
    if validate_params:
        try:
            from manim_mcp.core.param_validator import validate_and_fix
            code = validate_and_fix(code)
        except ImportError:
            logger.debug("[CODE-BRIDGE] param_validator not available, skipping validation")
        except Exception as e:
            logger.warning("[CODE-BRIDGE] Parameter validation failed: %s", e)

    # Fix layout issues (3b1b patterns: buff on edges, collision avoidance)
    try:
        from manim_mcp.core.layout_validator import validate_and_fix_layout
        code = validate_and_fix_layout(code)
    except ImportError:
        logger.debug("[CODE-BRIDGE] layout_validator not available, skipping")
    except Exception as e:
        logger.warning("[CODE-BRIDGE] Layout validation failed: %s", e)

    # Ensure proper import exists
    code = ensure_manimgl_import(code)

    return code
