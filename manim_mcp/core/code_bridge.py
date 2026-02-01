"""Code Bridge: Transforms Manim Community Edition code to manimgl (3Blue1Brown's fork).

LLMs often generate Manim CE code because it's more common in training data.
This module transforms CE patterns to manimgl equivalents before rendering.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


class CEToManimglTransformer(ast.NodeTransformer):
    """AST transformer that converts Manim CE patterns to manimgl."""

    # CE class names → manimgl equivalents
    CLASS_MAPPINGS = {
        # Tex-related
        "MathTex": "Tex",
        "SingleStringMathTex": "Tex",
        "OldTex": "Tex",  # Legacy 3b1b alias
        "OldTexText": "TexText",  # Legacy 3b1b alias
        # Shapes
        "SurroundRectangle": "SurroundingRectangle",  # CE uses SurroundRectangle
        "BackgroundRectangle": "SurroundingRectangle",  # CE has BackgroundRectangle
        "Cutout": "VMobject",  # CE-only, fallback
        # Scene types (manimgl uses Scene for everything)
        "ThreeDScene": "Scene",  # manimgl auto-detects 3D
        "ZoomedScene": "Scene",  # CE-only
        "MovingCameraScene": "Scene",  # CE-only, manimgl uses frame
        "VectorScene": "Scene",  # CE-only
        # Mobject types
        "MarkupText": "TexText",  # CE uses MarkupText for Pango, manimgl uses TexText
        "Paragraph": "TexText",  # CE-only, approximate
        "Title": "TexText",  # CE-only convenience class
        "BulletedList": "VGroup",  # CE-only
        "Code": "TexText",  # CE-only syntax highlighting
        # Math/geometric
        "ComplexPlane": "NumberPlane",  # Same functionality
        "PolarPlane": "NumberPlane",  # CE-only, approximate
        # Indicators
        "Cross": "VGroup",  # CE-only, fallback
        "Checkmark": "Tex",  # CE-only
        "Exmark": "Tex",  # CE-only
        # Path tracing (both support TracedPath, ensure it's recognized)
        "TracedPath": "TracedPath",  # Same in both, but ensure proper handling
        "VMobjectTrace": "TracedPath",  # CE alias
    }

    # CE function/animation names → manimgl equivalents
    FUNCTION_MAPPINGS = {
        "Create": "ShowCreation",
        "Uncreate": "Uncreate",  # Same in both
        "Unwrite": "Uncreate",  # CE has Unwrite, manimgl uses Uncreate
        "SpiralIn": "GrowFromCenter",  # CE-only, fallback to similar effect
        "AddTextLetterByLetter": "Write",  # CE-only text animation
        "AddTextWordByWord": "Write",  # CE-only text animation
        "RemoveTextLetterByLetter": "Uncreate",  # CE-only
        "TypeWithCursor": "Write",  # CE-only
        "UntypeWithCursor": "Uncreate",  # CE-only
        "Wiggle": "WiggleOutThenIn",  # Different name
        "Blink": "VFadeIn",  # CE-only, approximate with VFade
        "FadeToColor": "ApplyMethod",  # Handled differently in manimgl
        "ScaleInPlace": "ApplyMethod",  # Use mob.animate.scale() instead
        "ShrinkToCenter": "ApplyMethod",  # Use mob.animate.scale(0) instead
    }

    # Animations that accept multiple mobjects as varargs in CE but only one in manimgl
    # CE: FadeOut(mob1, mob2, mob3) → manimgl: FadeOut(VGroup(mob1, mob2, mob3))
    # This is a CRITICAL difference that causes "shapes cannot be broadcast" errors
    VARARGS_TO_VGROUP = {
        "FadeOut",
        "FadeIn",
        "GrowFromCenter",
        "GrowFromPoint",
        "GrowFromEdge",
        "ShowCreation",  # Might get multiple in CE
        "Write",
        "Uncreate",
    }

    # CE method names → manimgl equivalents
    METHOD_MAPPINGS = {
        # Axes/Graph methods
        "add_coordinates": "add_coordinate_labels",
        "get_coordinate_labels": "add_coordinate_labels",  # LLM hallucination - method doesn't exist
        "get_coordinates": "add_coordinate_labels",  # Another common hallucination
        "show_coordinates": "add_coordinate_labels",  # Another hallucination variant
        "plot": "get_graph",  # axes.plot() → axes.get_graph()
        "get_area": "get_area_under_graph",  # CE uses get_area, manimgl uses get_area_under_graph
        "get_riemann_rectangles": "get_riemann_rectangles",  # same in both
        "get_line_graph": "get_graph",  # CE-only
        "plot_line_graph": "get_graph",  # CE-only
        "plot_surface": "get_graph",  # CE 3D, approximate
        # Sizing methods (CE has scale_to_fit_*, manimgl uses set_*)
        "scale_to_fit_width": "set_width",  # CE → manimgl
        "scale_to_fit_height": "set_height",  # CE → manimgl
        "scale_to_fit_depth": "set_depth",  # CE → manimgl
        # Arrangement methods
        "arrange_submobjects": "arrange",  # CE uses arrange_submobjects
        "arrange_in_grid": "arrange_in_grid",  # Same but check params
        # Color methods
        "set_color_by_gradient": "set_color",  # CE-only, approximate
        "set_colors_by_radial_gradient": "set_color",  # CE-only
        # Positioning methods
        "align_to": "align_to",  # Same in both
        "match_x": "match_x",  # Same
        "match_y": "match_y",  # Same
        "match_z": "match_z",  # Same
        "match_width": "match_width",  # Same
        "match_height": "match_height",  # Same
        # Transform helpers
        "become": "become",  # Same in both
        "copy": "copy",  # Same
        # Animation-related
        "add_updater": "add_updater",  # Same
        "remove_updater": "remove_updater",  # Same
        "clear_updaters": "clear_updaters",  # Same
        # Note: set_color_by_tex exists in manimgl with same signature
    }

    # Parameters that need to be combined (e.g., x_min/x_max → x_range)
    # Format: {class: {(param1, param2): new_param_name}}
    PARAMS_TO_COMBINE = {
        "Axes": {
            ("x_min", "x_max"): "x_range",
            ("y_min", "y_max"): "y_range",
        },
        "NumberPlane": {
            ("x_min", "x_max"): "x_range",
            ("y_min", "y_max"): "y_range",
        },
        "ThreeDAxes": {
            ("x_min", "x_max"): "x_range",
            ("y_min", "y_max"): "y_range",
            ("z_min", "z_max"): "z_range",
        },
        "NumberLine": {
            ("x_min", "x_max"): "x_range",
        },
    }

    # Parameters to remove entirely (not supported in manimgl)
    PARAMS_TO_REMOVE = {
        # Coordinate systems
        "Axes": {"tips", "x_length", "y_length", "include_numbers", "label_direction", "axis_config"},
        "NumberPlane": {"tips", "x_length", "y_length", "include_numbers", "label_direction", "background_line_style", "faded_line_style", "faded_line_ratio"},
        "NumberLine": {"include_tip", "tip_width", "tip_height", "numbers_with_elongated_ticks", "label_direction", "include_numbers", "line_to_number_direction", "decimal_number_config"},
        "ThreeDAxes": {"tips", "x_length", "y_length", "include_numbers", "label_direction"},
        # Arrows
        "Arrow": {"tip_length", "tip_width", "max_tip_length_to_length_ratio", "max_stroke_width_to_length_ratio"},
        "Vector": {"tip_length", "tip_width"},
        "DoubleArrow": {"tip_length", "tip_width"},
        # Text
        "Tex": {"tex_environment"},  # CE-only param
        "MathTex": {"tex_environment"},
        "Text": {"line_spacing", "disable_ligatures"},  # CE-only params
        # Shapes
        "Rectangle": {"grid_xstep", "grid_ystep", "mark_paths_closed"},  # CE-only
        "Circle": {"num_components"},  # CE-only param
        "Polygon": {"num_components"},
        # Animations
        "FadeIn": {"target_position"},  # CE-only param - causes issues
        "FadeOut": {"target_position"},  # CE-only param
        "Write": {"reverse", "remover"},  # Different in manimgl
        "Transform": {"replace_mobject_with_target_in_scene"},  # Set via class attr in manimgl
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
        if func_name in self.PARAMS_TO_REMOVE or func_name in self.PARAM_MAPPINGS or func_name in self.PARAMS_TO_COMBINE:
            node.keywords = self._transform_keywords(func_name, node.keywords)

        # Handle varargs animations: FadeOut(a, b, c) → FadeOut(VGroup(a, b, c))
        # In ManimGL, FadeOut/FadeIn only accept a single mobject; extra args become kwargs
        if func_name in self.VARARGS_TO_VGROUP and len(node.args) > 1:
            # Wrap all positional args in VGroup
            vgroup_call = ast.Call(
                func=ast.Name(id="VGroup", ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )
            node.args = [vgroup_call]
            self.transformations_made.append(
                f"varargs: {func_name}(a, b, ...) → {func_name}(VGroup(a, b, ...))"
            )

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
        params_to_combine = self.PARAMS_TO_COMBINE.get(func_name, {})

        # First pass: collect values for params that need combining
        # and identify which params to skip in the second pass
        param_values: dict[str, ast.expr] = {}
        params_being_combined: set[str] = set()

        for (param1, param2), _ in params_to_combine.items():
            params_being_combined.add(param1)
            params_being_combined.add(param2)

        for kw in keywords:
            if kw.arg in params_being_combined:
                param_values[kw.arg] = kw.value

        new_keywords = []
        combined_params_added: set[str] = set()  # Track which combined params we've added

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

            # Check if this param should be combined with another
            combined = False
            for (param1, param2), new_param in params_to_combine.items():
                if kw.arg in (param1, param2):
                    combined = True
                    # Only add the combined param once
                    if new_param not in combined_params_added:
                        # Check if we have both values
                        if param1 in param_values and param2 in param_values:
                            # Create a tuple (param1_value, param2_value)
                            combined_value = ast.Tuple(
                                elts=[param_values[param1], param_values[param2]],
                                ctx=ast.Load()
                            )
                            new_kw = ast.keyword(arg=new_param, value=combined_value)
                            new_keywords.append(new_kw)
                            combined_params_added.add(new_param)
                            self.transformations_made.append(
                                f"param: {func_name}({param1}=..., {param2}=...) → {func_name}({new_param}=(...))"
                            )
                        elif param1 in param_values:
                            # Only have one param - just pass it through with warning
                            self.transformations_made.append(
                                f"param: {func_name}({param1}=...) → removed (missing {param2} for {new_param})"
                            )
                        elif param2 in param_values:
                            self.transformations_made.append(
                                f"param: {func_name}({param2}=...) → removed (missing {param1} for {new_param})"
                            )
                    break

            if combined:
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
        # === CLASS NAME FIXES ===
        # MathTex that might have been missed
        (r"\bMathTex\s*\(", "Tex("),
        (r"\bSingleStringMathTex\s*\(", "Tex("),
        # Legacy 3b1b aliases
        (r"\bOldTex\s*\(", "Tex("),
        (r"\bOldTexText\s*\(", "TexText("),
        # CE-only classes → manimgl alternatives
        (r"\bMarkupText\s*\(", "TexText("),
        (r"\bParagraph\s*\(", "TexText("),
        (r"\bTitle\s*\(", "TexText("),
        (r"\bBulletedList\s*\(", "VGroup("),
        (r"\bCode\s*\(", "TexText("),
        # Scene types
        (r"class\s+(\w+)\s*\(\s*ThreeDScene\s*\)", r"class \1(Scene)"),
        (r"class\s+(\w+)\s*\(\s*ZoomedScene\s*\)", r"class \1(Scene)"),
        (r"class\s+(\w+)\s*\(\s*MovingCameraScene\s*\)", r"class \1(Scene)"),
        (r"class\s+(\w+)\s*\(\s*VectorScene\s*\)", r"class \1(Scene)"),

        # === ANIMATION NAME FIXES ===
        # Create animation (CE) → ShowCreation (manimgl)
        (r"\bCreate\s*\(", "ShowCreation("),
        # CE-only animations → manimgl alternatives
        (r"\bCircumscribe\s*\(", "Indicate("),
        (r"\bSpiralIn\s*\(", "GrowFromCenter("),
        (r"\bUnwrite\s*\(", "Uncreate("),
        (r"\bAddTextLetterByLetter\s*\(", "Write("),
        (r"\bAddTextWordByWord\s*\(", "Write("),
        (r"\bRemoveTextLetterByLetter\s*\(", "Uncreate("),
        (r"\bTypeWithCursor\s*\(", "Write("),
        (r"\bUntypeWithCursor\s*\(", "Uncreate("),
        (r"\bWiggle\s*\(", "WiggleOutThenIn("),
        (r"\bBlink\s*\(", "VFadeIn("),
        (r"\bFadeToColor\s*\(", "ApplyMethod("),
        (r"\bScaleInPlace\s*\(", "ApplyMethod("),
        (r"\bShrinkToCenter\s*\(", "ApplyMethod("),
        (r"\bSpinInFromNothing\s*\(", "GrowFromCenter("),  # CE-only
        (r"\bFadeInFrom\s*\(", "FadeIn("),  # CE deprecated
        (r"\bFadeOutAndShift\s*\(", "FadeOut("),  # CE deprecated

        # === METHOD NAME FIXES ===
        (r"\.add_coordinates\s*\(", ".add_coordinate_labels("),
        (r"\.add_coordinates_labels\s*\(", ".add_coordinate_labels("),  # Fix typo
        (r"\.get_coordinate_labels\s*\(", ".add_coordinate_labels("),  # LLM hallucination
        (r"\.get_coordinates\s*\(", ".add_coordinate_labels("),  # Another hallucination
        (r"\.show_coordinates\s*\(", ".add_coordinate_labels("),  # Another hallucination
        (r"\.plot\s*\(", ".get_graph("),
        (r"\.get_area\s*\(", ".get_area_under_graph("),
        (r"\.arrange_submobjects\s*\(", ".arrange("),
        (r"\.get_line_graph\s*\(", ".get_graph("),
        (r"\.plot_line_graph\s*\(", ".get_graph("),
        # CE-only camera methods
        (r"self\.camera\.frame\.animate", "self.frame.animate"),  # CE camera syntax
        (r"self\.camera\.frame\.", "self.frame."),

        # === 3b1b CLASS ATTRIBUTE PATTERNS ===
        (r"self\.axes_color", "GREY"),
        (r"self\.graph_color", "BLUE"),
        (r"self\.area_color", "BLUE_E"),
        (r"self\.rect_color", "YELLOW"),
        (r"self\.label_color", "WHITE"),
        (r"self\.highlight_color", "YELLOW"),

        # === PARAMETER REMOVAL ===
        # Remove invalid parameters from get_riemann_rectangles
        (r",\s*n_rects\s*=\s*\d+", ""),
        (r"n_rects\s*=\s*\d+\s*,\s*", ""),
        (r",\s*riemann_sum_type\s*=\s*['\"][^'\"]*['\"]", ""),
        (r"riemann_sum_type\s*=\s*['\"][^'\"]*['\"]\s*,\s*", ""),
        # Remove target_position (CE-only for FadeIn/FadeOut)
        (r",\s*target_position\s*=\s*[^,\)]+", ""),
        (r"target_position\s*=\s*[^,\)]+\s*,\s*", ""),
        # Remove tex_environment (CE-only for Tex)
        (r",\s*tex_environment\s*=\s*['\"][^'\"]*['\"]", ""),
        (r"tex_environment\s*=\s*['\"][^'\"]*['\"]\s*,\s*", ""),

        # === CE-ONLY METHOD REMOVAL ===
        (r"\s*\w+\.add_tips\s*\([^)]*\)\s*\n?", "\n"),
        (r"\s*self\.add_tips\s*\([^)]*\)\s*\n?", "\n"),
        (r"\s*self\.add_fixed_in_frame_mobjects\s*\([^)]*\)\s*\n?", "\n"),  # CE-only

        # === COLOR NAME FIXES ===
        # CE uses American GRAY, manimgl uses British GREY
        (r"\bGRAY\b", "GREY"),
        (r"\bLIGHT_GRAY\b", "LIGHT_GREY"),
        (r"\bDARK_GRAY\b", "DARK_GREY"),
        # CE-specific color names
        (r"\bPURE_RED\b", "RED"),
        (r"\bPURE_GREEN\b", "GREEN"),
        (r"\bPURE_BLUE\b", "BLUE"),

        # === RATE FUNCTIONS FIX ===
        # CE uses rate_functions.smooth, manimgl uses smooth directly
        # rate_functions is a CE module; in manimgl these are top-level functions
        (r"\brate_functions\.smooth\b", "smooth"),
        (r"\brate_functions\.linear\b", "linear"),
        (r"\brate_functions\.there_and_back\b", "there_and_back"),
        (r"\brate_functions\.there_and_back_with_pause\b", "there_and_back_with_pause"),
        (r"\brate_functions\.rush_into\b", "rush_into"),
        (r"\brate_functions\.rush_from\b", "rush_from"),
        (r"\brate_functions\.slow_into\b", "slow_into"),
        (r"\brate_functions\.double_smooth\b", "double_smooth"),
        (r"\brate_functions\.lingering\b", "lingering"),
        (r"\brate_functions\.exponential_decay\b", "exponential_decay"),
        (r"\brate_functions\.running_start\b", "running_start"),
        (r"\brate_functions\.wiggle\b", "wiggle"),
        # CE ease_* functions don't exist in manimgl - replace with equivalents
        # ease_in_* (accelerating) → rush_into (similar behavior)
        # ease_out_* (decelerating) → rush_from (similar behavior)
        # ease_in_out_* (both) → smooth (similar behavior)
        (r"\brate_functions\.ease_in_sine\b", "rush_into"),
        (r"\brate_functions\.ease_out_sine\b", "rush_from"),
        (r"\brate_functions\.ease_in_out_sine\b", "smooth"),
        (r"\brate_functions\.ease_in_quad\b", "rush_into"),
        (r"\brate_functions\.ease_out_quad\b", "rush_from"),
        (r"\brate_functions\.ease_in_out_quad\b", "smooth"),
        (r"\brate_functions\.ease_in_cubic\b", "rush_into"),
        (r"\brate_functions\.ease_out_cubic\b", "rush_from"),
        (r"\brate_functions\.ease_in_out_cubic\b", "smooth"),
        (r"\brate_functions\.ease_in_expo\b", "rush_into"),
        (r"\brate_functions\.ease_out_expo\b", "rush_from"),
        (r"\brate_functions\.ease_in_out_expo\b", "smooth"),
        (r"\brate_functions\.ease_in_circ\b", "rush_into"),
        (r"\brate_functions\.ease_out_circ\b", "rush_from"),
        (r"\brate_functions\.ease_in_out_circ\b", "smooth"),
        (r"\brate_functions\.ease_in_back\b", "running_start"),
        (r"\brate_functions\.ease_out_back\b", "overshoot"),
        (r"\brate_functions\.ease_in_out_back\b", "smooth"),
        (r"\brate_functions\.ease_in_elastic\b", "wiggle"),
        (r"\brate_functions\.ease_out_elastic\b", "wiggle"),
        (r"\brate_functions\.ease_in_out_elastic\b", "wiggle"),
        (r"\brate_functions\.ease_in_bounce\b", "there_and_back"),
        (r"\brate_functions\.ease_out_bounce\b", "there_and_back"),
        (r"\brate_functions\.ease_in_out_bounce\b", "there_and_back"),
        # Generic fallback for any rate_functions.X we might have missed
        (r"\brate_functions\.(\w+)\b", r"\1"),

        # === BARE EASE FUNCTION NAMES (CE-only, don't exist in manimgl) ===
        # These are used directly without rate_functions. prefix
        # Must come AFTER the rate_functions. transforms above
        (r"\bease_in_sine\b", "rush_into"),
        (r"\bease_out_sine\b", "rush_from"),
        (r"\bease_in_out_sine\b", "smooth"),
        (r"\bease_in_quad\b", "rush_into"),
        (r"\bease_out_quad\b", "rush_from"),
        (r"\bease_in_out_quad\b", "smooth"),
        (r"\bease_in_cubic\b", "rush_into"),
        (r"\bease_out_cubic\b", "rush_from"),
        (r"\bease_in_out_cubic\b", "smooth"),
        (r"\bease_in_quart\b", "rush_into"),
        (r"\bease_out_quart\b", "rush_from"),
        (r"\bease_in_out_quart\b", "smooth"),
        (r"\bease_in_quint\b", "rush_into"),
        (r"\bease_out_quint\b", "rush_from"),
        (r"\bease_in_out_quint\b", "smooth"),
        (r"\bease_in_expo\b", "rush_into"),
        (r"\bease_out_expo\b", "rush_from"),
        (r"\bease_in_out_expo\b", "smooth"),
        (r"\bease_in_circ\b", "rush_into"),
        (r"\bease_out_circ\b", "rush_from"),
        (r"\bease_in_out_circ\b", "smooth"),
        (r"\bease_in_back\b", "running_start"),
        (r"\bease_out_back\b", "overshoot"),
        (r"\bease_in_out_back\b", "smooth"),
        (r"\bease_in_elastic\b", "wiggle"),
        (r"\bease_out_elastic\b", "wiggle"),
        (r"\bease_in_out_elastic\b", "wiggle"),
        (r"\bease_in_bounce\b", "there_and_back"),
        (r"\bease_out_bounce\b", "there_and_back"),
        (r"\bease_in_out_bounce\b", "there_and_back"),

        # === CONSTANT FIXES ===
        # CE uses DEGREES, manimgl uses DEG
        (r"\bDEGREES\b", "DEG"),
        # Frame dimensions (CE uses different names)
        (r"\bconfig\.frame_width\b", "FRAME_WIDTH"),
        (r"\bconfig\.frame_height\b", "FRAME_HEIGHT"),
        (r"\bconfig\.pixel_width\b", "1920"),
        (r"\bconfig\.pixel_height\b", "1080"),

        # === METHOD FIXES ===
        # set_stroke_opacity doesn't exist in manimgl - use set_stroke(opacity=X)
        (r"\.set_stroke_opacity\s*\(\s*([^)]+)\s*\)", r".set_stroke(opacity=\1)"),
        # set_fill_opacity doesn't exist in manimgl - use set_fill(opacity=X)
        (r"\.set_fill_opacity\s*\(\s*([^)]+)\s*\)", r".set_fill(opacity=\1)"),
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

    # Step 7: Fix varargs animations that AST might have missed (regex safety net)
    code = _fix_varargs_animations(code, transformations)

    # Step 8: Fix animate syntax differences
    code = _fix_animate_syntax(code, transformations)

    # Step 9: Fix TracedPath and always_redraw patterns
    code = _fix_traced_path_and_updaters(code, transformations)

    return code, transformations


def _fix_traced_path_and_updaters(code: str, transformations: list[str]) -> str:
    """Fix TracedPath and always_redraw patterns that often cause runtime errors.

    Common issues:
    1. TracedPath with incorrect callable signature
    2. always_redraw with set_value() instead of get_value()
    3. add_updater with dt parameter when not needed
    """
    import re

    # Fix 1: TracedPath needs a callable that returns a point
    # Common mistake: TracedPath(mob.get_center) instead of TracedPath(mob.get_center)
    # Actually this is usually correct, but let's handle traced_func issues

    # Fix 2: Warn about set_value in always_redraw (should be get_value)
    # Pattern: always_redraw(lambda: ... .set_value( ... )
    # This is a logic error - you read values in updaters, not set them
    if "always_redraw" in code and ".set_value(" in code:
        # Check if set_value appears in an always_redraw lambda
        pattern = r"always_redraw\s*\(\s*lambda[^:]*:\s*[^)]*\.set_value\s*\("
        if re.search(pattern, code):
            # Can't auto-fix this - it's a logic error
            # But log a warning that will help in debugging
            transformations.append(
                "WARNING: set_value() found in always_redraw - should probably be get_value()"
            )

    # Fix 3: Common always_redraw pattern issues
    # always_redraw(lambda dt: ...) - dt parameter is wrong for always_redraw
    # always_redraw takes no arguments, add_updater takes optional dt
    pattern = r"always_redraw\s*\(\s*lambda\s+dt\s*:"
    if re.search(pattern, code):
        code = re.sub(pattern, "always_redraw(lambda:", code)
        transformations.append("updater-fix: always_redraw lambda dt → lambda (no params)")

    # Fix 4: add_updater with dt when updating mobject based on another
    # Pattern: mob.add_updater(lambda m, dt: m.become(...))
    # This often doesn't need dt, but it's not wrong, so just note it

    # Fix 5: TracedPath dissipating_time parameter (CE-only)
    # Remove dissipating_time if present (manimgl uses different approach)
    pattern = r"TracedPath\s*\([^)]*dissipating_time\s*=\s*[^,)]+[,)]"
    if re.search(pattern, code):
        code = re.sub(r",\s*dissipating_time\s*=\s*[^,)]+", "", code)
        code = re.sub(r"dissipating_time\s*=\s*[^,)]+\s*,\s*", "", code)
        transformations.append("traced-path-fix: removed CE-only dissipating_time param")

    # Fix 6: TracedPath stroke_color vs color parameter
    # manimgl uses stroke_color, CE might use color
    pattern = r"TracedPath\s*\([^)]*\bcolor\s*="
    if re.search(pattern, code):
        # Only fix if there's no stroke_color already
        if "stroke_color" not in code:
            code = re.sub(r"(TracedPath\s*\([^)]*)(\bcolor\s*=)", r"\1stroke_color=", code)
            transformations.append("traced-path-fix: color → stroke_color")

    return code


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
    - Missing braces around \\to, \\infty, etc.
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

    # Fix common ellipsis issue: ... should be \\ldots in math mode
    # But only in Tex(), not TexText()
    if "Tex(" in code and "..." in code:
        # Replace ... with \\ldots only inside Tex() calls, not TexText()
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


def _fix_varargs_animations(code: str, transformations: list[str]) -> str:
    """Fix animations that take multiple mobjects in CE but single in manimgl.

    This is a regex-based safety net for cases the AST transformer missed.
    Converts FadeOut(a, b, c) → FadeOut(VGroup(a, b, c))
    """
    # Animations that need varargs → VGroup transformation
    varargs_animations = [
        "FadeOut", "FadeIn", "GrowFromCenter", "GrowFromPoint",
        "ShowCreation", "Write", "Uncreate"
    ]

    for anim in varargs_animations:
        # Pattern: AnimName(arg1, arg2, ...) where there are multiple comma-separated args
        # But NOT keyword args like shift=, scale=, etc.
        # This is tricky because we need to distinguish positional from keyword args

        # Simple heuristic: look for AnimName(something, something_else) where
        # something_else doesn't contain '=' before any comma or close paren
        pattern = rf"\b{anim}\s*\(\s*([^,\(\)]+(?:\[[^\]]*\])?)\s*,\s*([^=,\(\)]+(?:\[[^\]]*\])?(?:\s*,\s*[^=,\(\)]+(?:\[[^\]]*\])?)*)\s*\)"

        def make_replacement(m):
            first_arg = m.group(1).strip()
            rest_args = m.group(2).strip()
            # Check if rest_args contains '=' (keyword arg) - if so, don't transform
            if '=' in rest_args.split(',')[0]:
                return m.group(0)
            return f"{anim}(VGroup({first_arg}, {rest_args}))"

        if re.search(pattern, code):
            new_code = re.sub(pattern, make_replacement, code)
            if new_code != code:
                code = new_code
                transformations.append(f"varargs-fix: {anim}(a, b, ...) → {anim}(VGroup(...))")

    return code


def _fix_animate_syntax(code: str, transformations: list[str]) -> str:
    """Fix .animate syntax differences between CE and manimgl.

    Both use .animate, but there are subtle differences in method chaining.
    CE: mob.animate.scale(2).shift(UP)
    manimgl: Same, but some methods differ

    CRITICAL: ValueTracker.set_value() vs ValueTracker.animate.set_value()
    - tracker.set_value(x) - immediate update, NOT animatable
    - tracker.animate.set_value(x) - animated update, USE THIS in self.play()
    """
    # CE uses .animate.set_color() but manimgl might prefer .animate.set_fill()
    # This is generally compatible, but let's handle edge cases

    # CRITICAL FIX: ValueTracker set_value in self.play() context
    # LLM often generates: self.play(tracker.set_value, value, run_time=...)
    # Should be: self.play(tracker.animate.set_value(value), run_time=...)
    # Pattern 1: self.play(name.set_value, value, ...) - method reference with separate arg
    pattern1 = r"self\.play\s*\(\s*(\w+)\.set_value\s*,\s*([^,\)]+)"
    if re.search(pattern1, code):
        code = re.sub(pattern1, r"self.play(\1.animate.set_value(\2)", code)
        transformations.append("valuetracker-fix: tracker.set_value, val → tracker.animate.set_value(val)")

    # Pattern 2: self.play(name.set_value(value), ...) - direct call without animate
    # Need to be careful not to match tracker.animate.set_value which is correct
    pattern2 = r"self\.play\s*\(\s*(\w+)\.set_value\s*\(([^)]+)\)"
    if re.search(pattern2, code):
        # Check it's not already using .animate
        if not re.search(r"self\.play\s*\(\s*\w+\.animate\.set_value", code):
            code = re.sub(pattern2, r"self.play(\1.animate.set_value(\2))", code)
            transformations.append("valuetracker-fix: tracker.set_value(val) → tracker.animate.set_value(val)")

    # Pattern 3: LaggedStart or AnimationGroup containing set_value
    # e.g., LaggedStart(tracker.set_value, 5) or AnimationGroup(t.set_value(x))
    pattern3 = r"(LaggedStart|AnimationGroup|Succession)\s*\(\s*(\w+)\.set_value\s*,\s*([^,\)]+)"
    if re.search(pattern3, code):
        code = re.sub(pattern3, r"\1(\2.animate.set_value(\3)", code)
        transformations.append("valuetracker-fix: animation wrapper with set_value")

    # Fix animate.become() which doesn't exist - use Transform instead
    # CE: self.play(mob.animate.become(other))
    # This should be: self.play(Transform(mob, other))
    pattern = r"(\w+)\.animate\.become\(([^)]+)\)"
    if re.search(pattern, code):
        code = re.sub(pattern, r"Transform(\1, \2)", code)
        transformations.append("animate-fix: mob.animate.become() → Transform()")

    # Fix animate.match_style() if present
    pattern = r"(\w+)\.animate\.match_style\(([^)]+)\)"
    if re.search(pattern, code):
        # This is tricky - just remove the animation for now
        code = re.sub(pattern, r"\1.match_style(\2)", code)
        transformations.append("animate-fix: removed .animate from match_style")

    return code



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


def remove_dead_code(code: str) -> str:
    """Remove unused variables and duplicate assignments from code.

    Uses AST analysis to:
    - Track all variable assignments (name → line numbers)
    - Track all variable usages (Load context)
    - Find variables assigned but never used
    - Find variables assigned multiple times (keep only last assignment)
    - Remove dead lines from source

    Handles edge cases:
    - Preserves self.* attribute assignments
    - Preserves class attributes
    - Preserves variables used in nested scopes
    - Preserves underscore variables (_) which are intentionally unused

    Args:
        code: Python source code

    Returns:
        Code with dead code removed
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If we can't parse, return unchanged
        return code

    # Track assignments: name -> list of line numbers
    assignments: dict[str, list[int]] = {}
    # Track usages: name -> set of line numbers where used
    usages: dict[str, set[int]] = set()
    # Track all used variable names
    used_names: set[str] = set()

    class AssignmentVisitor(ast.NodeVisitor):
        """First pass: collect all assignments."""

        def __init__(self):
            self.current_class = None
            self.in_class_body = False

        def visit_ClassDef(self, node: ast.ClassDef):
            old_class = self.current_class
            old_in_body = self.in_class_body
            self.current_class = node.name
            self.in_class_body = True
            self.generic_visit(node)
            self.current_class = old_class
            self.in_class_body = old_in_body

        def visit_FunctionDef(self, node: ast.FunctionDef):
            # Function definitions are used if they're in a class or called
            self.in_class_body = False
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.in_class_body = False
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Skip underscore (intentionally unused) and class attributes at top level
                    if name != '_' and not (self.in_class_body and self.current_class):
                        if name not in assignments:
                            assignments[name] = []
                        assignments[name].append(node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                name = node.target.id
                if name != '_':
                    if name not in assignments:
                        assignments[name] = []
                    assignments[name].append(node.lineno)
            self.generic_visit(node)

    class UsageVisitor(ast.NodeVisitor):
        """Second pass: collect all variable usages."""

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            # self.something uses 'self', but we don't want to mark the attribute as used
            # unless it's being loaded
            self.generic_visit(node)

    # Run both visitors
    assignment_visitor = AssignmentVisitor()
    assignment_visitor.visit(tree)

    usage_visitor = UsageVisitor()
    usage_visitor.visit(tree)

    # Find lines to remove
    lines_to_remove: set[int] = set()

    for name, line_numbers in assignments.items():
        if name not in used_names:
            # Variable never used - remove all assignments
            # But be careful: some "unused" variables are actually used in string interpolation,
            # format strings, or passed to exec/eval. We'll be conservative here.
            # Only remove if the assignment is a simple assignment, not a complex expression
            lines_to_remove.update(line_numbers)
        elif len(line_numbers) > 1:
            # Multiple assignments - keep only the last one before first use
            # This is tricky; for simplicity, we'll just remove duplicate consecutive assignments
            # to the same value (which often happens with color constants)
            pass  # TODO: More sophisticated analysis needed

    if not lines_to_remove:
        return code

    # Remove dead lines
    source_lines = code.split('\n')
    result_lines = []

    for i, line in enumerate(source_lines, start=1):
        if i not in lines_to_remove:
            result_lines.append(line)
        else:
            # Verify this line is a simple assignment before removing
            stripped = line.strip()
            # Only remove if it's a simple assignment (name = something)
            # Don't remove if it's part of a larger statement or has side effects
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]', stripped):
                # Check it's not modifying self or an object attribute
                if not stripped.startswith('self.') and '.' not in stripped.split('=')[0]:
                    logger.debug("[DEAD-CODE] Removing unused assignment at line %d: %s", i, stripped[:50])
                    continue
            result_lines.append(line)

    return '\n'.join(result_lines)


def fix_variable_shadowing(code: str) -> str:
    """Fix patterns like `BLUE_A = BLUE_A` where a variable shadows itself.

    These patterns typically occur when:
    - LLM generates redundant assignments
    - Copy-paste errors
    - Mistaken attempt to create a local reference

    Transformations:
    - `NAME = NAME` where both sides are identical → `NAME_ALIAS = NAME` or removes the line
    - `BLUE_A = BLUE_A` → Removed (since it's a no-op)
    - `color = color` → Removed

    Args:
        code: Python source code

    Returns:
        Code with variable shadowing fixed
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Track lines with self-shadowing assignments
    shadowing_lines: dict[int, tuple[str, str]] = {}  # line -> (name, suggested_fix)

    class ShadowingVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            # Check for simple NAME = NAME patterns
            if len(node.targets) == 1:
                target = node.targets[0]
                value = node.value

                if isinstance(target, ast.Name) and isinstance(value, ast.Name):
                    if target.id == value.id:
                        # Found self-shadowing: NAME = NAME
                        name = target.id
                        # Suggest a fix: append _ALIAS or _COLOR suffix
                        if name.isupper():
                            # Constants like BLUE_A - these should just be removed
                            suggested = None
                        else:
                            # Variables like color - suggest alias
                            suggested = f"{name}_copy"

                        shadowing_lines[node.lineno] = (name, suggested)

            self.generic_visit(node)

    visitor = ShadowingVisitor()
    visitor.visit(tree)

    if not shadowing_lines:
        return code

    # Fix shadowing lines
    source_lines = code.split('\n')
    result_lines = []

    for i, line in enumerate(source_lines, start=1):
        if i in shadowing_lines:
            name, suggested = shadowing_lines[i]
            if suggested is None:
                # Remove the line entirely (it's a no-op)
                logger.debug("[SHADOWING] Removing no-op assignment at line %d: %s = %s", i, name, name)
                continue
            else:
                # Replace with suggested name
                # Find the pattern and replace the target
                pattern = rf'^(\s*){re.escape(name)}\s*=\s*{re.escape(name)}\s*$'
                replacement = rf'\1{suggested} = {name}'
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    logger.debug("[SHADOWING] Fixed at line %d: %s -> %s", i, line.strip(), new_line.strip())
                    result_lines.append(new_line)
                    continue

        result_lines.append(line)

    return '\n'.join(result_lines)


def _fix_mixed_quotes(line: str, line_num: int) -> tuple[str, list[str]]:
    """Fix strings with mixed quote types that cause syntax errors.

    Detects patterns like:
    - Text("Newton's First Law') - started with " but seems to end with '
    - Text('Some text with "quotes"') - actually valid, leave alone

    Strategy:
    1. Find potential string starts (quote after ( or = or ,)
    2. Track which quote opened the string
    3. Look for likely string content followed by wrong quote type
    4. Fix by using the correct closing quote

    Returns:
        Tuple of (fixed_line, list of fixes applied)
    """
    fixes = []

    # Skip lines without both quote types
    if '"' not in line or "'" not in line:
        return line, fixes

    # Skip triple-quoted strings
    if '"""' in line or "'''" in line:
        return line, fixes

    # Pattern: function call or assignment with string that has mismatched quotes
    # Look for: FuncName("content') or FuncName('content")
    # The key insight: a string started with " should end with ", not '

    # Find all potential string openings: ( or = or , followed by quote
    import re

    # Pattern to find mismatched quote strings in function calls
    # Match: (["']...[opposite quote])  where content doesn't contain the opening quote
    patterns = [
        # ("content') - double quote start, single quote end
        (r'\(\s*"([^"]*)\'\s*\)', r'("\1")'),
        # ('content") - single quote start, double quote end
        (r"\(\s*'([^']*)\"s*\)", r"('\1')"),
        # r"content') - raw string with mismatch
        (r'\(\s*r"([^"]*)\'\s*\)', r'(r"\1")'),
        (r"\(\s*r'([^']*)\"s*\)", r"(r'\1')"),
        # = "content' - assignment with mismatch
        (r'=\s*"([^"]*)\'(\s*[,\)\n])', r'="\1"\2'),
        (r"=\s*'([^']*)\"(\s*[,\)\n])", r"='\1'\2"),
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, line)
        if match:
            new_line = re.sub(pattern, replacement, line, count=1)
            if new_line != line:
                fixes.append(f"L{line_num}: fixed mismatched quote types")
                line = new_line
                break

    # More aggressive fix: detect "text with 'apostrophe' more text')
    # This is tricky because ' inside "" is valid, but LLM sometimes mixes them up
    # Look for lines ending with ') or ") where the quotes don't balance

    return line, fixes


def _fix_unterminated_strings(line: str, line_num: int) -> tuple[str, list[str]]:
    """Fix unterminated string literals by tracking which quote opened the string.

    Improved over naive approach:
    - Tracks which quote type STARTED the string
    - Closes with the SAME quote type
    - Handles escaped quotes properly
    - Handles raw strings (r"..." or r'...')

    Returns:
        Tuple of (fixed_line, list of fixes applied)
    """
    fixes = []

    # Skip triple-quoted strings
    if '"""' in line or "'''" in line:
        return line, fixes

    # Parse the line character by character to find string boundaries
    # Track: (start_pos, quote_char, is_raw)
    string_starts = []
    i = 0
    while i < len(line):
        char = line[i]

        # Check for raw string prefix
        is_raw = False
        if char in 'rRbBfFuU' and i + 1 < len(line) and line[i + 1] in '"\'':
            is_raw = char.lower() == 'r'
            i += 1
            char = line[i]

        if char in '"\'':
            # Check if this is a triple quote (skip those)
            if i + 2 < len(line) and line[i:i+3] == char * 3:
                # Skip past triple quote - find matching end
                end = line.find(char * 3, i + 3)
                if end != -1:
                    i = end + 3
                else:
                    i = len(line)
                continue

            # Check if we're starting or ending a string
            if not string_starts or string_starts[-1][1] != char:
                # Starting a new string (or this char doesn't match current string)
                # Check if we're inside another string
                if not string_starts:
                    string_starts.append((i, char, is_raw))
            else:
                # Check if this is escaped (if not raw string)
                escaped = False
                if not string_starts[-1][2]:  # Not a raw string
                    # Count preceding backslashes
                    num_backslashes = 0
                    j = i - 1
                    while j >= 0 and line[j] == '\\':
                        num_backslashes += 1
                        j -= 1
                    escaped = num_backslashes % 2 == 1

                if not escaped:
                    # Closing the string
                    string_starts.pop()

        i += 1

    # If we have unclosed strings, try to close them
    if string_starts:
        start_pos, quote_char, is_raw = string_starts[-1]
        stripped = line.rstrip()

        # Don't fix if line ends with backslash (continuation)
        if not stripped.endswith('\\'):
            # Check if we need to add closing paren too
            open_parens = stripped.count('(')
            close_parens = stripped.count(')')
            needs_close_paren = open_parens > close_parens

            # Build the fix
            fix_str = quote_char
            if needs_close_paren:
                fix_str += ')'
                fixes.append(f"L{line_num}: closed unterminated {quote_char} string (opened at col {start_pos+1}) and added )")
            else:
                fixes.append(f"L{line_num}: closed unterminated {quote_char} string (opened at col {start_pos+1})")

            line = stripped + fix_str + '\n'

    return line, fixes


def _regex_fallback_transform(code: str, transformations: list[str]) -> str:
    """Regex-based fallback when AST parsing fails."""
    replacements = [
        # Class names
        (r"\bMathTex\b", "Tex"),
        (r"\bSingleStringMathTex\b", "Tex"),
        (r"\bCreate\b", "ShowCreation"),

        # Text methods that don't exist in manimgl
        # .set_font_size(X) doesn't exist - remove the call (font_size should be set in constructor)
        (r"\.set_font_size\s*\([^)]*\)", ""),

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


def sanitize_latex_strings(code: str) -> tuple[str, list[str]]:
    """Fix common LaTeX string issues that cause syntax errors.

    LLMs often generate malformed strings with:
    - Unterminated string literals (missing closing quote)
    - Mixed quote types (started with " but contains unescaped ')
    - Unbalanced braces in LaTeX
    - Backslash at end of string causing issues

    Returns:
        Tuple of (fixed_code, list of fixes applied)
    """
    fixes = []
    lines = code.split('\n')
    fixed_lines = []

    # Track if we're inside a multi-line string
    in_multiline = False
    multiline_quote = None

    for i, line in enumerate(lines):
        original_line = line

        # Skip if we're in a multi-line string
        if in_multiline:
            if multiline_quote and multiline_quote in line:
                in_multiline = False
                multiline_quote = None
            fixed_lines.append(line)
            continue

        # Check for multi-line string start
        for quote in ['"""', "'''"]:
            count = line.count(quote)
            if count == 1:  # Opening without closing
                in_multiline = True
                multiline_quote = quote
                break

        # Fix 0: Apostrophe-in-text issue (ENHANCED)
        # LLM writes: TexText('Snell's Law') - apostrophe ends string prematurely
        # Also: TexText('The proof of Snell's law shows...') - apostrophe in MIDDLE
        # Python sees: TexText('Snell'  then  s Law')  - broken!
        # Fix: Find single-quoted strings containing word's pattern ANYWHERE
        #      and convert outer quotes to double quotes
        #
        # Strategy: Find all ('...') patterns and check if content has word's

        # Pattern 1: Simple case - possessive at START: ('Word's ...)
        apostrophe_match = re.search(r"\('(\w+)'s\s+(.+?)'\s*\)", line)
        if apostrophe_match:
            word1 = apostrophe_match.group(1)
            rest = apostrophe_match.group(2)
            full_content = f"{word1}'s {rest}"
            replacement = f'("{full_content}")'
            line = line[:apostrophe_match.start()] + replacement + line[apostrophe_match.end():]
            fixes.append(f"L{i+1}: fixed apostrophe-in-text with double quotes")
            fixed_lines.append(line)
            continue

        # Pattern 2: Possessive ANYWHERE in string: ('... word's ...')
        # Find single-quoted strings and check for internal possessives
        # Match: opening paren + single quote + content + single quote + closing paren
        # But content contains word's pattern
        sq_string_match = re.search(r"\('([^']*\w's[^']*)'\s*\)", line)
        if sq_string_match:
            content = sq_string_match.group(1)
            # Verify it has a possessive pattern (word followed by 's)
            if re.search(r"\w's\s", content) or re.search(r"\w's$", content):
                replacement = f'("{content}")'
                line = line[:sq_string_match.start()] + replacement + line[sq_string_match.end():]
                fixes.append(f"L{i+1}: fixed mid-string apostrophe with double quotes")
                fixed_lines.append(line)
                continue

        # Pattern 3: Single-quoted string WITHOUT parens but with possessive
        # e.g., label = 'Snell's Law'  or  title='Fermat's Principle'
        # Match: = 'content with word's' or , 'content with word's'
        bare_sq_match = re.search(r"([=,]\s*)'([^']*\w's[^']*)'\s*([,\)\]]|$)", line)
        if bare_sq_match:
            prefix = bare_sq_match.group(1)  # = or , with space
            content = bare_sq_match.group(2)
            suffix = bare_sq_match.group(3)
            if re.search(r"\w's\s", content) or re.search(r"\w's$", content):
                replacement = f'{prefix}"{content}"{suffix}'
                line = line[:bare_sq_match.start()] + replacement + line[bare_sq_match.end():]
                fixes.append(f"L{i+1}: fixed bare apostrophe string")
                fixed_lines.append(line)
                continue

        # Fix 0.5: Mixed quote detection and repair (NEW - more robust)
        # Detect strings that start with one quote type but seem to end with another
        # e.g., Text("Newton's First Law') - started with " but has unmatched '
        # Strategy: Parse the line to find properly opened strings and fix mismatched closers
        line, mixed_fixes = _fix_mixed_quotes(line, i + 1)
        if mixed_fixes:
            fixes.extend(mixed_fixes)
            fixed_lines.append(line)
            continue

        # Fix 1: Unterminated single-line strings (IMPROVED)
        # Now tracks which quote type STARTED the string to close with the SAME type
        line, unterminated_fixes = _fix_unterminated_strings(line, i + 1)
        if unterminated_fixes:
            fixes.extend(unterminated_fixes)

        # Fix 2: Unbalanced braces in r-strings (LaTeX)
        # Only for lines that look like Tex/TexText calls with r-strings
        if re.search(r'(Tex|TexText|MathTex)\s*\(\s*r["\']', line):
            # Count braces in the string portion
            match = re.search(r'r(["\'])(.*?)\1', line)
            if match:
                string_content = match.group(2)
                open_braces = string_content.count('{')
                close_braces = string_content.count('}')

                if open_braces > close_braces:
                    # Add missing closing braces
                    diff = open_braces - close_braces
                    quote = match.group(1)
                    end_pos = match.end() - 1  # Before closing quote
                    line = line[:end_pos] + '}' * diff + line[end_pos:]
                    fixes.append(f"L{i+1}: added {diff} missing closing braces in LaTeX")
                elif close_braces > open_braces:
                    # Remove extra closing braces (less common, but handle it)
                    diff = close_braces - open_braces
                    # Remove from end of string content
                    fixed_content = string_content
                    for _ in range(diff):
                        last_brace = fixed_content.rfind('}')
                        if last_brace >= 0:
                            fixed_content = fixed_content[:last_brace] + fixed_content[last_brace+1:]
                    line = line.replace(string_content, fixed_content)
                    fixes.append(f"L{i+1}: removed {diff} extra closing braces in LaTeX")

        # Fix 3: Backslash at end of line inside string (common LLM error)
        # This creates a line continuation that breaks the string
        stripped = line.rstrip()
        if stripped.endswith('\\') and not stripped.endswith('\\\\'):
            # Check if we're likely in a string context
            quote_count = stripped.count('"') + stripped.count("'")
            if quote_count % 2 == 1:  # Inside a string
                # Remove the trailing backslash
                line = stripped[:-1] + '\n'
                fixes.append(f"L{i+1}: removed trailing backslash in string")

        fixed_lines.append(line)

    fixed_code = '\n'.join(fixed_lines)

    if fixes:
        logger.info("[LATEX-SANITIZER] Applied %d fixes: %s",
                   len(fixes), "; ".join(fixes[:3]) + ("..." if len(fixes) > 3 else ""))

    return fixed_code, fixes


def bridge_code(
    code: str,
    validate_params: bool = True,
    rag: ChromaDBService | None = None,
) -> str:
    """Main entry point: Transform CE code to manimgl and ensure proper imports.

    This is the function that should be called before rendering.

    Args:
        code: Python code that may contain Manim CE patterns
        validate_params: Whether to validate and fix parameters (default True)
        rag: Optional ChromaDBService for dynamic API validation against 1,652+ indexed signatures

    Returns:
        Transformed code ready for manimgl execution
    """
    # First, sanitize LaTeX strings to fix common syntax errors
    # This runs BEFORE AST parsing to fix unterminated strings, unbalanced braces, etc.
    try:
        code, latex_fixes = sanitize_latex_strings(code)
    except Exception as e:
        logger.warning("[CODE-BRIDGE] LaTeX sanitization failed: %s", e)

    # Transform CE patterns to manimgl
    code, transformations = transform_ce_to_manimgl(code)

    if transformations:
        logger.info(
            "[CODE-BRIDGE] Made %d transformations: %s",
            len(transformations),
            "; ".join(transformations[:5]) + ("..." if len(transformations) > 5 else "")
        )

    # Validate and fix parameters against manimgl API
    # Uses dynamic ChromaDB validation when RAG is available
    if validate_params:
        try:
            from manim_mcp.core.param_validator import validate_and_fix
            code = validate_and_fix(code, rag=rag)
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

    # Clean up dead code and fix variable shadowing
    try:
        code = fix_variable_shadowing(code)
        code = remove_dead_code(code)
    except Exception as e:
        logger.warning("[CODE-BRIDGE] Dead code removal failed: %s", e)

    # Ensure proper import exists
    code = ensure_manimgl_import(code)

    return code
