"""AST-based API signature extraction for manimlib.

Extracts public method signatures from manimlib source code to enable:
1. Parameter validation before code execution
2. Auto-fixing invalid parameters
3. Better RAG retrieval for API queries
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParameterInfo:
    """Information about a function/method parameter."""

    name: str
    type_hint: str | None = None
    default: str | None = None
    is_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_hint,
            "default": self.default,
            "required": self.is_required,
        }


@dataclass
class APISignature:
    """Complete API signature for a function or method."""

    name: str
    class_name: str | None = None
    module: str | None = None
    parameters: list[ParameterInfo] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    source_file: str | None = None
    line_number: int | None = None

    @property
    def full_name(self) -> str:
        """Return fully qualified name like 'Axes.get_riemann_rectangles'."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    @property
    def signature_str(self) -> str:
        """Return function signature as a string."""
        params = []
        for p in self.parameters:
            if p.name in ("self", "cls"):
                params.append(p.name)
            elif p.type_hint and p.default:
                params.append(f"{p.name}: {p.type_hint} = {p.default}")
            elif p.type_hint:
                params.append(f"{p.name}: {p.type_hint}")
            elif p.default:
                params.append(f"{p.name}={p.default}")
            else:
                params.append(p.name)

        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"def {self.name}({', '.join(params)}){ret}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.full_name,
            "name": self.name,
            "class_name": self.class_name,
            "module": self.module,
            "signature": self.signature_str,
            "parameters": {p.name: p.to_dict() for p in self.parameters if p.name not in ("self", "cls")},
            "return_type": self.return_type,
            "docstring": self.docstring,
            "is_method": self.is_method,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }

    def to_document(self) -> str:
        """Convert to a document string for RAG indexing."""
        parts = [
            f"# {self.full_name}",
            "",
            f"```python",
            self.signature_str,
            "```",
        ]

        if self.docstring:
            parts.extend(["", "## Description", self.docstring])

        if self.parameters:
            parts.extend(["", "## Parameters"])
            for p in self.parameters:
                if p.name in ("self", "cls"):
                    continue
                type_str = f" ({p.type_hint})" if p.type_hint else ""
                default_str = f", default={p.default}" if p.default else ""
                required_str = " [required]" if p.is_required else ""
                parts.append(f"- **{p.name}**{type_str}{default_str}{required_str}")

        if self.class_name:
            parts.extend(["", f"## Class", f"Member of `{self.class_name}`"])

        return "\n".join(parts)


class APIExtractor(ast.NodeVisitor):
    """AST visitor that extracts API signatures from Python source."""

    def __init__(self, module_name: str | None = None, source_file: str | None = None):
        self.module_name = module_name
        self.source_file = source_file
        self.signatures: list[APISignature] = []
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition and extract method signatures."""
        # Skip private classes
        if node.name.startswith("_"):
            self.generic_visit(node)
            return

        old_class = self._current_class
        self._current_class = node.name

        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self._extract_function(item, is_method=True)

        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition at module level."""
        if self._current_class is None:
            self._extract_function(node, is_method=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition at module level."""
        if self._current_class is None:
            self._extract_function(node, is_method=False)

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool,
    ) -> None:
        """Extract signature from a function/method definition."""
        # Skip private functions (but allow __init__)
        if node.name.startswith("_") and node.name != "__init__":
            return

        # Check decorators
        is_classmethod = False
        is_staticmethod = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "classmethod":
                    is_classmethod = True
                elif decorator.id == "staticmethod":
                    is_staticmethod = True

        # Extract parameters
        parameters = self._extract_parameters(node.args)

        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._annotation_to_str(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        sig = APISignature(
            name=node.name,
            class_name=self._current_class,
            module=self.module_name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            source_file=self.source_file,
            line_number=node.lineno,
        )

        self.signatures.append(sig)

    def _extract_parameters(self, args: ast.arguments) -> list[ParameterInfo]:
        """Extract parameter information from function arguments."""
        parameters: list[ParameterInfo] = []

        # Calculate how many args have defaults
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_idx = num_args - num_defaults

        # Regular positional/keyword args
        for i, arg in enumerate(args.args):
            has_default = i >= first_default_idx
            default_value = None
            if has_default:
                default_idx = i - first_default_idx
                default_value = self._default_to_str(args.defaults[default_idx])

            type_hint = None
            if arg.annotation:
                type_hint = self._annotation_to_str(arg.annotation)

            parameters.append(ParameterInfo(
                name=arg.arg,
                type_hint=type_hint,
                default=default_value,
                is_required=not has_default,
            ))

        # *args
        if args.vararg:
            type_hint = None
            if args.vararg.annotation:
                type_hint = self._annotation_to_str(args.vararg.annotation)
            parameters.append(ParameterInfo(
                name=f"*{args.vararg.arg}",
                type_hint=type_hint,
                is_required=False,
            ))

        # keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            default_value = None
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                default_value = self._default_to_str(args.kw_defaults[i])

            type_hint = None
            if arg.annotation:
                type_hint = self._annotation_to_str(arg.annotation)

            parameters.append(ParameterInfo(
                name=arg.arg,
                type_hint=type_hint,
                default=default_value,
                is_required=default_value is None,
            ))

        # **kwargs
        if args.kwarg:
            type_hint = None
            if args.kwarg.annotation:
                type_hint = self._annotation_to_str(args.kwarg.annotation)
            parameters.append(ParameterInfo(
                name=f"**{args.kwarg.arg}",
                type_hint=type_hint,
                is_required=False,
            ))

        return parameters

    def _annotation_to_str(self, node: ast.expr) -> str:
        """Convert an annotation AST node to a string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "Any"

    def _default_to_str(self, node: ast.expr | None) -> str | None:
        """Convert a default value AST node to a string."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return "..."


def extract_api_signatures(manimlib_path: Path) -> list[APISignature]:
    """Extract all public method signatures from manimlib source.

    Args:
        manimlib_path: Path to manimlib directory

    Returns:
        List of APISignature objects
    """
    signatures: list[APISignature] = []

    if not manimlib_path.exists():
        logger.error("manimlib path does not exist: %s", manimlib_path)
        return signatures

    # Process all Python files
    for py_file in manimlib_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)

            # Calculate module name relative to manimlib
            rel_path = py_file.relative_to(manimlib_path.parent)
            module_name = str(rel_path.with_suffix("")).replace("/", ".")

            extractor = APIExtractor(
                module_name=module_name,
                source_file=str(rel_path),
            )
            extractor.visit(tree)

            signatures.extend(extractor.signatures)
            logger.debug("Extracted %d signatures from %s", len(extractor.signatures), py_file.name)

        except SyntaxError as e:
            logger.debug("Syntax error in %s: %s", py_file, e)
        except Exception as e:
            logger.debug("Error processing %s: %s", py_file, e)

    logger.info("Extracted %d total API signatures from manimlib", len(signatures))
    return signatures


def get_known_manimgl_parameters() -> dict[str, dict[str, list[str]]]:
    """Return known manimgl parameter names for key methods.

    This is a hardcoded fallback for critical methods where we know
    the exact parameters, even if AST extraction misses them.

    IMPORTANT: Entries here get `is_verified=True` status in ChromaDB,
    making them prioritized during RAG search for parameter validation.

    Returns:
        Dict mapping "Class.method" -> {"valid": [params], "invalid": [params]}
    """
    return {
        # === AXES & COORDINATE SYSTEMS ===
        "Axes.get_riemann_rectangles": {
            "valid": ["graph", "x_range", "dx", "input_sample_type", "fill_opacity", "colors", "stroke_width", "stroke_color", "negative_color", "stroke_background", "show_signed_area"],
            "invalid": ["n_rects", "color", "fill_color", "opacity", "sample_type", "riemann_sum_type"],
        },
        "Axes.get_area_under_graph": {
            "valid": ["graph", "x_range", "fill_color", "fill_opacity", "stroke_width", "stroke_color"],
            "invalid": ["color", "opacity", "bounded"],
        },
        "Axes.get_graph": {
            "valid": ["function", "x_range", "color", "stroke_width"],
            "invalid": ["use_smoothing", "discontinuities"],
        },
        "Axes.__init__": {
            "valid": ["x_range", "y_range", "width", "height", "axis_config", "x_axis_config", "y_axis_config"],
            "invalid": ["x_length", "y_length", "tips", "include_numbers", "label_direction"],
        },
        "NumberLine.__init__": {
            "valid": ["x_range", "length", "unit_size", "include_ticks", "tick_size", "big_tick_numbers", "include_tip", "tip_config"],
            "invalid": ["include_numbers", "numbers_with_elongated_ticks", "tip_width", "tip_height", "label_direction"],
        },
        "NumberPlane.__init__": {
            "valid": ["x_range", "y_range", "width", "height", "axis_config", "background_line_style", "faded_line_style", "faded_line_ratio"],
            "invalid": ["x_length", "y_length", "tips", "include_numbers"],
        },

        # === VECTOR FIELDS (REQUIRE coordinate_system!) ===
        "VectorField.__init__": {
            "valid": ["func", "coordinate_system", "density", "magnitude_range", "color",
                      "color_map_name", "color_map", "stroke_opacity", "stroke_width",
                      "tip_width_ratio", "tip_len_to_width", "max_vect_len",
                      "max_vect_len_to_step_size", "flat_stroke", "norm_to_opacity_func"],
            "invalid": [],
            "required": ["func", "coordinate_system"],  # Mark required params
        },
        "StreamLines.__init__": {
            "valid": ["func", "coordinate_system", "density", "n_repeats", "noise_factor",
                      "solution_time", "dt", "arc_len", "max_time_steps", "n_samples_per_line",
                      "cutoff_norm", "stroke_width", "stroke_color", "stroke_opacity",
                      "color_by_magnitude", "magnitude_range", "taper_stroke_width", "color_map"],
            "invalid": [],
            "required": ["func", "coordinate_system"],
        },
        "TimeVaryingVectorField.__init__": {
            "valid": ["time_func", "coordinate_system"],
            "invalid": [],
            "required": ["time_func", "coordinate_system"],
        },

        # === GEOMETRY PRIMITIVES ===
        "Dot.__init__": {
            "valid": ["point", "radius", "stroke_color", "stroke_width", "fill_opacity", "fill_color"],
            "invalid": ["color"],  # Common mistake: use fill_color, not color
        },
        "SmallDot.__init__": {
            "valid": ["point", "radius"],
            "invalid": [],
        },
        "Arrow.__init__": {
            "valid": ["start", "end", "buff", "path_arc", "fill_color", "fill_opacity",
                      "stroke_width", "thickness", "tip_width_ratio", "tip_angle",
                      "max_tip_length_to_length_ratio", "max_width_to_length_ratio"],
            "invalid": ["color"],  # Use fill_color
        },
        "Vector.__init__": {
            "valid": ["direction", "buff"],
            "invalid": [],
        },
        "Line.__init__": {
            "valid": ["start", "end", "stroke_color", "stroke_width", "path_arc", "buff"],
            "invalid": ["color"],  # Use stroke_color
        },
        "Circle.__init__": {
            "valid": ["radius", "arc_center", "start_angle", "angle", "n_components",
                      "stroke_color", "stroke_width", "fill_color", "fill_opacity"],
            "invalid": ["color"],  # Use stroke_color or fill_color
        },
        "Rectangle.__init__": {
            "valid": ["width", "height", "stroke_color", "stroke_width", "fill_color", "fill_opacity"],
            "invalid": ["color"],
        },
        "Square.__init__": {
            "valid": ["side_length", "stroke_color", "stroke_width", "fill_color", "fill_opacity"],
            "invalid": ["color"],
        },

        # === 3D SURFACES ===
        "ParametricSurface.__init__": {
            "valid": ["uv_func", "u_range", "v_range", "resolution", "color", "opacity"],
            "invalid": [],
            "required": ["uv_func"],
        },
        "Surface.__init__": {
            "valid": ["u_range", "v_range", "resolution", "color", "opacity",
                      "gloss", "shadow", "prefered_creation_axis"],
            "invalid": [],
        },
        "Sphere.__init__": {
            "valid": ["radius", "u_range", "v_range", "resolution"],
            "invalid": [],
        },

        # === TEXT & LABELS ===
        "Tex.__init__": {
            "valid": ["tex_string", "color", "font_size", "alignment", "isolate", "tex_to_color_map"],
            "invalid": ["tex_environment"],  # CE-only
        },
        "TexText.__init__": {
            "valid": ["text", "color", "font_size", "alignment"],
            "invalid": [],
        },
        "Text.__init__": {
            "valid": ["text", "font", "font_size", "color", "line_spacing", "slant", "weight"],
            "invalid": ["size"],  # Use font_size
        },

        # === GROUPS & CONTAINERS ===
        "VGroup.__init__": {
            "valid": [],  # Accepts *vmobjects
            "invalid": [],
        },
        "Group.__init__": {
            "valid": [],  # Accepts *mobjects
            "invalid": [],
        },

        # === ARCS & CURVES ===
        "Arc.__init__": {
            "valid": ["start_angle", "angle", "radius", "n_components", "arc_center",
                      "stroke_color", "stroke_width", "fill_color", "fill_opacity"],
            "invalid": ["color"],  # Use stroke_color
        },
        "ArcBetweenPoints.__init__": {
            "valid": ["start", "end", "angle", "stroke_color", "stroke_width"],
            "invalid": [],
        },
        "CurvedArrow.__init__": {
            "valid": ["start_point", "end_point", "angle"],
            "invalid": [],
        },

        # === COORDINATE SYSTEMS (additional) ===
        "ComplexPlane.__init__": {
            # Inherits from NumberPlane
            "valid": ["x_range", "y_range", "width", "height", "axis_config",
                      "background_line_style", "faded_line_style", "faded_line_ratio"],
            "invalid": ["x_length", "y_length"],
        },
        "ThreeDAxes.__init__": {
            "valid": ["x_range", "y_range", "z_range", "width", "height", "depth",
                      "axis_config", "x_axis_config", "y_axis_config", "z_axis_config"],
            "invalid": ["x_length", "y_length", "z_length"],
        },

        # === TRANSFORMS & ANIMATIONS ===
        "Transform.__init__": {
            "valid": ["mobject", "target_mobject", "path_arc", "path_arc_axis",
                      "path_func", "replace_mobject_with_target_in_scene"],
            "invalid": [],
        },
        "FadeIn.__init__": {
            "valid": ["mobject", "shift", "scale", "lag_ratio"],
            "invalid": ["target_position"],  # CE-only
        },
        "FadeOut.__init__": {
            "valid": ["mobject", "shift", "scale", "lag_ratio"],
            "invalid": ["target_position"],
        },
    }


# Pre-extracted error patterns from code_bridge.py knowledge
ERROR_PATTERNS = [
    {
        "pattern": "n_rects",
        "method": "get_riemann_rectangles",
        "fix": "Use dx instead. Calculate: dx = (x_max - x_min) / n_rects",
        "example": "get_riemann_rectangles(graph, dx=0.25)  # instead of n_rects=4",
    },
    {
        "pattern": "fill_color=",
        "method": "get_riemann_rectangles",
        "fix": "Use colors=(COLOR,) instead of fill_color. manimgl uses colors iterable for gradient.",
        "example": "get_riemann_rectangles(graph, colors=(BLUE,))",
    },
    {
        "pattern": "color=",
        "method": "get_riemann_rectangles",
        "fix": "Use colors=(COLOR,) instead of color. manimgl uses colors iterable for gradient.",
        "example": "get_riemann_rectangles(graph, colors=(BLUE,))",
    },
    {
        "pattern": "sample_type=",
        "method": "get_riemann_rectangles",
        "fix": "Use input_sample_type instead of sample_type. Valid values: 'left', 'right', 'center'",
        "example": "get_riemann_rectangles(graph, input_sample_type='left')",
    },
    {
        "pattern": "Create(",
        "method": None,
        "fix": "Use ShowCreation() instead of Create() in manimgl",
        "example": "self.play(ShowCreation(circle))",
    },
    {
        "pattern": "MathTex(",
        "method": None,
        "fix": "Use Tex() instead of MathTex() in manimgl",
        "example": "Tex(r'\\int_a^b f(x) dx')",
    },
    {
        "pattern": "add_coordinates(",
        "method": None,
        "fix": "Use add_coordinate_labels() instead of add_coordinates() in manimgl",
        "example": "axes.add_coordinate_labels()",
    },
    {
        "pattern": "x_length=",
        "method": "Axes",
        "fix": "Use width= instead of x_length= for Axes in manimgl",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2], width=10, height=6)",
    },
    {
        "pattern": "y_length=",
        "method": "Axes",
        "fix": "Use height= instead of y_length= for Axes in manimgl",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2], width=10, height=6)",
    },
    {
        "pattern": "tips=",
        "method": "Axes",
        "fix": "tips= parameter doesn't exist in manimgl Axes. Remove it.",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2])  # no tips parameter",
    },
    {
        "pattern": ".plot(",
        "method": None,
        "fix": "Use .get_graph() instead of .plot() in manimgl",
        "example": "graph = axes.get_graph(lambda x: x**2)",
    },
    {
        "pattern": ".get_area(",
        "method": None,
        "fix": "Use .get_area_under_graph() instead of .get_area() in manimgl",
        "example": "area = axes.get_area_under_graph(graph, x_range=[0, 2])",
    },
    {
        "pattern": "opacity=",
        "method": "get_area_under_graph",
        "fix": "Use fill_opacity= instead of opacity= in manimgl",
        "example": "axes.get_area_under_graph(graph, fill_opacity=0.5)",
    },
    {
        "pattern": "from manim import",
        "method": None,
        "fix": "Use 'from manimlib import *' instead of 'from manim import' for manimgl",
        "example": "from manimlib import *",
    },
    {
        "pattern": ".arrange_submobjects(",
        "method": None,
        "fix": "Use .arrange() instead of .arrange_submobjects() in manimgl",
        "example": "group.arrange(DOWN, buff=0.5)",
    },
    {
        "pattern": "FadeOut(",
        "method": None,
        "fix": "In manimgl, FadeOut() only accepts ONE mobject. For multiple mobjects, wrap them in VGroup: FadeOut(VGroup(a, b, c)) instead of FadeOut(a, b, c). The extra args in manimgl become keyword arguments like shift=, which causes 'operands could not be broadcast' errors.",
        "example": "self.play(FadeOut(VGroup(label1, label2, label3)))  # NOT FadeOut(label1, label2, label3)",
    },
    {
        "pattern": "FadeIn(",
        "method": None,
        "fix": "In manimgl, FadeIn() only accepts ONE mobject. For multiple mobjects, wrap them in VGroup: FadeIn(VGroup(a, b, c)) instead of FadeIn(a, b, c).",
        "example": "self.play(FadeIn(VGroup(obj1, obj2)))  # NOT FadeIn(obj1, obj2)",
    },
    {
        "pattern": "operands could not be broadcast",
        "method": None,
        "fix": "This numpy error often means you passed a mobject where manimgl expected a vector. Common cause: FadeOut(a, b, c) where the extra mobjects get interpreted as shift vectors. Use FadeOut(VGroup(a, b, c)) instead.",
        "example": "self.play(FadeOut(VGroup(mob1, mob2, mob3)))  # Wrap in VGroup",
    },
    # === CE-ONLY ANIMATION ERRORS ===
    {
        "pattern": "Circumscribe",
        "method": None,
        "fix": "Circumscribe is CE-only. In manimgl, use Indicate() or FlashAround() instead.",
        "example": "self.play(Indicate(mobject))  # Instead of Circumscribe",
    },
    {
        "pattern": "SpiralIn",
        "method": None,
        "fix": "SpiralIn is CE-only. In manimgl, use GrowFromCenter() or GrowFromPoint() instead.",
        "example": "self.play(GrowFromCenter(mobject))  # Instead of SpiralIn",
    },
    {
        "pattern": "AddTextLetterByLetter",
        "method": None,
        "fix": "AddTextLetterByLetter is CE-only. In manimgl, use Write() instead.",
        "example": "self.play(Write(text))  # Instead of AddTextLetterByLetter",
    },
    {
        "pattern": "Blink",
        "method": None,
        "fix": "Blink is CE-only. In manimgl, use VFadeIn/VFadeOut or create a custom animation.",
        "example": "self.play(VFadeOut(mob), VFadeIn(mob))  # Approximate blink effect",
    },
    # === IMPORT ERRORS ===
    {
        "pattern": "cannot import name 'Create'",
        "method": None,
        "fix": "manimgl uses ShowCreation instead of Create. Change 'Create' to 'ShowCreation'.",
        "example": "from manimlib import *  # then use ShowCreation(mobject)",
    },
    {
        "pattern": "cannot import name 'MathTex'",
        "method": None,
        "fix": "manimgl uses Tex instead of MathTex. Change 'MathTex' to 'Tex'.",
        "example": "formula = Tex(r'\\int_a^b f(x) dx')",
    },
    # === ATTRIBUTE ERRORS ===
    {
        "pattern": "has no attribute 'plot'",
        "method": None,
        "fix": "manimgl uses get_graph() instead of plot(). Change axes.plot() to axes.get_graph().",
        "example": "graph = axes.get_graph(lambda x: x**2)",
    },
    {
        "pattern": "has no attribute 'add_coordinates'",
        "method": None,
        "fix": "manimgl uses add_coordinate_labels() instead of add_coordinates().",
        "example": "axes.add_coordinate_labels()",
    },
    {
        "pattern": "has no attribute 'get_area'",
        "method": None,
        "fix": "manimgl uses get_area_under_graph() instead of get_area().",
        "example": "area = axes.get_area_under_graph(graph, x_range=[0, 2])",
    },
    # === PARAMETER ERRORS ===
    {
        "pattern": "unexpected keyword argument 'tips'",
        "method": "Axes",
        "fix": "manimgl Axes doesn't have a 'tips' parameter. Remove it.",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2])  # No tips parameter",
    },
    {
        "pattern": "unexpected keyword argument 'x_length'",
        "method": "Axes",
        "fix": "manimgl uses 'width' instead of 'x_length' for Axes.",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2], width=10)",
    },
    {
        "pattern": "unexpected keyword argument 'y_length'",
        "method": "Axes",
        "fix": "manimgl uses 'height' instead of 'y_length' for Axes.",
        "example": "Axes(x_range=[-3, 3], y_range=[-2, 2], height=6)",
    },
    {
        "pattern": "unexpected keyword argument 'target_position'",
        "method": None,
        "fix": "manimgl FadeIn/FadeOut doesn't support 'target_position'. Use 'shift' parameter instead.",
        "example": "FadeIn(mob, shift=UP)  # Instead of target_position",
    },
    # === TYPE ERRORS ===
    {
        "pattern": "takes 1 positional argument but",
        "method": None,
        "fix": "Many manimgl animations only accept ONE mobject. Wrap multiple mobjects in VGroup().",
        "example": "FadeOut(VGroup(mob1, mob2))  # NOT FadeOut(mob1, mob2)",
    },
    {
        "pattern": "'Mobject' object is not subscriptable",
        "method": None,
        "fix": "You're trying to index a Mobject directly. Use .submobjects[i] or store submobjects in a list.",
        "example": "parts = VGroup(*[...])  # Then access parts[0], parts[1], etc.",
    },
    # === RUNTIME ERRORS ===
    {
        "pattern": "GREY",
        "method": None,
        "fix": "manimgl uses British spelling GREY, not American GRAY.",
        "example": "circle.set_color(GREY)  # Not GRAY",
    },
    {
        "pattern": "DEGREES",
        "method": None,
        "fix": "manimgl uses DEG, not DEGREES for angle units.",
        "example": "angle = 45 * DEG  # Not 45 * DEGREES",
    },
    {
        "pattern": "ThreeDScene",
        "method": None,
        "fix": "manimgl doesn't have ThreeDScene. Use regular Scene - 3D is automatic with 3D objects.",
        "example": "class MyScene(Scene):  # Not ThreeDScene",
    },
]


def get_error_patterns() -> list[dict]:
    """Return pre-defined error patterns for indexing."""
    return ERROR_PATTERNS
