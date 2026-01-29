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

    Returns:
        Dict mapping "Class.method" -> {"valid": [params], "invalid": [params]}
    """
    return {
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
]


def get_error_patterns() -> list[dict]:
    """Return pre-defined error patterns for indexing."""
    return ERROR_PATTERNS
