"""Pre-generation Linter: Validates and auto-fixes generated code before rendering.

Runs ruff (or falls back to ast.parse) on generated code BEFORE rendering to catch
common issues like undefined names, unused variables, and syntax errors.

Supports two modes:
1. Ruff-based analysis: Fast, comprehensive linting when ruff is available
2. Pure Python fallback: Uses ast module when ruff is not installed
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


# Check if ruff is available at module load time
RUFF_AVAILABLE = shutil.which("ruff") is not None


class LintIssue(NamedTuple):
    """A single lint issue."""

    code: str  # e.g., "F841", "E999", "F823"
    line: int
    column: int
    message: str


@dataclass
class LintResult:
    """Result of linting code."""

    original_code: str
    fixed_code: str
    issues: list[LintIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_fixed: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any unfixed errors."""
        return len(self.warnings) > 0

    @property
    def error_count(self) -> int:
        """Count of unfixed errors."""
        return len(self.warnings)


# Common built-in names that should not be flagged as undefined
BUILTIN_NAMES = {
    # Python builtins
    "True", "False", "None", "print", "range", "len", "int", "float", "str",
    "list", "dict", "set", "tuple", "bool", "type", "isinstance", "issubclass",
    "hasattr", "getattr", "setattr", "delattr", "property", "staticmethod",
    "classmethod", "super", "object", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "AttributeError", "RuntimeError", "StopIteration",
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "min", "max",
    "sum", "abs", "round", "pow", "divmod", "hex", "oct", "bin", "chr", "ord",
    "repr", "hash", "id", "input", "open", "file", "iter", "next", "callable",
    "any", "all", "format", "vars", "dir", "globals", "locals", "exec", "eval",
    "compile", "complex", "bytes", "bytearray", "memoryview", "frozenset",
    "slice", "ascii", "breakpoint", "NotImplemented", "Ellipsis",
    # Common imports assumed available in Manim context
    "np", "numpy", "math", "random", "os", "sys", "re", "json", "time",
    "itertools", "functools", "collections", "copy", "typing",
    # Manim/manimgl globals (commonly used without explicit import)
    "Scene", "VGroup", "VMobject", "Mobject", "Group", "Point",
    "Circle", "Square", "Rectangle", "Triangle", "Polygon", "Line", "Arrow",
    "Dot", "Arc", "Ellipse", "Annulus", "AnnularSector", "Sector",
    "Tex", "TexText", "Text", "MathTex", "DecimalNumber", "Integer",
    "NumberLine", "NumberPlane", "Axes", "ThreeDAxes", "ComplexPlane",
    "ParametricCurve", "FunctionGraph", "ImplicitFunction",
    "Surface", "ParametricSurface", "Sphere", "Cube", "Cylinder", "Cone",
    "Prism", "Torus", "Arrow3D", "Line3D", "Dot3D",
    "Write", "FadeIn", "FadeOut", "GrowFromCenter", "ShowCreation", "Create",
    "Transform", "ReplacementTransform", "MoveToTarget", "ApplyMethod",
    "Indicate", "Circumscribe", "Flash", "ShowPassingFlash", "AnimationGroup",
    "Succession", "LaggedStart", "LaggedStartMap", "Rotate", "Rotating",
    "MoveAlongPath", "Homotopy", "ComplexHomotopy", "PhaseFlow",
    "Wait", "Uncreate", "Unwrite", "GrowArrow", "SpinInFromNothing",
    "DrawBorderThenFill", "ShowIncreasingSubsets", "ShowSubmobjectsOneByOne",
    "ApplyWave", "WiggleOutThenIn", "TurnInsideOut", "UpdateFromFunc",
    "UpdateFromAlphaFunc", "Restore", "ApplyFunction", "ClockwiseTransform",
    "CounterclockwiseTransform", "TransformFromCopy", "FadeTransform",
    "ShrinkToCenter", "GrowFromPoint", "GrowFromEdge",
    # Common direction/color constants
    "UP", "DOWN", "LEFT", "RIGHT", "UL", "UR", "DL", "DR", "ORIGIN",
    "OUT", "IN", "TAU", "PI", "DEGREES", "RADIANS",
    "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PURPLE",
    "PINK", "TEAL", "MAROON", "GOLD", "GREY", "GRAY",
    "RED_A", "RED_B", "RED_C", "RED_D", "RED_E",
    "GREEN_A", "GREEN_B", "GREEN_C", "GREEN_D", "GREEN_E",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E",
    "YELLOW_A", "YELLOW_B", "YELLOW_C", "YELLOW_D", "YELLOW_E",
    "TEAL_A", "TEAL_B", "TEAL_C", "TEAL_D", "TEAL_E",
    "PURPLE_A", "PURPLE_B", "PURPLE_C", "PURPLE_D", "PURPLE_E",
    # Rate functions
    "linear", "smooth", "rush_into", "rush_from", "slow_into", "double_smooth",
    "there_and_back", "there_and_back_with_pause", "running_start", "not_quite_there",
    "wiggle", "squish_rate_func", "lingering", "exponential_decay",
    # Config
    "config", "FRAME_HEIGHT", "FRAME_WIDTH", "FRAME_X_RADIUS", "FRAME_Y_RADIUS",
    # Other common manimgl globals
    "TexTemplate", "TexTemplateLibrary", "ManimColor", "color_gradient",
    "interpolate", "interpolate_color", "invert_color", "average_color",
    "get_norm", "normalize", "cross", "get_unit_normal",
    "angle_of_vector", "rotation_matrix", "z_to_vector",
    "VectorizedPoint", "DashedLine", "DashedVMobject", "TangentLine",
    "Brace", "BraceLabel", "BraceBetweenPoints",
    "SurroundingRectangle", "BackgroundRectangle", "Cross", "Exmark", "Checkmark",
    "Matrix", "IntegerMatrix", "DecimalMatrix", "MobjectMatrix",
    "BarChart", "PieChart", "Graph", "DiGraph",
    "Table", "MathTable", "IntegerTable", "DecimalTable",
    "TracedPath", "always_redraw", "always_shift", "always_rotate", "f_always",
    "ValueTracker", "ComplexValueTracker",
    "self",  # self is always valid inside methods
}


def check_syntax(code: str) -> list[str]:
    """Check for syntax errors using ast.parse.

    Args:
        code: Python code to check

    Returns:
        List of error messages (empty if no syntax errors)
    """
    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        line_info = f"line {e.lineno}" if e.lineno else "unknown line"
        col_info = f", column {e.offset}" if e.offset else ""
        errors.append(f"E999 Syntax error at {line_info}{col_info}: {e.msg}")
    return errors


def check_undefined_names(code: str) -> list[str]:
    """Check for undefined variable names (F823).

    Uses AST analysis to find names that are used but never defined.

    Args:
        code: Python code to check

    Returns:
        List of warning messages for undefined names
    """
    warnings = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't analyze code with syntax errors
        return []

    # Collect all defined names
    defined_names: set[str] = set(BUILTIN_NAMES)

    class DefinitionCollector(ast.NodeVisitor):
        """Collect all defined names in the code."""

        def __init__(self) -> None:
            self.scope_stack: list[set[str]] = [set()]

        def current_scope(self) -> set[str]:
            return self.scope_stack[-1]

        def all_defined(self) -> set[str]:
            result = set()
            for scope in self.scope_stack:
                result.update(scope)
            return result

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.current_scope().add(node.name)
            # Add function arguments to a new scope
            self.scope_stack.append(set())
            for arg in node.args.args:
                self.current_scope().add(arg.arg)
            for arg in node.args.posonlyargs:
                self.current_scope().add(arg.arg)
            for arg in node.args.kwonlyargs:
                self.current_scope().add(arg.arg)
            if node.args.vararg:
                self.current_scope().add(node.args.vararg.arg)
            if node.args.kwarg:
                self.current_scope().add(node.args.kwarg.arg)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.current_scope().add(node.name)
            self.scope_stack.append(set())
            for arg in node.args.args:
                self.current_scope().add(arg.arg)
            for arg in node.args.posonlyargs:
                self.current_scope().add(arg.arg)
            for arg in node.args.kwonlyargs:
                self.current_scope().add(arg.arg)
            if node.args.vararg:
                self.current_scope().add(node.args.vararg.arg)
            if node.args.kwarg:
                self.current_scope().add(node.args.kwarg.arg)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.current_scope().add(node.name)
            self.scope_stack.append(set())
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                self.current_scope().add(node.id)
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                self.current_scope().add(name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name == "*":
                    # Star imports - we can't know what they import
                    pass
                else:
                    name = alias.asname if alias.asname else alias.name
                    self.current_scope().add(name)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                self.current_scope().add(node.name)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            self._handle_assignment_target(node.target)
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                if item.optional_vars:
                    self._handle_assignment_target(item.optional_vars)
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            self._handle_assignment_target(node.target)
            self.generic_visit(node)

        def _handle_assignment_target(self, node: ast.expr) -> None:
            """Handle assignment targets (Name, Tuple, List)."""
            if isinstance(node, ast.Name):
                self.current_scope().add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    self._handle_assignment_target(elt)
            elif isinstance(node, ast.Starred):
                self._handle_assignment_target(node.value)

    collector = DefinitionCollector()
    collector.visit(tree)
    defined_names.update(collector.all_defined())

    # Now find all used names that aren't defined
    class UsageChecker(ast.NodeVisitor):
        """Check for undefined name usage."""

        def __init__(self) -> None:
            self.undefined: list[tuple[str, int, int]] = []
            self.local_scopes: list[set[str]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Create new scope with function arguments
            local_scope = set()
            for arg in node.args.args:
                local_scope.add(arg.arg)
            for arg in node.args.posonlyargs:
                local_scope.add(arg.arg)
            for arg in node.args.kwonlyargs:
                local_scope.add(arg.arg)
            if node.args.vararg:
                local_scope.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_scope.add(node.args.kwarg.arg)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            local_scope = set()
            for arg in node.args.args:
                local_scope.add(arg.arg)
            for arg in node.args.posonlyargs:
                local_scope.add(arg.arg)
            for arg in node.args.kwonlyargs:
                local_scope.add(arg.arg)
            if node.args.vararg:
                local_scope.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_scope.add(node.args.kwarg.arg)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.local_scopes.append(set())
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_ListComp(self, node: ast.ListComp) -> None:
            # Comprehension creates a new scope
            local_scope = set()
            for gen in node.generators:
                self._add_comp_targets(gen.target, local_scope)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_SetComp(self, node: ast.SetComp) -> None:
            local_scope = set()
            for gen in node.generators:
                self._add_comp_targets(gen.target, local_scope)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_DictComp(self, node: ast.DictComp) -> None:
            local_scope = set()
            for gen in node.generators:
                self._add_comp_targets(gen.target, local_scope)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            local_scope = set()
            for gen in node.generators:
                self._add_comp_targets(gen.target, local_scope)
            self.local_scopes.append(local_scope)
            self.generic_visit(node)
            self.local_scopes.pop()

        def _add_comp_targets(self, node: ast.expr, scope: set[str]) -> None:
            if isinstance(node, ast.Name):
                scope.add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    self._add_comp_targets(elt, scope)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                name = node.id
                # Check if defined in any local scope
                for scope in self.local_scopes:
                    if name in scope:
                        return
                # Check global defined names
                if name not in defined_names:
                    self.undefined.append((name, node.lineno, node.col_offset))

        def visit_Assign(self, node: ast.Assign) -> None:
            # Visit value first, then targets
            self.visit(node.value)
            for target in node.targets:
                self._add_to_local_scope(target)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value:
                self.visit(node.value)
            self._add_to_local_scope(node.target)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.value)
            # AugAssign requires the name to already exist
            self.visit(node.target)

        def visit_For(self, node: ast.For) -> None:
            # Visit iter first
            self.visit(node.iter)
            # Add target to scope
            self._add_to_local_scope(node.target)
            # Visit body
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)

        def _add_to_local_scope(self, node: ast.expr) -> None:
            """Add assignment targets to local scope."""
            if not self.local_scopes:
                return
            if isinstance(node, ast.Name):
                self.local_scopes[-1].add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    self._add_to_local_scope(elt)
            elif isinstance(node, ast.Starred):
                self._add_to_local_scope(node.value)

    checker = UsageChecker()
    checker.visit(tree)

    # Generate warnings for undefined names
    seen = set()
    for name, line, col in checker.undefined:
        if name not in seen:
            warnings.append(f"F823 Undefined name '{name}' at line {line}")
            seen.add(name)

    return warnings


def check_unused_variables(code: str) -> list[str]:
    """Check for unused variables (F841).

    Finds variables that are assigned but never used.

    Args:
        code: Python code to check

    Returns:
        List of warning messages for unused variables
    """
    warnings = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    class UnusedVariableChecker(ast.NodeVisitor):
        """Check for unused variable assignments."""

        def __init__(self) -> None:
            self.assigned: dict[str, tuple[int, int]] = {}  # name -> (line, col)
            self.used: set[str] = set()
            self.scope_stack: list[dict[str, tuple[int, int]]] = []
            self.used_stack: list[set[str]] = []

        def push_scope(self) -> None:
            self.scope_stack.append({})
            self.used_stack.append(set())

        def pop_scope(self) -> list[tuple[str, int, int]]:
            """Pop scope and return unused variables from that scope."""
            assigned = self.scope_stack.pop()
            used = self.used_stack.pop()
            unused = []
            for name, (line, col) in assigned.items():
                if name not in used and not name.startswith("_"):
                    unused.append((name, line, col))
            return unused

        def assign(self, name: str, line: int, col: int) -> None:
            if self.scope_stack:
                self.scope_stack[-1][name] = (line, col)
            else:
                self.assigned[name] = (line, col)

        def use(self, name: str) -> None:
            if self.used_stack:
                self.used_stack[-1].add(name)
            else:
                self.used.add(name)
            # Also mark as used in all parent scopes
            for used_set in self.used_stack:
                used_set.add(name)
            self.used.add(name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Function name itself is assigned
            self.assign(node.name, node.lineno, node.col_offset)
            self.push_scope()
            self.generic_visit(node)
            unused = self.pop_scope()
            for name, line, col in unused:
                warnings.append(f"F841 Local variable '{name}' assigned but never used at line {line}")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.assign(node.name, node.lineno, node.col_offset)
            self.push_scope()
            self.generic_visit(node)
            unused = self.pop_scope()
            for name, line, col in unused:
                warnings.append(f"F841 Local variable '{name}' assigned but never used at line {line}")

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.assign(node.name, node.lineno, node.col_offset)
            self.push_scope()
            self.generic_visit(node)
            self.pop_scope()  # Don't report unused in classes (could be used externally)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                self.assign(node.id, node.lineno, node.col_offset)
            elif isinstance(node.ctx, ast.Load):
                self.use(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # self.x is a use of self
            self.generic_visit(node)

    checker = UnusedVariableChecker()
    checker.visit(tree)

    # Check top-level unused (but don't warn for these - they might be module exports)
    # Only warn about unused in functions

    return warnings


async def _run_ruff(code: str) -> tuple[str, list[LintIssue]]:
    """Run ruff on code and return fixed code and issues.

    Args:
        code: Python code to lint

    Returns:
        Tuple of (fixed_code, list of issues)
    """
    issues: list[LintIssue] = []

    # Create temp file for ruff
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Run ruff check first to get issues
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", "--output-format=json",
            "--select=E999,F823,F841,F401,F811,E501,E711,E712",
            str(temp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        # Parse ruff JSON output
        if stdout:
            import json
            try:
                ruff_issues = json.loads(stdout.decode())
                for issue in ruff_issues:
                    issues.append(LintIssue(
                        code=issue.get("code", ""),
                        line=issue.get("location", {}).get("row", 0),
                        column=issue.get("location", {}).get("column", 0),
                        message=issue.get("message", ""),
                    ))
            except json.JSONDecodeError:
                logger.warning("Failed to parse ruff output: %s", stdout.decode()[:200])

        # Run ruff fix to auto-fix what we can
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", "--fix", "--unsafe-fixes",
            "--select=F401,F841,E711,E712",  # Only auto-fix safe issues
            str(temp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await proc.communicate()

        # Read fixed code
        fixed_code = temp_path.read_text()

    finally:
        temp_path.unlink(missing_ok=True)

    return fixed_code, issues


def _auto_fix_unused_imports(code: str) -> tuple[str, list[str]]:
    """Remove unused imports from code.

    Args:
        code: Python code

    Returns:
        Tuple of (fixed_code, list of fixes applied)
    """
    fixes = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    # Find all imported names
    imported: dict[str, tuple[int, str]] = {}  # name -> (line, full_line)
    lines = code.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                imported[name] = (node.lineno, lines[node.lineno - 1] if node.lineno <= len(lines) else "")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname if alias.asname else alias.name
                    imported[name] = (node.lineno, lines[node.lineno - 1] if node.lineno <= len(lines) else "")

    # Find all used names
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For x.y.z, we need to find the root name
            current = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                used.add(current.id)

    # Find unused imports
    unused = set(imported.keys()) - used

    # Remove unused imports (simple approach - remove entire lines)
    lines_to_remove = set()
    for name in unused:
        if name in imported:
            line_num, line_content = imported[name]
            # Only remove if it's a simple single import
            if f"import {name}" in line_content or f"from " in line_content:
                # Check if this is the only import on the line
                if line_content.strip().startswith(f"import {name}") and "," not in line_content:
                    lines_to_remove.add(line_num - 1)
                    fixes.append(f"Removed unused import '{name}'")
                elif f"from " in line_content and "," not in line_content:
                    lines_to_remove.add(line_num - 1)
                    fixes.append(f"Removed unused import '{name}'")

    if lines_to_remove:
        new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
        code = "\n".join(new_lines)

    return code, fixes


def _auto_fix_comparison_to_none(code: str) -> tuple[str, list[str]]:
    """Fix comparisons like 'x == None' to 'x is None'.

    Args:
        code: Python code

    Returns:
        Tuple of (fixed_code, list of fixes applied)
    """
    fixes = []

    # Pattern: x == None -> x is None
    pattern_eq = r'(\w+)\s*==\s*None'
    matches = list(re.finditer(pattern_eq, code))
    if matches:
        code = re.sub(pattern_eq, r'\1 is None', code)
        fixes.append(f"Fixed {len(matches)} '== None' comparisons to 'is None'")

    # Pattern: x != None -> x is not None
    pattern_neq = r'(\w+)\s*!=\s*None'
    matches = list(re.finditer(pattern_neq, code))
    if matches:
        code = re.sub(pattern_neq, r'\1 is not None', code)
        fixes.append(f"Fixed {len(matches)} '!= None' comparisons to 'is not None'")

    return code, fixes


async def lint_code(code: str) -> tuple[str, list[str]]:
    """Lint and auto-fix code.

    Runs ruff if available, otherwise uses pure Python AST analysis.
    Automatically fixes issues where possible.

    Args:
        code: Python code to lint

    Returns:
        Tuple of (fixed_code, list of warnings for unfixed issues)
    """
    if not code or not code.strip():
        return code, []

    warnings: list[str] = []
    auto_fixed: list[str] = []
    fixed_code = code

    # First check for syntax errors - these prevent all other analysis
    syntax_errors = check_syntax(code)
    if syntax_errors:
        # Can't auto-fix syntax errors, return immediately
        return code, syntax_errors

    if RUFF_AVAILABLE:
        # Use ruff for comprehensive linting
        try:
            fixed_code, issues = await _run_ruff(code)

            # Track which issues were fixed by comparing
            if fixed_code != code:
                auto_fixed.append("Applied ruff auto-fixes")

            # Generate warnings for unfixed issues
            for issue in issues:
                # Check if this issue might have been fixed
                if issue.code in ("F401",):  # These are typically auto-fixed
                    continue
                warnings.append(f"{issue.code} {issue.message} at line {issue.line}")

        except Exception as e:
            logger.warning("Ruff failed, falling back to AST analysis: %s", e)
            # Fall through to AST-based analysis
            fixed_code = code
    else:
        # Pure Python fallback
        logger.debug("Ruff not available, using AST-based linting")

    # Apply Python-based auto-fixes regardless of ruff
    # These catch things ruff might miss or provide fallback

    # Fix comparison to None
    fixed_code, fixes = _auto_fix_comparison_to_none(fixed_code)
    auto_fixed.extend(fixes)

    # Remove unused imports (if ruff didn't handle it)
    if not RUFF_AVAILABLE:
        fixed_code, fixes = _auto_fix_unused_imports(fixed_code)
        auto_fixed.extend(fixes)

    # Run pure Python checks on the fixed code
    # Check for undefined names
    undefined_warnings = check_undefined_names(fixed_code)
    warnings.extend(undefined_warnings)

    # Check for unused variables (informational, don't auto-fix)
    unused_warnings = check_unused_variables(fixed_code)
    warnings.extend(unused_warnings)

    if auto_fixed:
        logger.info("[LINTER] Auto-fixed: %s", "; ".join(auto_fixed[:3]))
    if warnings:
        logger.warning("[LINTER] Warnings: %s", "; ".join(warnings[:3]))

    return fixed_code, warnings


async def lint_and_reject(code: str, max_errors: int = 3) -> tuple[str, bool]:
    """Lint code and determine if it should be regenerated.

    Args:
        code: Python code to lint
        max_errors: Maximum number of errors before suggesting regeneration

    Returns:
        Tuple of (possibly_fixed_code, should_regenerate)
        If should_regenerate is True, the code has too many issues to fix.
    """
    if not code or not code.strip():
        return code, False

    # Check for syntax errors first
    syntax_errors = check_syntax(code)
    if syntax_errors:
        # Syntax errors are fatal - definitely regenerate
        logger.warning("[LINTER] Syntax errors found, suggesting regeneration: %s", syntax_errors[0])
        return code, True

    # Run full linting
    fixed_code, warnings = await lint_code(code)

    # Count serious errors (not just style warnings)
    serious_errors = [w for w in warnings if any(
        w.startswith(prefix) for prefix in ("F823", "E999", "F821")
    )]

    should_regenerate = len(serious_errors) >= max_errors

    if should_regenerate:
        logger.warning(
            "[LINTER] Too many errors (%d >= %d), suggesting regeneration",
            len(serious_errors), max_errors
        )

    return fixed_code, should_regenerate


# Convenience function for synchronous usage
def lint_code_sync(code: str) -> tuple[str, list[str]]:
    """Synchronous version of lint_code.

    Args:
        code: Python code to lint

    Returns:
        Tuple of (fixed_code, list of warnings)
    """
    return asyncio.run(lint_code(code))


def lint_and_reject_sync(code: str, max_errors: int = 3) -> tuple[str, bool]:
    """Synchronous version of lint_and_reject.

    Args:
        code: Python code to lint
        max_errors: Maximum number of errors before suggesting regeneration

    Returns:
        Tuple of (possibly_fixed_code, should_regenerate)
    """
    return asyncio.run(lint_and_reject(code, max_errors))
