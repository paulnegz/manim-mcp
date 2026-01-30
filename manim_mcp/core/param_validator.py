"""Parameter Validator: Validates and auto-fixes manimgl API parameters.

This module validates generated code against known manimgl API signatures
and automatically fixes common parameter errors (like using CE parameters
that don't exist in manimgl).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found in the code."""

    method: str
    param: str
    issue_type: str  # "invalid_param", "wrong_param_name", "missing_required"
    message: str
    fix: str | None = None
    line_number: int | None = None


@dataclass
class ValidationResult:
    """Result of validating code against API signatures."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    fixed_code: str | None = None


# Known parameter mappings: CE param -> manimgl param
PARAM_FIXES = {
    # get_riemann_rectangles
    ("get_riemann_rectangles", "n_rects"): {
        "action": "remove",
        "message": "n_rects doesn't exist in manimgl. Use dx instead.",
        "comment": "# TODO: Replace n_rects with dx calculation",
    },
    ("get_riemann_rectangles", "color"): {
        "action": "remove",
        "message": "Use colors=(COLOR,) tuple instead of color= for manimgl get_riemann_rectangles.",
    },
    ("get_riemann_rectangles", "fill_color"): {
        "action": "remove",
        "message": "Use colors=(COLOR,) tuple instead of fill_color= for manimgl get_riemann_rectangles.",
    },
    ("get_riemann_rectangles", "opacity"): {
        "action": "rename",
        "new_name": "fill_opacity",
        "message": "Use fill_opacity instead of opacity in manimgl.",
    },
    ("get_riemann_rectangles", "sample_type"): {
        "action": "rename",
        "new_name": "input_sample_type",
        "message": "Use input_sample_type instead of sample_type in manimgl.",
    },
    ("get_riemann_rectangles", "riemann_sum_type"): {
        "action": "remove",
        "message": "riemann_sum_type doesn't exist in manimgl. Use input_sample_type.",
    },
    # get_area_under_graph
    ("get_area_under_graph", "color"): {
        "action": "rename",
        "new_name": "fill_color",
        "message": "Use fill_color instead of color in manimgl.",
    },
    ("get_area_under_graph", "opacity"): {
        "action": "rename",
        "new_name": "fill_opacity",
        "message": "Use fill_opacity instead of opacity in manimgl.",
    },
    # Axes
    ("Axes", "x_length"): {
        "action": "rename",
        "new_name": "width",
        "message": "Use width instead of x_length in manimgl.",
    },
    ("Axes", "y_length"): {
        "action": "rename",
        "new_name": "height",
        "message": "Use height instead of y_length in manimgl.",
    },
    ("Axes", "tips"): {
        "action": "remove",
        "message": "tips parameter doesn't exist in manimgl Axes.",
    },
    ("Axes", "include_numbers"): {
        "action": "remove",
        "message": "include_numbers doesn't exist in manimgl. Use add_coordinate_labels() method.",
    },
    # NumberPlane
    ("NumberPlane", "x_length"): {
        "action": "rename",
        "new_name": "width",
        "message": "Use width instead of x_length in manimgl.",
    },
    ("NumberPlane", "y_length"): {
        "action": "rename",
        "new_name": "height",
        "message": "Use height instead of y_length in manimgl.",
    },
    ("NumberPlane", "tips"): {
        "action": "remove",
        "message": "tips parameter doesn't exist in manimgl NumberPlane.",
    },
    # NumberLine
    ("NumberLine", "include_numbers"): {
        "action": "remove",
        "message": "include_numbers doesn't exist in manimgl. Use add_numbers() method.",
    },
    ("NumberLine", "numbers_with_elongated_ticks"): {
        "action": "rename",
        "new_name": "big_tick_numbers",
        "message": "Use big_tick_numbers instead of numbers_with_elongated_ticks in manimgl.",
    },
}


class ParameterValidator:
    """Validates and auto-fixes code against manimgl API signatures."""

    def __init__(self, rag: ChromaDBService | None = None):
        """Initialize validator with optional RAG for dynamic lookups."""
        self.rag = rag

    async def validate(self, code: str) -> ValidationResult:
        """Validate code against known API signatures.

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with issues and optionally fixed code
        """
        issues: list[ValidationIssue] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    method="<syntax>",
                    param="",
                    issue_type="syntax_error",
                    message=f"Syntax error: {e}",
                )],
            )

        # Find all function calls and check their parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method_name = self._get_call_name(node)
                if not method_name:
                    continue

                # Check each keyword argument
                for kw in node.keywords:
                    if kw.arg is None:
                        continue  # **kwargs

                    fix_key = (method_name, kw.arg)
                    if fix_key in PARAM_FIXES:
                        fix_info = PARAM_FIXES[fix_key]
                        issues.append(ValidationIssue(
                            method=method_name,
                            param=kw.arg,
                            issue_type="invalid_param" if fix_info["action"] == "remove" else "wrong_param_name",
                            message=fix_info["message"],
                            fix=fix_info.get("new_name"),
                            line_number=node.lineno,
                        ))

        # If issues found, attempt to fix
        fixed_code = None
        if issues:
            fixed_code = self._auto_fix(code, issues)

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            fixed_code=fixed_code,
        )

    def validate_sync(self, code: str) -> ValidationResult:
        """Synchronous version of validate for use in non-async contexts."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.validate(code))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.validate(code))

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the function/method name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _auto_fix(self, code: str, issues: list[ValidationIssue]) -> str:
        """Attempt to automatically fix the issues in the code.

        Uses regex-based replacement to handle the fixes safely.
        """
        fixed = code

        for issue in issues:
            if issue.issue_type == "invalid_param":
                # Remove the parameter
                # Pattern: method_name(..., param=value, ...) or method_name(param=value, ...)
                patterns = [
                    # param=value, (with trailing comma)
                    rf",\s*{re.escape(issue.param)}\s*=\s*[^,)]+(?=\s*[,)])",
                    # param=value at start (with comma after)
                    rf"{re.escape(issue.param)}\s*=\s*[^,)]+\s*,\s*",
                    # param=value as only arg or last arg
                    rf",?\s*{re.escape(issue.param)}\s*=\s*[^,)]+",
                ]
                for pattern in patterns:
                    if re.search(pattern, fixed):
                        fixed = re.sub(pattern, "", fixed)
                        logger.info(
                            "[PARAM-VALIDATOR] Removed invalid param %s from %s",
                            issue.param, issue.method
                        )
                        break

            elif issue.issue_type == "wrong_param_name" and issue.fix:
                # Rename the parameter
                pattern = rf"\b{re.escape(issue.param)}\s*="
                replacement = f"{issue.fix}="
                if re.search(pattern, fixed):
                    fixed = re.sub(pattern, replacement, fixed)
                    logger.info(
                        "[PARAM-VALIDATOR] Renamed %s -> %s in %s",
                        issue.param, issue.fix, issue.method
                    )

        return fixed


def validate_and_fix(code: str, rag: ChromaDBService | None = None) -> str:
    """Validate code and return fixed version.

    Convenience function for use in code_bridge.py.

    Args:
        code: Python code to validate
        rag: Optional RAG service for dynamic lookups

    Returns:
        Fixed code (or original if no issues)
    """
    validator = ParameterValidator(rag)

    try:
        result = validator.validate_sync(code)
    except Exception as e:
        logger.warning("[PARAM-VALIDATOR] Validation failed: %s", e)
        return code

    if result.issues:
        logger.info(
            "[PARAM-VALIDATOR] Found %d issues, applying fixes",
            len(result.issues)
        )
        for issue in result.issues:
            logger.debug(
                "[PARAM-VALIDATOR] %s.%s: %s",
                issue.method, issue.param, issue.message
            )

        if result.fixed_code:
            return result.fixed_code

    return code
