"""Parameter Validator: Validates and auto-fixes manimgl API parameters.

This module validates generated code against known manimgl API signatures
and automatically fixes common parameter errors (like using CE parameters
that don't exist in manimgl).

Supports two modes:
1. Dynamic validation: Query ChromaDB manim_api collection for real API signatures
2. Fallback validation: Use hardcoded PARAM_FIXES rules when RAG unavailable
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


# Cache for API signature lookups (method_name -> APIInfo)
# Using module-level dict for async compatibility (lru_cache doesn't work with async)
_api_cache: dict[str, "APIInfo | None"] = {}


@dataclass
class APIInfo:
    """Cached API information from ChromaDB."""

    full_name: str
    valid_params: set[str]
    invalid_params: set[str]
    is_verified: bool = False

    @classmethod
    def from_chromadb_result(cls, result: dict) -> "APIInfo":
        """Create APIInfo from ChromaDB query result."""
        metadata = result.get("metadata", {})
        content = result.get("content", "")

        # Get method name
        full_name = metadata.get("id", metadata.get("name", ""))

        # Parse valid params from metadata
        valid_params_str = metadata.get("valid_params", "")
        valid_params = set(valid_params_str.split(",")) if valid_params_str else set()

        # Also parse from parameter_names if available (from AST extraction)
        param_names_str = metadata.get("parameter_names", "")
        if param_names_str:
            valid_params.update(param_names_str.split(","))

        # Parse invalid params (CE-only params)
        invalid_params_str = metadata.get("invalid_params", "")
        invalid_params = set(invalid_params_str.split(",")) if invalid_params_str else set()

        # Check if this is a verified/known method
        is_verified = metadata.get("is_verified", False)

        # Remove empty strings
        valid_params.discard("")
        invalid_params.discard("")

        return cls(
            full_name=full_name,
            valid_params=valid_params,
            invalid_params=invalid_params,
            is_verified=is_verified,
        )


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
    # VMobject.set_stroke - common CE/manimgl parameter confusion
    ("set_stroke", "background"): {
        "action": "rename",
        "new_name": "behind",
        "message": "Use 'behind=True' instead of 'background=True' in manimgl set_stroke().",
    },
    # Also add for explicit class context
    ("VMobject.set_stroke", "background"): {
        "action": "rename",
        "new_name": "behind",
        "message": "Use 'behind=True' instead of 'background=True' in manimgl set_stroke().",
    },
    # Text/Tex stroke behind
    ("Text", "stroke_behind"): {
        "action": "rename",
        "new_name": "behind",
        "message": "Use 'behind=True' via set_stroke() instead of stroke_behind.",
    },
}


class ParameterValidator:
    """Validates and auto-fixes code against manimgl API signatures.

    Uses a two-tier approach:
    1. Query ChromaDB for dynamic API signatures (1,652+ indexed methods)
    2. Fall back to hardcoded PARAM_FIXES for critical methods
    """

    def __init__(self, rag: ChromaDBService | None = None):
        """Initialize validator with optional RAG for dynamic lookups.

        Args:
            rag: ChromaDBService instance for querying API signatures.
                 If None, falls back to hardcoded PARAM_FIXES only.
        """
        self.rag = rag
        self._use_dynamic = rag is not None and rag.available
        if self._use_dynamic:
            logger.info("[PARAM-VALIDATOR] Dynamic validation enabled (RAG available)")
        else:
            logger.debug("[PARAM-VALIDATOR] Using hardcoded rules only (RAG unavailable)")

    async def _get_api_info(self, method_name: str, class_context: str | None = None) -> APIInfo | None:
        """Get API information from cache or ChromaDB.

        Args:
            method_name: Name of the method (e.g., 'get_riemann_rectangles')
            class_context: Optional class name context for better matching

        Returns:
            APIInfo if found, None otherwise
        """
        # Build lookup key
        if class_context:
            full_name = f"{class_context}.{method_name}"
        else:
            full_name = method_name

        # Check cache first
        if full_name in _api_cache:
            return _api_cache[full_name]

        # Also check just method name for caching
        if method_name in _api_cache and not class_context:
            return _api_cache[method_name]

        if not self._use_dynamic or not self.rag:
            return None

        try:
            # First try exact lookup by full name
            if class_context:
                result = await self.rag.get_api_signature(full_name)
                if result:
                    api_info = APIInfo.from_chromadb_result(result)
                    _api_cache[full_name] = api_info
                    logger.debug("[PARAM-VALIDATOR] Found API: %s (%d valid params)",
                                full_name, len(api_info.valid_params))
                    return api_info

            # Search for method name
            results = await self.rag.search_api_signatures(method_name, n_results=3)
            for result in results:
                metadata = result.get("metadata", {})
                result_name = metadata.get("name", "")
                result_class = metadata.get("class_name", "")

                # Match method name, optionally with class context
                if result_name == method_name:
                    if not class_context or result_class == class_context:
                        api_info = APIInfo.from_chromadb_result(result)
                        cache_key = f"{result_class}.{result_name}" if result_class else result_name
                        _api_cache[cache_key] = api_info
                        _api_cache[method_name] = api_info  # Also cache by method name only
                        logger.debug("[PARAM-VALIDATOR] Found API via search: %s (%d valid params)",
                                    cache_key, len(api_info.valid_params))
                        return api_info

            # Cache miss - remember we didn't find it
            _api_cache[full_name] = None
            return None

        except Exception as e:
            logger.warning("[PARAM-VALIDATOR] API lookup failed for %s: %s", method_name, e)
            return None

    async def validate(self, code: str) -> ValidationResult:
        """Validate code against manimgl API signatures.

        Uses dynamic ChromaDB lookups when available, falls back to hardcoded rules.

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

                # Try to get class context from the call (e.g., axes.get_riemann_rectangles)
                class_context = self._get_class_context(node)

                # Get API info from ChromaDB (if available)
                api_info = await self._get_api_info(method_name, class_context)

                # Check if method exists (when we have class context and RAG)
                if api_info is None and class_context and self._use_dynamic:
                    # Method not found in RAG - check if it's a known hallucination
                    # that should be transformed to a real method
                    method_suggestions = await self._suggest_similar_method(method_name, class_context)
                    if method_suggestions:
                        issues.append(ValidationIssue(
                            method=method_name,
                            param="",
                            issue_type="unknown_method",
                            message=f"Method '{method_name}' not found on {class_context}. Did you mean: {method_suggestions[0]}?",
                            fix=method_suggestions[0],
                            line_number=node.lineno,
                        ))

                # Check each keyword argument
                for kw in node.keywords:
                    if kw.arg is None:
                        continue  # **kwargs

                    # First check dynamic API info
                    if api_info:
                        issue = self._check_param_dynamic(method_name, kw.arg, api_info, node.lineno)
                        if issue:
                            issues.append(issue)
                            continue

                    # Fall back to hardcoded rules
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

    def _get_class_context(self, node: ast.Call) -> str | None:
        """Try to infer class context from a method call.

        For calls like `axes.get_riemann_rectangles(...)`, tries to infer
        that this is likely an Axes method.
        """
        # Check if it's an attribute access (obj.method)
        if isinstance(node.func, ast.Attribute):
            # Check if the object is a known class instantiation or common variable names
            obj = node.func.value

            # Map common variable names to likely class types
            var_class_hints = {
                "axes": "Axes",
                "ax": "Axes",
                "plane": "NumberPlane",
                "number_plane": "NumberPlane",
                "line": "NumberLine",
                "number_line": "NumberLine",
                "graph": "ParametricCurve",
            }

            if isinstance(obj, ast.Name):
                var_name = obj.id.lower()
                for hint_var, hint_class in var_class_hints.items():
                    if hint_var in var_name:
                        return hint_class

        return None

    def _check_param_dynamic(
        self,
        method_name: str,
        param_name: str,
        api_info: APIInfo,
        line_number: int,
    ) -> ValidationIssue | None:
        """Check a parameter against dynamic API info from ChromaDB.

        Returns:
            ValidationIssue if the parameter is invalid, None if valid
        """
        # First, check hardcoded PARAM_FIXES (highest priority - known CE→manimgl mappings)
        fix_key = (method_name, param_name)
        fix_info = PARAM_FIXES.get(fix_key, {})
        if not fix_info:
            # Also try with full class.method name
            fix_key_full = (api_info.full_name, param_name)
            fix_info = PARAM_FIXES.get(fix_key_full, {})

        if fix_info:
            return ValidationIssue(
                method=method_name,
                param=param_name,
                issue_type="invalid_param" if fix_info.get("action") == "remove" else "wrong_param_name",
                message=fix_info.get("message", f"Parameter '{param_name}' is not valid in manimgl {method_name}."),
                fix=fix_info.get("new_name"),
                line_number=line_number,
            )

        # Check if explicitly invalid (CE-only param from RAG)
        if param_name in api_info.invalid_params:
            return ValidationIssue(
                method=method_name,
                param=param_name,
                issue_type="invalid_param",
                message=f"Parameter '{param_name}' is not valid in manimgl {method_name}. CE-only parameter.",
                fix=None,
                line_number=line_number,
            )

        # Check against valid params list (from parameter_names or valid_params)
        # NOTE: Don't require is_verified - if we have valid_params, use them
        if api_info.valid_params:
            if param_name not in api_info.valid_params:
                # Unknown param - not in the known valid params list
                # Suggest similar param if exists
                similar = None
                for valid_p in api_info.valid_params:
                    if param_name in valid_p or valid_p in param_name:
                        similar = valid_p
                        break

                return ValidationIssue(
                    method=method_name,
                    param=param_name,
                    issue_type="wrong_param_name",
                    message=f"Parameter '{param_name}' not found in manimgl {method_name}. Valid params: {', '.join(sorted(api_info.valid_params)[:6])}",
                    fix=similar,  # Suggest similar param if found
                    line_number=line_number,
                )

        return None

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

    async def _suggest_similar_method(self, method_name: str, class_context: str) -> list[str]:
        """Suggest similar method names when a method is not found.

        Uses fuzzy matching against known API methods for the class.

        Args:
            method_name: The method name that wasn't found
            class_context: The class the method was called on

        Returns:
            List of suggested method names, most likely first
        """
        if not self.rag:
            return []

        suggestions = []

        # Known method name corrections (common LLM hallucinations)
        KNOWN_CORRECTIONS = {
            ("Axes", "get_coordinate_labels"): "add_coordinate_labels",
            ("Axes", "get_coordinates"): "add_coordinate_labels",
            ("Axes", "show_coordinates"): "add_coordinate_labels",
            ("Axes", "plot"): "get_graph",
            ("Axes", "add_coordinates"): "add_coordinate_labels",
            ("NumberPlane", "get_coordinate_labels"): "add_coordinate_labels",
            ("NumberPlane", "plot"): "get_graph",
            ("NumberLine", "get_labels"): "add_numbers",
            ("NumberLine", "add_labels"): "add_numbers",
        }

        # Check known corrections first
        key = (class_context, method_name)
        if key in KNOWN_CORRECTIONS:
            suggestions.append(KNOWN_CORRECTIONS[key])
            return suggestions

        # Search RAG for similar method names on this class
        try:
            query = f"{class_context}.{method_name}"
            results = await self.rag.search_api_signatures(query, n_results=5)

            for result in results:
                metadata = result.get("metadata", {})
                result_class = metadata.get("class_name", "")
                result_method = metadata.get("name", "")

                # Only suggest methods from the same class
                if result_class == class_context and result_method != method_name:
                    # Check if method name is similar (contains similar substring)
                    if self._is_similar_name(method_name, result_method):
                        suggestions.append(result_method)

            # Deduplicate and limit
            seen = set()
            unique = []
            for s in suggestions:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique[:3]

        except Exception as e:
            logger.warning("[PARAM-VALIDATOR] Method suggestion failed: %s", e)
            return []

    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """Check if two method names are similar enough to suggest.

        Uses simple heuristics:
        - Share common prefix (>3 chars)
        - Share common suffix (>3 chars)
        - One contains the other
        - Similar word patterns (get_X vs add_X)
        """
        # Normalize names
        n1 = name1.lower().replace("_", "")
        n2 = name2.lower().replace("_", "")

        # One contains the other
        if n1 in n2 or n2 in n1:
            return True

        # Share significant prefix
        prefix_len = 0
        for c1, c2 in zip(n1, n2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        if prefix_len >= 4:
            return True

        # Share significant suffix
        suffix_len = 0
        for c1, c2 in zip(reversed(n1), reversed(n2)):
            if c1 == c2:
                suffix_len += 1
            else:
                break
        if suffix_len >= 5:
            return True

        # Check word-level similarity (get_X vs add_X)
        words1 = set(name1.lower().split("_"))
        words2 = set(name2.lower().split("_"))
        common_words = words1 & words2
        if len(common_words) >= 1 and any(len(w) > 3 for w in common_words):
            return True

        return False

    def _auto_fix(self, code: str, issues: list[ValidationIssue]) -> str:
        """Attempt to automatically fix the issues in the code.

        Uses regex-based replacement to handle the fixes safely.
        """
        fixed = code

        for issue in issues:
            if issue.issue_type == "unknown_method" and issue.fix:
                # Replace the method name with the suggested fix
                # Pattern: .method_name( -> .fix(
                pattern = rf"\.{re.escape(issue.method)}\s*\("
                replacement = f".{issue.fix}("
                if re.search(pattern, fixed):
                    fixed = re.sub(pattern, replacement, fixed)
                    logger.info(
                        "[PARAM-VALIDATOR] Fixed unknown method: .%s() -> .%s()",
                        issue.method, issue.fix
                    )

            elif issue.issue_type == "invalid_param":
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


def clear_api_cache() -> int:
    """Clear the API signature cache.

    Useful when reloading ChromaDB or after re-indexing API signatures.

    Returns:
        Number of entries cleared
    """
    global _api_cache
    count = len(_api_cache)
    _api_cache.clear()
    logger.info("[PARAM-VALIDATOR] Cleared API cache (%d entries)", count)
    return count


def get_cache_stats() -> dict:
    """Get statistics about the API cache.

    Returns:
        Dict with cache statistics
    """
    total = len(_api_cache)
    hits = sum(1 for v in _api_cache.values() if v is not None)
    misses = total - hits

    return {
        "total_entries": total,
        "cached_apis": hits,
        "cached_misses": misses,
        "cache_keys": list(_api_cache.keys())[:20],  # First 20 keys for debugging
    }


def validate_and_fix(code: str, rag: ChromaDBService | None = None) -> str:
    """Validate code and return fixed version.

    Convenience function for use in code_bridge.py.

    Uses dynamic API validation from ChromaDB when RAG is available,
    falls back to hardcoded PARAM_FIXES rules otherwise.

    Args:
        code: Python code to validate
        rag: Optional RAG service for dynamic lookups (enables ChromaDB queries)

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
            "[PARAM-VALIDATOR] Found %d issues, applying fixes (dynamic=%s)",
            len(result.issues),
            validator._use_dynamic
        )
        for issue in result.issues:
            logger.debug(
                "[PARAM-VALIDATOR] %s.%s: %s",
                issue.method, issue.param, issue.message
            )

        if result.fixed_code:
            # Validate that fixes didn't introduce syntax errors
            try:
                import ast
                ast.parse(result.fixed_code)
                return result.fixed_code
            except SyntaxError as e:
                logger.warning(
                    "[PARAM-VALIDATOR] Fixes introduced syntax error at line %d: %s. "
                    "Returning original code.",
                    e.lineno, e.msg
                )
                return code

    return code


async def validate_and_fix_async(code: str, rag: ChromaDBService | None = None) -> str:
    """Async version of validate_and_fix.

    Prefer this in async contexts to avoid blocking on ChromaDB queries.

    Args:
        code: Python code to validate
        rag: Optional RAG service for dynamic lookups

    Returns:
        Fixed code (or original if no issues)
    """
    validator = ParameterValidator(rag)

    try:
        result = await validator.validate(code)
    except Exception as e:
        logger.warning("[PARAM-VALIDATOR] Validation failed: %s", e)
        return code

    if result.issues:
        logger.info(
            "[PARAM-VALIDATOR] Found %d issues, applying fixes (dynamic=%s)",
            len(result.issues),
            validator._use_dynamic
        )
        for issue in result.issues:
            logger.debug(
                "[PARAM-VALIDATOR] %s.%s: %s",
                issue.method, issue.param, issue.message
            )

        if result.fixed_code:
            # Validate that fixes didn't introduce syntax errors
            try:
                import ast
                ast.parse(result.fixed_code)
                return result.fixed_code
            except SyntaxError as e:
                logger.warning(
                    "[PARAM-VALIDATOR] Fixes introduced syntax error at line %d: %s. "
                    "Returning original code.",
                    e.lineno, e.msg
                )
                return code

    return code
