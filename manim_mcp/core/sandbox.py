"""AST-based code validation to prevent dangerous operations."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from manim_mcp.exceptions import CodeValidationError, DangerousCodeError
from manim_mcp.models import ValidationResult

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

BLOCKED_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "socket", "http", "urllib",
    "requests", "importlib", "ctypes", "multiprocessing", "pickle",
    "tempfile", "glob", "ftplib", "smtplib", "webbrowser", "pathlib",
    "signal", "resource", "pty", "pipes", "fcntl", "termios",
    "code", "codeop", "compileall", "py_compile",
})

ALLOWED_PREFIXES = ("manim", "numpy", "math", "colour", "scipy")

DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "open", "input", "breakpoint",
})

BLOCKED_DUNDERS = frozenset({
    "__subclasses__", "__class__", "__bases__", "__globals__",
    "__code__", "__builtins__", "__mro__", "__qualname__",
})


class CodeSandbox:
    def __init__(self, config: ManimMCPConfig) -> None:
        self.max_code_length = config.max_code_length

    def validate(self, code: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if len(code) > self.max_code_length:
            errors.append(f"Code exceeds maximum length of {self.max_code_length} characters")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
            errors.append(msg)
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        self._check_imports(tree, errors)
        self._check_dangerous_builtins(tree, errors)
        self._check_dunder_access(tree, errors)
        self._check_attribute_patterns(tree, errors)

        scenes_found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = _get_name(base)
                    if base_name and "Scene" in base_name:
                        scenes_found.append(node.name)
                        break

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            scenes_found=scenes_found,
        )

    def validate_or_raise(self, code: str) -> ValidationResult:
        result = self.validate(code)
        if not result.valid:
            for err in result.errors:
                if "Blocked" in err or "Dangerous" in err or "dunder" in err.lower():
                    raise DangerousCodeError("; ".join(result.errors))
            raise CodeValidationError("; ".join(result.errors))
        return result

    def _check_imports(self, tree: ast.AST, errors: list[str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_module_name(alias.name, node.lineno, errors)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._check_module_name(node.module, node.lineno, errors)

    def _check_module_name(self, module: str, lineno: int, errors: list[str]) -> None:
        top_level = module.split(".")[0]
        if top_level in BLOCKED_MODULES:
            errors.append(f"Blocked import '{module}' at line {lineno}")
            return
        if not any(module.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            # Allow standard library math-adjacent modules
            allowed_extras = {"random", "functools", "itertools", "collections", "typing", "enum", "abc", "dataclasses", "copy"}
            if top_level not in allowed_extras:
                errors.append(f"Blocked import '{module}' at line {lineno}: only manim, numpy, math, colour, scipy and standard math utilities are allowed")

    def _check_dangerous_builtins(self, tree: ast.AST, errors: list[str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_call_name(node)
                if name in DANGEROUS_BUILTINS:
                    errors.append(f"Dangerous builtin '{name}' at line {node.lineno}")
            elif isinstance(node, ast.Name) and node.id in DANGEROUS_BUILTINS:
                if isinstance(node.ctx, ast.Load):
                    errors.append(f"Dangerous builtin reference '{node.id}' at line {node.lineno}")

    def _check_dunder_access(self, tree: ast.AST, errors: list[str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_DUNDERS:
                errors.append(f"Blocked dunder access '{node.attr}' at line {node.lineno}")

    def _check_attribute_patterns(self, tree: ast.AST, errors: list[str]) -> None:
        dangerous_patterns = {
            ("os", "system"), ("os", "popen"), ("os", "exec"),
            ("os", "execv"), ("os", "execve"), ("os", "fork"),
            ("subprocess", "Popen"), ("subprocess", "call"),
            ("subprocess", "run"), ("subprocess", "check_output"),
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                pair = (node.value.id, node.attr)
                if pair in dangerous_patterns:
                    errors.append(f"Blocked attribute access '{node.value.id}.{node.attr}' at line {node.lineno}")


def _get_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _get_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None
