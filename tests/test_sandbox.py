"""Tests for AST-based code sandbox."""

from __future__ import annotations

import pytest

from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.exceptions import CodeValidationError, DangerousCodeError
from tests.conftest import (
    DANGEROUS_CODE_DUNDER,
    DANGEROUS_CODE_EVAL,
    DANGEROUS_CODE_OS,
    DANGEROUS_CODE_SUBPROCESS,
    NO_SCENE_CODE,
    SYNTAX_ERROR_CODE,
    VALID_SCENE_CODE,
)


class TestCodeSandbox:
    def test_valid_code_passes(self, sandbox: CodeSandbox):
        result = sandbox.validate(VALID_SCENE_CODE)
        assert result.valid is True
        assert result.errors == []
        assert "MyScene" in result.scenes_found

    def test_syntax_error_caught(self, sandbox: CodeSandbox):
        result = sandbox.validate(SYNTAX_ERROR_CODE)
        assert result.valid is False
        assert any("Syntax error" in e for e in result.errors)

    def test_os_import_blocked(self, sandbox: CodeSandbox):
        result = sandbox.validate(DANGEROUS_CODE_OS)
        assert result.valid is False
        assert any("Blocked import 'os'" in e for e in result.errors)

    def test_subprocess_import_blocked(self, sandbox: CodeSandbox):
        result = sandbox.validate(DANGEROUS_CODE_SUBPROCESS)
        assert result.valid is False
        assert any("Blocked import 'subprocess'" in e for e in result.errors)

    def test_eval_blocked(self, sandbox: CodeSandbox):
        result = sandbox.validate(DANGEROUS_CODE_EVAL)
        assert result.valid is False
        assert any("Dangerous builtin" in e for e in result.errors)

    def test_dunder_access_blocked(self, sandbox: CodeSandbox):
        result = sandbox.validate(DANGEROUS_CODE_DUNDER)
        assert result.valid is False
        assert any("dunder" in e.lower() or "Blocked" in e for e in result.errors)

    def test_code_too_long(self, sandbox: CodeSandbox):
        long_code = "x = 1\n" * 100000
        result = sandbox.validate(long_code)
        assert result.valid is False
        assert any("maximum length" in e for e in result.errors)

    def test_validate_or_raise_valid(self, sandbox: CodeSandbox):
        result = sandbox.validate_or_raise(VALID_SCENE_CODE)
        assert result.valid is True

    def test_validate_or_raise_dangerous(self, sandbox: CodeSandbox):
        with pytest.raises(DangerousCodeError):
            sandbox.validate_or_raise(DANGEROUS_CODE_OS)

    def test_validate_or_raise_syntax_error(self, sandbox: CodeSandbox):
        with pytest.raises(CodeValidationError):
            sandbox.validate_or_raise(SYNTAX_ERROR_CODE)

    def test_numpy_import_allowed(self, sandbox: CodeSandbox):
        code = "import numpy as np\nx = np.array([1, 2, 3])"
        result = sandbox.validate(code)
        assert result.valid is True

    def test_manim_import_allowed(self, sandbox: CodeSandbox):
        code = "from manim import *\nfrom manim.utils.color import WHITE"
        result = sandbox.validate(code)
        assert result.valid is True

    def test_no_scene_still_valid(self, sandbox: CodeSandbox):
        result = sandbox.validate(NO_SCENE_CODE)
        assert result.valid is True
        assert result.scenes_found == []

    def test_open_builtin_blocked(self, sandbox: CodeSandbox):
        code = 'from manim import *\nclass S(Scene):\n    def construct(self):\n        open("/etc/passwd")'
        result = sandbox.validate(code)
        assert result.valid is False

    def test_import_star_blocked_for_unknown(self, sandbox: CodeSandbox):
        code = "import json"
        result = sandbox.validate(code)
        assert result.valid is False
        assert any("Blocked import" in e for e in result.errors)
