"""Shared test fixtures."""

from __future__ import annotations

import pytest

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser


@pytest.fixture
def config() -> ManimMCPConfig:
    return ManimMCPConfig(
        tracker_db_path=":memory:",
        s3_endpoint="localhost:9000",
        s3_access_key="test",
        s3_secret_key="test",
        gemini_api_key="test-key",
    )


@pytest.fixture
def sandbox(config: ManimMCPConfig) -> CodeSandbox:
    return CodeSandbox(config)


@pytest.fixture
def scene_parser() -> SceneParser:
    return SceneParser()


VALID_SCENE_CODE = """
from manim import *

class MyScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
"""

MULTI_SCENE_CODE = """
from manim import *

class FirstScene(Scene):
    def construct(self):
        self.add(Circle())

class SecondScene(ThreeDScene):
    def construct(self):
        self.add(Sphere())
"""

DANGEROUS_CODE_OS = """
import os
os.system("rm -rf /")
"""

DANGEROUS_CODE_SUBPROCESS = """
import subprocess
subprocess.run(["ls"])
"""

DANGEROUS_CODE_EVAL = """
from manim import *
class Bad(Scene):
    def construct(self):
        eval("print('hacked')")
"""

DANGEROUS_CODE_DUNDER = """
from manim import *
class Bad(Scene):
    def construct(self):
        x = "".__class__.__bases__[0].__subclasses__()
"""

NO_SCENE_CODE = """
from manim import *
x = Circle()
"""

SYNTAX_ERROR_CODE = """
def foo(
    print("missing paren")
"""
