"""Integration tests for the text-to-video flow (no server, no LLM)."""

from __future__ import annotations

from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.config import ManimMCPConfig


class TestToolIntegration:
    """Verify sandbox + parser work together for generated code patterns."""

    def test_typical_generated_code_validates(self):
        config = ManimMCPConfig(tracker_db_path=":memory:", gemini_api_key="test")
        sandbox = CodeSandbox(config)
        parser = SceneParser()

        code = """
from manim import *

class MyAnimation(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.play(circle.animate.shift(RIGHT * 2))
        self.wait()
"""
        validation = sandbox.validate(code)
        assert validation.valid is True

        scenes = parser.parse_scenes(code)
        assert len(scenes) == 1
        assert scenes[0].name == "MyAnimation"
        assert scenes[0].has_construct is True

    def test_dangerous_code_rejected(self):
        config = ManimMCPConfig(tracker_db_path=":memory:", gemini_api_key="test")
        sandbox = CodeSandbox(config)

        code = """
import subprocess
from manim import *

class BadScene(Scene):
    def construct(self):
        subprocess.run(["ls"])
"""
        validation = sandbox.validate(code)
        assert validation.valid is False
        assert any("subprocess" in e for e in validation.errors)

    def test_latex_expression_pattern_validates(self):
        config = ManimMCPConfig(tracker_db_path=":memory:", gemini_api_key="test")
        sandbox = CodeSandbox(config)
        parser = SceneParser()

        code = r'''from manim import *

class LatexDemo(Scene):
    def construct(self):
        eq = MathTex(r"E = mc^2", font_size=48)
        eq.set_color(BLUE)
        self.play(Write(eq))
        self.wait(2)
'''
        validation = sandbox.validate(code)
        assert validation.valid is True

        scenes = parser.parse_scenes(code)
        assert scenes[0].name == "LatexDemo"

    def test_numpy_usage_validates(self):
        config = ManimMCPConfig(tracker_db_path=":memory:", gemini_api_key="test")
        sandbox = CodeSandbox(config)

        code = """
from manim import *
import numpy as np

class GraphScene(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3], y_range=[-1, 1])
        graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        self.play(Create(axes), Create(graph))
"""
        validation = sandbox.validate(code)
        assert validation.valid is True
