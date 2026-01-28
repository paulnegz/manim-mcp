"""Tests for the render engine with mocked subprocess."""

from __future__ import annotations

import os
import tempfile

import pytest

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.renderer import ManimRenderer
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.exceptions import CodeValidationError
from manim_mcp.models import OutputFormat, RenderQuality, RenderSceneInput


@pytest.fixture
def config():
    return ManimMCPConfig(tracker_db_path=":memory:", render_timeout=10, gemini_api_key="test")


@pytest.fixture
def renderer(config):
    scene_parser = SceneParser()
    return ManimRenderer(config, scene_parser)


VALID_CODE = """
from manim import *

class TestScene(Scene):
    def construct(self):
        self.add(Circle())
"""


class TestManimRenderer:
    def test_resolve_scene_name_single(self, renderer: ManimRenderer):
        scenes = renderer.scene_parser.parse_scenes(VALID_CODE)
        name = renderer._resolve_scene_name(scenes, None)
        assert name == "TestScene"

    def test_resolve_scene_name_empty_raises(self, renderer: ManimRenderer):
        with pytest.raises(CodeValidationError, match="No Scene subclass"):
            renderer._resolve_scene_name([], None)

    def test_resolve_scene_name_multiple_no_name(self, renderer: ManimRenderer):
        code = """
from manim import *
class A(Scene):
    def construct(self): pass
class B(Scene):
    def construct(self): pass
"""
        scenes = renderer.scene_parser.parse_scenes(code)
        with pytest.raises(CodeValidationError, match="Multiple scenes"):
            renderer._resolve_scene_name(scenes, None)

    def test_resolve_scene_name_multiple_with_name(self, renderer: ManimRenderer):
        code = """
from manim import *
class A(Scene):
    def construct(self): pass
class B(Scene):
    def construct(self): pass
"""
        scenes = renderer.scene_parser.parse_scenes(code)
        name = renderer._resolve_scene_name(scenes, "B")
        assert name == "B"

    def test_resolve_scene_name_not_found(self, renderer: ManimRenderer):
        code = """
from manim import *
class A(Scene):
    def construct(self): pass
class B(Scene):
    def construct(self): pass
"""
        scenes = renderer.scene_parser.parse_scenes(code)
        with pytest.raises(CodeValidationError, match="not found"):
            renderer._resolve_scene_name(scenes, "NonExistent")

    def test_build_command(self, renderer: ManimRenderer):
        input_model = RenderSceneInput(
            code="...",
            quality=RenderQuality.high,
            format=OutputFormat.gif,
            fps=24,
            transparent=True,
            background_color="#FF0000",
        )
        cmd = renderer._build_command(input_model, "TestScene", "/tmp/test", "/tmp/test/scene.py")
        assert "manim" in cmd[0]
        assert "-qh" in cmd
        assert "gif" in cmd
        assert "24" in cmd
        assert "-t" in cmd
        assert "#FF0000" in cmd
        assert "TestScene" in cmd

    def test_build_command_with_resolution(self, renderer: ManimRenderer):
        input_model = RenderSceneInput(
            code="...",
            quality=RenderQuality.medium,
            format=OutputFormat.mp4,
            resolution="1920x1080",
        )
        cmd = renderer._build_command(input_model, "S", "/tmp/t", "/tmp/t/scene.py")
        assert "-r" in cmd
        assert "1920x1080" in cmd

    def test_find_output_none_when_empty(self, renderer: ManimRenderer):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "media"))
        input_model = RenderSceneInput(code="...", quality=RenderQuality.low, format=OutputFormat.mp4)
        result = renderer._find_output(tmp, "TestScene", input_model)
        assert result is None

    def test_find_output_finds_video(self, renderer: ManimRenderer):
        tmp = tempfile.mkdtemp()
        video_dir = os.path.join(tmp, "media", "videos", "scene", "480p15")
        os.makedirs(video_dir)
        video_path = os.path.join(video_dir, "TestScene.mp4")
        with open(video_path, "w") as f:
            f.write("fake")

        input_model = RenderSceneInput(code="...", quality=RenderQuality.low, format=OutputFormat.mp4)
        result = renderer._find_output(tmp, "TestScene", input_model)
        assert result == video_path

    def test_parse_resolution_from_quality(self, renderer: ManimRenderer):
        input_model = RenderSceneInput(code="...", quality=RenderQuality.high, format=OutputFormat.mp4)
        w, h = renderer._parse_resolution(input_model)
        assert (w, h) == (1920, 1080)

    def test_parse_resolution_override(self, renderer: ManimRenderer):
        input_model = RenderSceneInput(
            code="...", quality=RenderQuality.low, format=OutputFormat.mp4, resolution="3840x2160"
        )
        w, h = renderer._parse_resolution(input_model)
        assert (w, h) == (3840, 2160)
