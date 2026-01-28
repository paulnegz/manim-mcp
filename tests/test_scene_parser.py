"""Tests for scene class discovery."""

from __future__ import annotations

from manim_mcp.core.scene_parser import SceneParser
from tests.conftest import MULTI_SCENE_CODE, NO_SCENE_CODE, SYNTAX_ERROR_CODE, VALID_SCENE_CODE


class TestSceneParser:
    def test_single_scene(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(VALID_SCENE_CODE)
        assert len(scenes) == 1
        assert scenes[0].name == "MyScene"
        assert scenes[0].has_construct is True
        assert "Scene" in scenes[0].base_classes

    def test_multiple_scenes(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(MULTI_SCENE_CODE)
        assert len(scenes) == 2
        names = {s.name for s in scenes}
        assert names == {"FirstScene", "SecondScene"}

    def test_threed_scene_detected(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(MULTI_SCENE_CODE)
        second = next(s for s in scenes if s.name == "SecondScene")
        assert "ThreeDScene" in second.base_classes

    def test_no_scenes(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(NO_SCENE_CODE)
        assert scenes == []

    def test_syntax_error_returns_empty(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(SYNTAX_ERROR_CODE)
        assert scenes == []

    def test_no_construct_method(self, scene_parser: SceneParser):
        code = """
from manim import *
class EmptyScene(Scene):
    pass
"""
        scenes = scene_parser.parse_scenes(code)
        assert len(scenes) == 1
        assert scenes[0].has_construct is False

    def test_line_number_tracked(self, scene_parser: SceneParser):
        scenes = scene_parser.parse_scenes(VALID_SCENE_CODE)
        assert scenes[0].line_number > 0

    def test_moving_camera_scene(self, scene_parser: SceneParser):
        code = """
from manim import *
class CamScene(MovingCameraScene):
    def construct(self):
        self.camera.frame.animate.set_width(10)
"""
        scenes = scene_parser.parse_scenes(code)
        assert len(scenes) == 1
        assert scenes[0].name == "CamScene"
        assert "MovingCameraScene" in scenes[0].base_classes
