"""Tests for the animation pipeline with mocked LLM and renderer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.llm import GeminiClient, _strip_fences
from manim_mcp.core.pipeline import AnimationPipeline
from manim_mcp.core.renderer import ManimRenderer, RenderOutput
from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.core.storage import S3Storage
from manim_mcp.core.tracker import RenderTracker
from manim_mcp.exceptions import CodeValidationError, ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality, RenderStatus

GOOD_CODE = """
from manim import *

class PythagoreanTheorem(Scene):
    def construct(self):
        title = Text("Pythagorean Theorem")
        self.play(Write(title))
        self.wait(1)
"""

DANGEROUS_CODE = """
import os
os.system("rm -rf /")

class Bad(Scene):
    def construct(self):
        pass
"""


@pytest.fixture
def config():
    return ManimMCPConfig(
        tracker_db_path=":memory:",
        gemini_api_key="test-key",
        gemini_max_retries=2,
    )


@pytest.fixture
async def tracker(config):
    t = RenderTracker(config)
    await t.initialize()
    yield t
    await t.close()


@pytest.fixture
def storage():
    s = MagicMock(spec=S3Storage)
    s.available = False
    s.upload_file = AsyncMock()
    s.generate_presigned_url = AsyncMock(return_value=None)
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=GeminiClient)
    llm.generate_code = AsyncMock(return_value=GOOD_CODE)
    llm.edit_code = AsyncMock(return_value=GOOD_CODE)
    llm.fix_code = AsyncMock(return_value=GOOD_CODE)
    return llm


@pytest.fixture
def mock_renderer():
    renderer = MagicMock(spec=ManimRenderer)
    renderer.render_scene = AsyncMock(return_value=RenderOutput(
        scene_name="PythagoreanTheorem",
        local_path="/tmp/video.mp4",
        file_size_bytes=1024,
        render_time_seconds=5.0,
        width=1280,
        height=720,
        format="mp4",
        quality="medium",
    ))
    return renderer


@pytest.fixture
def pipeline(config, mock_llm, mock_renderer, storage, tracker):
    sandbox = CodeSandbox(config)
    scene_parser = SceneParser()
    return AnimationPipeline(config, mock_llm, mock_renderer, sandbox, scene_parser, tracker, storage)


class TestAnimationPipeline:
    async def test_generate_success(self, pipeline, mock_llm):
        result = await pipeline.generate("Explain the Pythagorean theorem")
        assert result.status == RenderStatus.completed
        assert result.render_id is not None
        assert result.source_code == GOOD_CODE
        assert result.prompt == "Explain the Pythagorean theorem"
        mock_llm.generate_code.assert_called_once_with("Explain the Pythagorean theorem")

    async def test_generate_stores_in_tracker(self, pipeline, tracker):
        result = await pipeline.generate("Test prompt")
        stored = await tracker.get_render(result.render_id)
        assert stored.status == RenderStatus.completed
        assert stored.original_prompt == "Test prompt"
        assert stored.source_code == GOOD_CODE

    async def test_edit_success(self, pipeline, mock_llm, tracker):
        # First generate
        gen_result = await pipeline.generate("Original animation")

        # Then edit
        edit_result = await pipeline.edit(
            gen_result.render_id, "Make the title red"
        )
        assert edit_result.status == RenderStatus.completed
        assert edit_result.render_id != gen_result.render_id
        mock_llm.edit_code.assert_called_once()

        # Check parent link
        stored = await tracker.get_render(edit_result.render_id)
        assert stored.parent_render_id == gen_result.render_id
        assert stored.edit_instructions == "Make the title red"

    async def test_edit_nonexistent_render(self, pipeline):
        with pytest.raises(Exception, match="not found"):
            await pipeline.edit("nonexistent", "Change something")

    async def test_edit_no_source_code(self, pipeline, tracker):
        # Create a render with no source code
        await tracker.create_render("empty-render")
        with pytest.raises(ManimMCPError, match="no source code"):
            await pipeline.edit("empty-render", "Change something")

    async def test_self_heal_on_validation_failure(self, pipeline, mock_llm):
        # First call returns dangerous code, fix_code returns good code
        mock_llm.generate_code = AsyncMock(return_value=DANGEROUS_CODE)
        mock_llm.fix_code = AsyncMock(return_value=GOOD_CODE)

        result = await pipeline.generate("Something")
        assert result.status == RenderStatus.completed
        mock_llm.fix_code.assert_called_once()

    async def test_validation_fails_after_retries(self, pipeline, mock_llm):
        mock_llm.generate_code = AsyncMock(return_value=DANGEROUS_CODE)
        mock_llm.fix_code = AsyncMock(return_value=DANGEROUS_CODE)

        with pytest.raises(CodeValidationError, match="failed validation"):
            await pipeline.generate("Something bad")

    async def test_generate_failed_updates_tracker(self, pipeline, mock_llm, tracker):
        mock_llm.generate_code = AsyncMock(return_value=DANGEROUS_CODE)
        mock_llm.fix_code = AsyncMock(return_value=DANGEROUS_CODE)

        with pytest.raises(CodeValidationError):
            await pipeline.generate("Bad prompt")

        renders = await tracker.list_renders(status=RenderStatus.failed)
        assert len(renders) == 1


class TestStripFences:
    def test_strips_python_fences(self):
        text = "```python\nprint('hello')\n```"
        assert _strip_fences(text) == "print('hello')"

    def test_strips_plain_fences(self):
        text = "```\nprint('hello')\n```"
        assert _strip_fences(text) == "print('hello')"

    def test_no_fences_unchanged(self):
        text = "print('hello')"
        assert _strip_fences(text) == "print('hello')"

    def test_strips_whitespace(self):
        text = "  \n```python\ncode\n```\n  "
        assert _strip_fences(text) == "code"
