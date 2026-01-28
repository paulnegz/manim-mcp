"""End-to-end test: Gemini generates code → Manim renders video → verify output.

Run with: pytest tests/test_e2e.py -v -s
Requires: MANIM_MCP_GEMINI_API_KEY set, manim + ffmpeg installed.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.llm import GeminiClient
from manim_mcp.core.pipeline import AnimationPipeline, RenderParams
from manim_mcp.core.renderer import ManimRenderer
from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.core.storage import S3Storage
from manim_mcp.core.tracker import RenderTracker
from manim_mcp.models import OutputFormat, RenderQuality, RenderStatus

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

pytestmark = pytest.mark.skipif(
    not os.environ.get("MANIM_MCP_GEMINI_API_KEY"),
    reason="MANIM_MCP_GEMINI_API_KEY not set",
)


@pytest.fixture
async def pipeline():
    config = ManimMCPConfig(tracker_db_path=":memory:")

    tracker = RenderTracker(config)
    await tracker.initialize()

    storage = S3Storage(config)
    await storage.initialize()

    sandbox = CodeSandbox(config)
    scene_parser = SceneParser()
    llm = GeminiClient(config)
    renderer = ManimRenderer(config, scene_parser)
    pipe = AnimationPipeline(config, llm, renderer, sandbox, scene_parser, tracker, storage)

    yield pipe, tracker, storage

    await tracker.close()


class TestEndToEnd:
    async def test_generate_laws_of_motion(self, pipeline):
        pipe, tracker, storage = pipeline

        result = await pipe.generate(
            "Create a short video on Newton's laws of motion",
            params=RenderParams(quality=RenderQuality.low, fmt=OutputFormat.mp4),
        )

        # Verify result
        assert result.status == RenderStatus.completed
        assert result.render_id is not None
        assert result.source_code is not None
        assert "Scene" in result.source_code
        assert result.prompt == "Create a short video on Newton's laws of motion"
        assert result.message == "Animation created successfully"

        # Verify a video file was produced
        stored = await tracker.get_render(result.render_id)
        assert stored.status == RenderStatus.completed
        assert stored.source_code is not None
        assert stored.file_size_bytes is not None
        assert stored.file_size_bytes > 0
        assert stored.render_time_seconds is not None

        # If local path set, verify file exists
        if stored.local_path:
            assert os.path.exists(stored.local_path), f"Video file missing: {stored.local_path}"
            assert os.path.getsize(stored.local_path) > 0

        # If S3 available, verify upload
        if storage.available and stored.s3_object_key:
            exists = await storage.object_exists(stored.s3_object_key)
            assert exists, f"S3 object missing: {stored.s3_object_key}"

        print(f"\n  render_id: {result.render_id}")
        print(f"  file_size: {stored.file_size_bytes} bytes")
        print(f"  render_time: {stored.render_time_seconds}s")
        print(f"  local_path: {stored.local_path}")
        print(f"  s3_url: {stored.s3_url}")
        print(f"  url: {result.url}")
