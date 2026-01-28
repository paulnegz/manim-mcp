"""Tests for SQLite render tracker."""

from __future__ import annotations

import pytest

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.tracker import RenderTracker
from manim_mcp.exceptions import RenderNotFoundError
from manim_mcp.models import RenderStatus


@pytest.fixture
async def tracker():
    config = ManimMCPConfig(tracker_db_path=":memory:")
    t = RenderTracker(config)
    await t.initialize()
    yield t
    await t.close()


class TestRenderTracker:
    async def test_create_and_get(self, tracker: RenderTracker):
        meta = await tracker.create_render("test-001", scene_name="MyScene", quality="medium", fmt="mp4")
        assert meta.render_id == "test-001"
        assert meta.status == RenderStatus.pending

        fetched = await tracker.get_render("test-001")
        assert fetched.render_id == "test-001"
        assert fetched.scene_name == "MyScene"

    async def test_update_render(self, tracker: RenderTracker):
        await tracker.create_render("test-002")
        updated = await tracker.update_render("test-002", status=RenderStatus.rendering)
        assert updated.status == RenderStatus.rendering

    async def test_update_nonexistent_raises(self, tracker: RenderTracker):
        with pytest.raises(RenderNotFoundError):
            await tracker.update_render("nonexistent", status=RenderStatus.completed)

    async def test_get_nonexistent_raises(self, tracker: RenderTracker):
        with pytest.raises(RenderNotFoundError):
            await tracker.get_render("nonexistent")

    async def test_list_renders(self, tracker: RenderTracker):
        await tracker.create_render("r1")
        await tracker.create_render("r2")
        await tracker.create_render("r3")

        renders = await tracker.list_renders(limit=10)
        assert len(renders) == 3

    async def test_list_renders_with_status_filter(self, tracker: RenderTracker):
        await tracker.create_render("r1")
        await tracker.create_render("r2")
        await tracker.update_render("r2", status=RenderStatus.completed)

        pending = await tracker.list_renders(status=RenderStatus.pending)
        assert len(pending) == 1
        assert pending[0].render_id == "r1"

    async def test_list_renders_pagination(self, tracker: RenderTracker):
        for i in range(5):
            await tracker.create_render(f"r{i}")

        page = await tracker.list_renders(limit=2, offset=2)
        assert len(page) == 2

    async def test_delete_render(self, tracker: RenderTracker):
        await tracker.create_render("del-me")
        await tracker.delete_render("del-me")

        with pytest.raises(RenderNotFoundError):
            await tracker.get_render("del-me")

    async def test_delete_nonexistent_raises(self, tracker: RenderTracker):
        with pytest.raises(RenderNotFoundError):
            await tracker.delete_render("nope")

    async def test_count_active_renders(self, tracker: RenderTracker):
        await tracker.create_render("a1")
        await tracker.create_render("a2")
        await tracker.update_render("a2", status=RenderStatus.rendering)
        await tracker.create_render("a3")
        await tracker.update_render("a3", status=RenderStatus.completed)

        count = await tracker.count_active_renders()
        assert count == 2  # a1 (pending) + a2 (rendering)

    async def test_find_by_code_hash(self, tracker: RenderTracker):
        await tracker.create_render("h1", code_hash="abc123")
        await tracker.update_render("h1", status=RenderStatus.completed, s3_url="s3://test")

        found = await tracker.find_by_code_hash("abc123")
        assert found is not None
        assert found.render_id == "h1"

    async def test_find_by_code_hash_not_found(self, tracker: RenderTracker):
        result = await tracker.find_by_code_hash("nonexistent_hash")
        assert result is None
