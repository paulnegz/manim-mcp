"""Generation tools: generate_animation, edit_animation.

generate_animation and edit_animation now support two modes:

1. **Async mode** (default when SSE is available):
   Returns immediately with render_id + status "rendering".
   The pipeline runs in a background task, pushing SSE events.
   Clients subscribe via GET /progress/{render_id}.

2. **Sync mode** (fallback / MCP progress):
   Runs the pipeline synchronously and returns the full result.
   Used when called directly via MCP tool protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time

from mcp.server.fastmcp import Context, FastMCP

from manim_mcp.core.pipeline import RenderParams
from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality
from manim_mcp.progress import get_broadcaster

logger = logging.getLogger(__name__)

# ── Background task registry ──────────────────────────────────────────
# Prevents background tasks from being garbage-collected.
_background_tasks: set[asyncio.Task] = set()


def register_generate_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    async def generate_animation(
        ctx: Context,
        prompt: str,
        quality: str = "low",
        format: str = "mp4",
        resolution: str | None = None,
        fps: int | None = None,
        background_color: str | None = None,
        transparent: bool = False,
        save_last_frame: bool = False,
        audio: bool = True,
        voice: str | None = None,
        async_mode: bool = True,
    ) -> dict:
        """Create a Manim animation from a text description.

        Describe what you want to see -- the server generates Manim code and
        renders it into a video automatically.

        Args:
            prompt: What to animate (e.g. "Explain the Pythagorean theorem with a visual proof").
            quality: Render quality: low (480p), medium (720p), high (1080p), production (1440p), fourk (4K).
            format: Output format: mp4, gif, webm, mov, png.
            resolution: Resolution override as WxH (e.g. '1920x1080'). Overrides quality default.
            fps: Frames per second override.
            background_color: Background color hex (e.g. '#000000').
            transparent: Transparent background (for png/gif).
            save_last_frame: Save last frame as image instead of video.
            audio: Generate audio narration for the animation (default: True).
            voice: TTS voice (Puck, Charon, Kore, Fenrir, Aoede, etc.).
            async_mode: If True, return immediately with render_id and stream progress via SSE.

        Returns:
            In async_mode: {render_id, status: "rendering", progress_url}
            In sync mode:  Full AnimationResult dict.
        """
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt

        logger.info("=" * 60)
        logger.info("GENERATE_ANIMATION TOOL CALLED")
        logger.info("  Prompt: %s", prompt_preview)
        logger.info("  Quality: %s, Format: %s, Async: %s", quality, format, async_mode)
        logger.info("=" * 60)

        app = ctx.request_context.lifespan_context

        params = RenderParams(
            quality=RenderQuality(quality),
            fmt=OutputFormat(format),
            resolution=resolution,
            fps=fps,
            background_color=background_color,
            transparent=transparent,
            save_last_frame=save_last_frame,
            audio=audio,
            voice=voice,
        )

        if async_mode:
            return await _start_async_generate(app, prompt, params, start_time)
        else:
            return await _run_sync_generate(ctx, app, prompt, params, start_time)

    @mcp.tool()
    async def edit_animation(
        ctx: Context,
        render_id: str,
        instructions: str,
        quality: str = "low",
        format: str = "mp4",
        resolution: str | None = None,
        fps: int | None = None,
        background_color: str | None = None,
        transparent: bool = False,
        save_last_frame: bool = False,
        audio: bool = True,
        voice: str | None = None,
        async_mode: bool = True,
    ) -> dict:
        """Edit an existing animation by describing what to change.

        Pass the render_id from a previous generate_animation or edit_animation
        call, plus natural-language instructions for what to modify.

        Args:
            render_id: ID of the animation to edit (from a previous result).
            instructions: What to change (e.g. "Make the circle red and add axis labels").
            quality: Render quality for the new version.
            format: Output format for the new version.
            resolution: Resolution override as WxH (e.g. '1920x1080').
            fps: Frames per second override.
            background_color: Background color hex (e.g. '#000000').
            transparent: Transparent background.
            save_last_frame: Save last frame as image instead of video.
            audio: Generate audio narration for the animation (default: True).
            voice: TTS voice (Puck, Charon, Kore, Fenrir, Aoede, etc.).
            async_mode: If True, return immediately with render_id and stream progress via SSE.

        Returns:
            In async_mode: {render_id, status: "rendering", progress_url}
            In sync mode:  Full AnimationResult dict.
        """
        start_time = time.time()
        instructions_preview = instructions[:100] + "..." if len(instructions) > 100 else instructions

        logger.info("=" * 60)
        logger.info("EDIT_ANIMATION TOOL CALLED")
        logger.info("  Render ID: %s", render_id)
        logger.info("  Instructions: %s", instructions_preview)
        logger.info("  Quality: %s, Format: %s, Async: %s", quality, format, async_mode)
        logger.info("=" * 60)

        app = ctx.request_context.lifespan_context

        params = RenderParams(
            quality=RenderQuality(quality),
            fmt=OutputFormat(format),
            resolution=resolution,
            fps=fps,
            background_color=background_color,
            transparent=transparent,
            save_last_frame=save_last_frame,
            audio=audio,
            voice=voice,
        )

        if async_mode:
            return await _start_async_edit(app, render_id, instructions, params, start_time)
        else:
            return await _run_sync_edit(ctx, app, render_id, instructions, params, start_time)


# ── Async mode (background task + SSE) ───────────────────────────────

async def _start_async_generate(app, prompt: str, params: RenderParams, start_time: float) -> dict:
    """Start pipeline in background, return render_id immediately."""
    import uuid

    broadcaster = get_broadcaster()
    render_id = uuid.uuid4().hex[:12]
    broadcaster.create_render(render_id)

    # Send initial progress event
    broadcaster.send_progress(render_id, 0, "queued")

    def thumbnail_callback(rid: str, thumb_url: str):
        """Push early thumbnail to SSE subscribers."""
        broadcaster.send_thumbnail(rid, thumb_url)

    async def _run():
        try:
            async def progress_callback(stage: str, pct: int):
                broadcaster.send_progress(render_id, pct, stage)

            if app.config.agent_mode == "advanced":
                result = await app.pipeline.generate_advanced(
                    prompt=prompt, params=params,
                    progress_callback=progress_callback,
                    thumbnail_callback=thumbnail_callback,
                    render_id=render_id,
                )
            else:
                result = await app.pipeline.generate(
                    prompt=prompt, params=params,
                    progress_callback=progress_callback,
                    thumbnail_callback=thumbnail_callback,
                    render_id=render_id,
                )

            elapsed = time.time() - start_time
            logger.info("GENERATE_ANIMATION (async) SUCCEEDED in %.2fs | render_id=%s", elapsed, result.render_id)

            broadcaster.send_complete(
                render_id,
                url=result.url,
                thumbnail_url=result.thumbnail_url,
                result=result.model_dump(),
            )
        except ManimMCPError as e:
            elapsed = time.time() - start_time
            logger.error("GENERATE_ANIMATION (async) FAILED after %.2fs: %s", elapsed, e)
            broadcaster.send_error(render_id, str(e))
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("GENERATE_ANIMATION (async) ERROR after %.2fs: %s", elapsed, e)
            broadcaster.send_error(render_id, f"Unexpected error: {e}")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "render_id": render_id,
        "status": "rendering",
        "progress_url": f"/progress/{render_id}",
        "message": "Animation rendering started. Subscribe to progress_url for real-time updates.",
    }


async def _start_async_edit(app, render_id: str, instructions: str, params: RenderParams, start_time: float) -> dict:
    """Start edit pipeline in background, return new render_id immediately."""
    import uuid

    broadcaster = get_broadcaster()
    new_render_id = uuid.uuid4().hex[:12]
    broadcaster.create_render(new_render_id)

    broadcaster.send_progress(new_render_id, 0, "queued")

    def thumbnail_callback(rid: str, thumb_url: str):
        broadcaster.send_thumbnail(rid, thumb_url)

    async def _run():
        try:
            async def progress_callback(stage: str, pct: int):
                broadcaster.send_progress(new_render_id, pct, stage)

            result = await app.pipeline.edit(
                render_id=render_id,
                instructions=instructions,
                params=params,
                progress_callback=progress_callback,
                thumbnail_callback=thumbnail_callback,
                new_render_id=new_render_id,
            )

            elapsed = time.time() - start_time
            logger.info("EDIT_ANIMATION (async) SUCCEEDED in %.2fs | render_id=%s", elapsed, result.render_id)

            broadcaster.send_complete(
                new_render_id,
                url=result.url,
                thumbnail_url=result.thumbnail_url,
                result=result.model_dump(),
            )
        except ManimMCPError as e:
            elapsed = time.time() - start_time
            logger.error("EDIT_ANIMATION (async) FAILED after %.2fs: %s", elapsed, e)
            broadcaster.send_error(new_render_id, str(e))
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("EDIT_ANIMATION (async) ERROR after %.2fs: %s", elapsed, e)
            broadcaster.send_error(new_render_id, f"Unexpected error: {e}")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "render_id": new_render_id,
        "parent_render_id": render_id,
        "status": "rendering",
        "progress_url": f"/progress/{new_render_id}",
        "message": "Edit rendering started. Subscribe to progress_url for real-time updates.",
    }


# ── Sync mode (original blocking behavior) ───────────────────────────

async def _run_sync_generate(ctx: Context, app, prompt: str, params: RenderParams, start_time: float) -> dict:
    """Run pipeline synchronously (original behavior)."""
    async def progress_callback(stage: str, pct: int):
        await ctx.report_progress(pct, 100)

    try:
        await ctx.report_progress(0, 100)

        if app.config.agent_mode == "advanced":
            result = await app.pipeline.generate_advanced(
                prompt=prompt, params=params, progress_callback=progress_callback,
            )
        else:
            result = await app.pipeline.generate(
                prompt=prompt, params=params, progress_callback=progress_callback,
            )
        await ctx.report_progress(100, 100)

        elapsed = time.time() - start_time
        logger.info("GENERATE_ANIMATION (sync) SUCCEEDED in %.2fs | render_id=%s", elapsed, result.render_id)
        return result.model_dump()

    except ManimMCPError as e:
        elapsed = time.time() - start_time
        logger.error("GENERATE_ANIMATION (sync) FAILED after %.2fs: %s", elapsed, e)
        return {"error": True, "message": str(e)}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("GENERATE_ANIMATION (sync) ERROR after %.2fs: %s", elapsed, e)
        return {"error": True, "message": f"Unexpected error: {e}"}


async def _run_sync_edit(ctx: Context, app, render_id: str, instructions: str, params: RenderParams, start_time: float) -> dict:
    """Run edit pipeline synchronously (original behavior)."""
    async def progress_callback(stage: str, pct: int):
        await ctx.report_progress(pct, 100)

    try:
        await ctx.report_progress(0, 100)
        result = await app.pipeline.edit(
            render_id=render_id,
            instructions=instructions,
            params=params,
            progress_callback=progress_callback,
        )
        await ctx.report_progress(100, 100)

        elapsed = time.time() - start_time
        logger.info("EDIT_ANIMATION (sync) SUCCEEDED in %.2fs | render_id=%s", elapsed, result.render_id)
        return result.model_dump()

    except ManimMCPError as e:
        elapsed = time.time() - start_time
        logger.error("EDIT_ANIMATION (sync) FAILED after %.2fs: %s", elapsed, e)
        return {"error": True, "message": str(e)}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("EDIT_ANIMATION (sync) ERROR after %.2fs: %s", elapsed, e)
        return {"error": True, "message": f"Unexpected error: {e}"}
