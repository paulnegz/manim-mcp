"""Generation tools: generate_animation, edit_animation."""

from __future__ import annotations

import logging
import time

from mcp.server.fastmcp import Context, FastMCP

from manim_mcp.core.pipeline import RenderParams
from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality

logger = logging.getLogger(__name__)


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
    ) -> dict:
        """Create a Manim animation from a text description.

        Describe what you want to see — the server generates Manim code and
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

        Returns:
            Animation result with video URL, render ID (for editing), and generated source code.
        """
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt

        logger.info("=" * 60)
        logger.info("GENERATE_ANIMATION TOOL CALLED")
        logger.info(f"  Prompt: {prompt_preview}")
        logger.info(f"  Quality: {quality}, Format: {format}")
        logger.info(f"  Resolution: {resolution}, FPS: {fps}")
        logger.info("=" * 60)

        app = ctx.request_context.lifespan_context

        async def progress_callback(stage: str, pct: int):
            """Report progress to keep SSE connection alive."""
            await ctx.report_progress(pct, 100)

        try:
            await ctx.report_progress(0, 100)
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
            # Use simple or advanced pipeline based on config
            if app.config.agent_mode == "advanced":
                result = await app.pipeline.generate_advanced(
                    prompt=prompt, params=params, progress_callback=progress_callback
                )
            else:
                result = await app.pipeline.generate(
                    prompt=prompt, params=params, progress_callback=progress_callback
                )
            await ctx.report_progress(100, 100)

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"GENERATE_ANIMATION SUCCEEDED in {elapsed:.2f}s")
            logger.info(f"  Render ID: {result.render_id}")
            logger.info(f"  URL: {result.url}")
            logger.info("=" * 60)

            return result.model_dump()
        except ManimMCPError as e:
            elapsed = time.time() - start_time
            logger.error(f"GENERATE_ANIMATION FAILED after {elapsed:.2f}s: {e}")
            return {"error": True, "message": str(e)}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"GENERATE_ANIMATION ERROR after {elapsed:.2f}s: {e}")
            return {"error": True, "message": f"Unexpected error: {e}"}

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

        Returns:
            New animation result with updated video URL and new render ID.
        """
        start_time = time.time()
        instructions_preview = instructions[:100] + "..." if len(instructions) > 100 else instructions

        logger.info("=" * 60)
        logger.info("EDIT_ANIMATION TOOL CALLED")
        logger.info(f"  Render ID: {render_id}")
        logger.info(f"  Instructions: {instructions_preview}")
        logger.info(f"  Quality: {quality}, Format: {format}")
        logger.info("=" * 60)

        app = ctx.request_context.lifespan_context

        async def progress_callback(stage: str, pct: int):
            """Report progress to keep SSE connection alive."""
            await ctx.report_progress(pct, 100)

        try:
            await ctx.report_progress(0, 100)
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
            result = await app.pipeline.edit(
                render_id=render_id,
                instructions=instructions,
                params=params,
                progress_callback=progress_callback,
            )
            await ctx.report_progress(100, 100)

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"EDIT_ANIMATION SUCCEEDED in {elapsed:.2f}s")
            logger.info(f"  New Render ID: {result.render_id}")
            logger.info(f"  URL: {result.url}")
            logger.info("=" * 60)

            return result.model_dump()
        except ManimMCPError as e:
            elapsed = time.time() - start_time
            logger.error(f"EDIT_ANIMATION FAILED after {elapsed:.2f}s: {e}")
            return {"error": True, "message": str(e)}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"EDIT_ANIMATION ERROR after {elapsed:.2f}s: {e}")
            return {"error": True, "message": f"Unexpected error: {e}"}
