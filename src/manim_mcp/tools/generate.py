"""Generation tools: generate_animation, edit_animation."""

from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from manim_mcp.core.pipeline import RenderParams
from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality


def register_generate_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    async def generate_animation(
        ctx: Context,
        prompt: str,
        quality: str = "medium",
        format: str = "mp4",
        resolution: str | None = None,
        fps: int | None = None,
        background_color: str | None = None,
        transparent: bool = False,
        save_last_frame: bool = False,
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

        Returns:
            Animation result with video URL, render ID (for editing), and generated source code.
        """
        app = ctx.request_context.lifespan_context
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
            )
            result = await app.pipeline.generate(prompt=prompt, params=params)
            await ctx.report_progress(100, 100)
            return result.model_dump()
        except ManimMCPError as e:
            return {"error": True, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {e}"}

    @mcp.tool()
    async def edit_animation(
        ctx: Context,
        render_id: str,
        instructions: str,
        quality: str = "medium",
        format: str = "mp4",
        resolution: str | None = None,
        fps: int | None = None,
        background_color: str | None = None,
        transparent: bool = False,
        save_last_frame: bool = False,
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

        Returns:
            New animation result with updated video URL and new render ID.
        """
        app = ctx.request_context.lifespan_context
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
            )
            result = await app.pipeline.edit(
                render_id=render_id,
                instructions=instructions,
                params=params,
            )
            await ctx.report_progress(100, 100)
            return result.model_dump()
        except ManimMCPError as e:
            return {"error": True, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {e}"}
