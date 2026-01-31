"""Gemini FunctionDeclaration schemas and execute_tool() dispatch for the CLI agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai.types import FunctionDeclaration, Schema, Type

from manim_mcp.core.pipeline import RenderParams
from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality, RenderStatus

if TYPE_CHECKING:
    from manim_mcp.bootstrap import AppContext

# ── FunctionDeclaration objects ───────────────────────────────────────

TOOL_DECLARATIONS = [
    FunctionDeclaration(
        name="generate_animation",
        description=(
            "Create a Manim animation from a text description. "
            "Returns a video URL, render ID, and generated source code."
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "prompt": Schema(
                    type=Type.STRING,
                    description="Text description of the animation to create.",
                ),
                "quality": Schema(
                    type=Type.STRING,
                    description="Render quality (default: low). Options: low, medium, high, production, fourk. Only specify if user explicitly requests higher quality.",
                ),
                "format": Schema(
                    type=Type.STRING,
                    description="Output format: mp4, gif, webm, mov, png.",
                ),
            },
            required=["prompt"],
        ),
    ),
    FunctionDeclaration(
        name="edit_animation",
        description=(
            "Edit an existing animation by describing what to change. "
            "Pass the render_id from a previous result and natural-language instructions."
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "render_id": Schema(
                    type=Type.STRING,
                    description="ID of the animation to edit.",
                ),
                "instructions": Schema(
                    type=Type.STRING,
                    description="What to change (e.g. 'make the circle red').",
                ),
                "quality": Schema(
                    type=Type.STRING,
                    description="Render quality (default: low). Options: low, medium, high, production, fourk. Only specify if user explicitly requests higher quality.",
                ),
                "format": Schema(
                    type=Type.STRING,
                    description="Output format: mp4, gif, webm, mov, png.",
                ),
            },
            required=["render_id", "instructions"],
        ),
    ),
    FunctionDeclaration(
        name="list_renders",
        description="List past animations with optional status filter and pagination.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "limit": Schema(
                    type=Type.INTEGER,
                    description="Maximum number of results (1-100).",
                ),
                "offset": Schema(
                    type=Type.INTEGER,
                    description="Pagination offset.",
                ),
                "status": Schema(
                    type=Type.STRING,
                    description="Filter by status: pending, generating, rendering, uploading, completed, failed.",
                ),
            },
        ),
    ),
    FunctionDeclaration(
        name="get_render",
        description="Get full details for a render, including a fresh download URL.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "render_id": Schema(
                    type=Type.STRING,
                    description="The render job ID.",
                ),
            },
            required=["render_id"],
        ),
    ),
    FunctionDeclaration(
        name="delete_render",
        description="Permanently delete an animation and its files.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "render_id": Schema(
                    type=Type.STRING,
                    description="The render job ID to delete.",
                ),
            },
            required=["render_id"],
        ),
    ),
]


# ── Dispatch ──────────────────────────────────────────────────────────

async def execute_tool(name: str, args: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
    """Execute a tool call against the AppContext and return a JSON-serialisable dict."""
    try:
        if name == "generate_animation":
            params = _build_render_params(args)
            result = await ctx.pipeline.generate(prompt=args["prompt"], params=params)
            return result.model_dump()

        if name == "edit_animation":
            params = _build_render_params(args)
            result = await ctx.pipeline.edit(
                render_id=args["render_id"],
                instructions=args["instructions"],
                params=params,
            )
            return result.model_dump()

        if name == "list_renders":
            status_str = args.get("status")
            status_enum = RenderStatus(status_str) if status_str else None
            limit = min(max(args.get("limit", 20), 1), 100)
            offset = max(args.get("offset", 0), 0)
            renders = await ctx.tracker.list_renders(
                limit=limit, offset=offset, status=status_enum,
            )
            return {"renders": [r.model_dump() for r in renders], "count": len(renders)}

        if name == "get_render":
            metadata = await ctx.tracker.get_render(args["render_id"])
            result_dict = metadata.model_dump()
            if metadata.s3_object_key:
                presigned = await ctx.storage.generate_presigned_url(metadata.s3_object_key)
                result_dict["presigned_url"] = presigned
            return result_dict

        if name == "delete_render":
            metadata = await ctx.tracker.get_render(args["render_id"])
            if metadata.s3_object_key:
                await ctx.storage.delete_object(metadata.s3_object_key)
                thumb_key = metadata.s3_object_key.rsplit("/", 1)[0] + "/thumbnail.png"
                await ctx.storage.delete_object(thumb_key)
            await ctx.tracker.delete_render(args["render_id"])
            return {"deleted": True, "render_id": args["render_id"]}

        return {"error": True, "message": f"Unknown tool: {name}"}

    except ManimMCPError as e:
        return {"error": True, "message": str(e)}
    except Exception as e:
        return {"error": True, "message": f"Unexpected error: {e}"}


def _build_render_params(args: dict[str, Any]) -> RenderParams:
    quality_str = args.get("quality", "low")
    fmt_str = args.get("format", "mp4")
    return RenderParams(
        quality=RenderQuality(quality_str),
        fmt=OutputFormat(fmt_str),
    )
