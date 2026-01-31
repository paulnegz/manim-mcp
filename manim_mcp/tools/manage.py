"""Management tools: list_renders, get_render, delete_render."""

from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import RenderStatus


def register_manage_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    async def list_renders(
        ctx: Context,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict:
        """List past animations with pagination and fresh URLs.

        Args:
            limit: Maximum results (1-100).
            offset: Pagination offset.
            status: Filter by status: pending, generating, rendering, uploading, completed, failed.
        """
        app = ctx.request_context.lifespan_context
        try:
            status_enum = RenderStatus(status) if status else None
            renders = await app.tracker.list_renders(
                limit=min(max(limit, 1), 100),
                offset=max(offset, 0),
                status=status_enum,
            )

            # Regenerate fresh presigned URLs for each render
            results = []
            for r in renders:
                render_dict = r.model_dump()
                if r.s3_object_key:
                    presigned = await app.storage.generate_presigned_url(r.s3_object_key)
                    render_dict["presigned_url"] = presigned
                if r.thumbnail_s3_key:
                    thumb_url = await app.storage.generate_presigned_url(r.thumbnail_s3_key)
                    render_dict["thumbnail_url"] = thumb_url
                results.append(render_dict)

            return {"renders": results, "count": len(results)}
        except ManimMCPError as e:
            return {"error": True, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {e}"}

    @mcp.tool()
    async def get_render(ctx: Context, render_id: str) -> dict:
        """Get full details for a render, including fresh download URLs.

        Args:
            render_id: The render job ID.
        """
        app = ctx.request_context.lifespan_context
        try:
            metadata = await app.tracker.get_render(render_id)
            result = metadata.model_dump()

            # Regenerate fresh presigned URL for video
            if metadata.s3_object_key:
                presigned = await app.storage.generate_presigned_url(metadata.s3_object_key)
                result["presigned_url"] = presigned

            # Regenerate fresh presigned URL for thumbnail
            if metadata.thumbnail_s3_key:
                thumb_url = await app.storage.generate_presigned_url(metadata.thumbnail_s3_key)
                result["thumbnail_url"] = thumb_url
            elif metadata.s3_object_key:
                # Fallback: try standard thumbnail path if key not stored
                thumb_key = metadata.s3_object_key.rsplit("/", 1)[0] + "/thumbnail.png"
                if await app.storage.object_exists(thumb_key):
                    thumb_url = await app.storage.generate_presigned_url(thumb_key)
                    result["thumbnail_url"] = thumb_url

            return result
        except ManimMCPError as e:
            return {"error": True, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {e}"}

    @mcp.tool()
    async def delete_render(ctx: Context, render_id: str) -> dict:
        """Delete an animation and its files. Irreversible.

        Args:
            render_id: The render job ID to delete.
        """
        app = ctx.request_context.lifespan_context
        try:
            metadata = await app.tracker.get_render(render_id)

            if metadata.s3_object_key:
                await app.storage.delete_object(metadata.s3_object_key)
                thumb_key = metadata.s3_object_key.rsplit("/", 1)[0] + "/thumbnail.png"
                await app.storage.delete_object(thumb_key)

            await app.tracker.delete_render(render_id)
            return {"deleted": True, "render_id": render_id}
        except ManimMCPError as e:
            return {"error": True, "message": str(e)}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {e}"}
