"""MCP server for text-to-Manim-video generation.

Runs the FastMCP server with an additional SSE progress endpoint
mounted at /progress/{render_id} for real-time render tracking.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from manim_mcp.bootstrap import AppContext, app_context
from manim_mcp.progress import format_sse, get_broadcaster
from manim_mcp.tools import register_all_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    async with app_context() as ctx:
        yield ctx


def create_server() -> FastMCP:
    host = os.environ.get("MANIM_MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("MANIM_MCP_SERVER_PORT", "8000"))
    mcp = FastMCP(
        "manim-mcp",
        lifespan=app_lifespan,
        host=host,
        port=port,
        stateless_http=True,  # Allow stateless HTTP requests from live service
    )
    register_all_tools(mcp)
    return mcp


# ── SSE progress endpoint ────────────────────────────────────────────

async def _sse_progress_endpoint(request):
    """Stream SSE progress events for a given render_id.

    GET /progress/{render_id}
    Content-Type: text/event-stream

    Events:
        progress  - percentage + friendly message
        thumbnail - early thumbnail URL
        complete  - final result with video URL
        error     - render failure
    """
    from starlette.responses import StreamingResponse

    render_id = request.path_params["render_id"]
    broadcaster = get_broadcaster()

    if not broadcaster.has_render(render_id):
        from starlette.responses import JSONResponse
        return JSONResponse(
            {"error": "Render not found", "render_id": render_id},
            status_code=404,
        )

    async def event_generator():
        async for event in broadcaster.subscribe(render_id):
            yield format_sse(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def _health_endpoint(request):
    """Simple health check endpoint."""
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "service": "manim-mcp"})


def create_app():
    """Create a Starlette ASGI app that combines the MCP server with SSE routes.

    The MCP server handles its own routes (/mcp by default).
    Additional routes:
        GET /health                  - health check
        GET /progress/{render_id}    - SSE progress stream
    """
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount

    mcp = create_server()

    # Call streamable_http_app() to initialize the session manager,
    # but manage its lifecycle from the outer Starlette app's lifespan
    # (mounted sub-app lifespans don't run).
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    routes = [
        Route("/health", _health_endpoint, methods=["GET"]),
        Route("/progress/{render_id}", _sse_progress_endpoint, methods=["GET"]),
        Mount("/", app=mcp_app),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    return app


def main():
    """Start the server with SSE progress support.

    When the MANIM_MCP_SSE_ENABLED env var is set (default: true),
    the server starts with the combined Starlette app that includes
    both MCP and SSE endpoints. Otherwise, it runs the plain MCP server.
    """
    sse_enabled = os.environ.get("MANIM_MCP_SSE_ENABLED", "true").lower() in ("true", "1", "yes")

    if sse_enabled:
        import uvicorn
        host = os.environ.get("MANIM_MCP_SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("MANIM_MCP_SERVER_PORT", "8000"))
        app = create_app()
        logger.info("Starting manim-mcp with SSE progress on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port)
    else:
        server = create_server()
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
