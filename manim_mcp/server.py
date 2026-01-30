"""MCP server for text-to-Manim-video generation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from manim_mcp.bootstrap import AppContext, app_context
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


def main():
    server = create_server()
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
