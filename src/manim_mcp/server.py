"""MCP server for text-to-Manim-video generation."""

from __future__ import annotations

import logging
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
    mcp = FastMCP("manim-mcp", lifespan=app_lifespan)
    register_all_tools(mcp)
    return mcp


def main():
    server = create_server()
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
