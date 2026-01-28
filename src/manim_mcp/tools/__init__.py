"""Tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_all_tools(mcp: FastMCP) -> None:
    from manim_mcp.tools.generate import register_generate_tools
    from manim_mcp.tools.manage import register_manage_tools

    register_generate_tools(mcp)
    register_manage_tools(mcp)
