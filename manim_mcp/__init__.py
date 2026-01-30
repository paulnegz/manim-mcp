"""Manim MCP Server — text-to-video animation via the Model Context Protocol."""

from manim_mcp.cli import cli_main
from manim_mcp.server import create_server, main

__all__ = ["cli_main", "create_server", "main"]
