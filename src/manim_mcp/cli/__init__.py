"""CLI entry point for manim-mcp."""

from __future__ import annotations

import argparse
import asyncio
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manim-mcp",
        description="Generate Manim animations from text — CLI, agent, or MCP server.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate (alias: gen) ────────────────────────────────────
    gen = sub.add_parser("generate", aliases=["gen"], help="Generate an animation from a prompt")
    gen.add_argument("prompt", help="Text description of the animation")
    _add_render_flags(gen)

    # ── edit ──────────────────────────────────────────────────────
    edit = sub.add_parser("edit", help="Edit an existing animation")
    edit.add_argument("render_id", help="Render ID to edit")
    edit.add_argument("instructions", help="What to change")
    _add_render_flags(edit)

    # ── list (alias: ls) ─────────────────────────────────────────
    ls = sub.add_parser("list", aliases=["ls"], help="List past renders")
    ls.add_argument("--status", choices=[
        "pending", "generating", "rendering", "uploading", "completed", "failed",
    ], help="Filter by status")
    ls.add_argument("--limit", type=int, default=20, help="Max results (default 20)")
    ls.add_argument("--offset", type=int, default=0, help="Pagination offset")

    # ── get ───────────────────────────────────────────────────────
    get = sub.add_parser("get", help="Get details for a render")
    get.add_argument("render_id", help="Render ID")

    # ── delete (alias: rm) ───────────────────────────────────────
    rm = sub.add_parser("delete", aliases=["rm"], help="Delete a render")
    rm.add_argument("render_id", help="Render ID to delete")
    rm.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # ── prompt (agent mode) ──────────────────────────────────────
    prompt = sub.add_parser("prompt", help="Agent mode — Gemini interprets and runs tools")
    prompt.add_argument("prompt", help="Natural-language request")
    prompt.add_argument("--max-turns", type=int, default=10, help="Max agent turns (default 10)")

    # ── serve ────────────────────────────────────────────────────
    serve = sub.add_parser("serve", help="Start the MCP server")
    serve.add_argument(
        "--transport",
        choices=["streamable-http", "stdio", "sse"],
        default="streamable-http",
        help="MCP transport (default: streamable-http)",
    )

    return parser


def _add_render_flags(parser: argparse.ArgumentParser) -> None:
    """Add shared render-parameter flags to generate/edit subparsers."""
    parser.add_argument("--quality", "-q", default="medium", choices=[
        "low", "medium", "high", "production", "fourk",
    ], help="Render quality (default: medium)")
    parser.add_argument("--format", "-f", default="mp4", choices=[
        "mp4", "gif", "webm", "mov", "png",
    ], help="Output format (default: mp4)")
    parser.add_argument("--resolution", default=None, help="Resolution as WxH (e.g. 1920x1080)")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second override")
    parser.add_argument("--background-color", default=None, help="Background color hex")
    parser.add_argument("--transparent", action="store_true", help="Transparent background")
    parser.add_argument("--save-last-frame", action="store_true", help="Save last frame as image")


def cli_main() -> None:
    """Entry point for the ``manim-mcp`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    # --verbose → DEBUG logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # serve is special — it hands off to the MCP server
    if args.command == "serve":
        import os
        from manim_mcp.server import create_server
        server = create_server()
        # FastMCP uses UVICORN_HOST/UVICORN_PORT env vars (set in docker-compose)
        server.run(transport=args.transport)
        return

    # Everything else runs through the async dispatcher
    from manim_mcp.cli.commands import dispatch

    try:
        exit_code = asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        sys.exit(130)

    sys.exit(exit_code)
