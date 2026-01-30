"""Async command handlers for each CLI subcommand."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from manim_mcp.bootstrap import AppContext, app_context
from manim_mcp.cli.output import Printer, spinner
from manim_mcp.core.pipeline import RenderParams
from manim_mcp.exceptions import ManimMCPError
from manim_mcp.models import OutputFormat, RenderQuality, RenderStatus

if TYPE_CHECKING:
    pass


# ── Dispatcher ────────────────────────────────────────────────────────

async def dispatch(args: argparse.Namespace) -> int:
    """Bootstrap the app, create a printer, and route to the right handler."""
    printer = Printer(json_mode=getattr(args, "json", False))

    try:
        async with app_context() as ctx:
            handler = _HANDLERS.get(args.command)
            if handler is None:
                printer.error(f"Unknown command: {args.command}")
                return 1
            return await handler(args, ctx, printer)
    except ManimMCPError as e:
        printer.error(str(e))
        return 1
    except Exception as e:
        printer.error(f"Unexpected error: {e}")
        return 1


# ── Individual commands ───────────────────────────────────────────────

async def cmd_generate(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    params = _params_from_args(args)
    mode = getattr(args, "mode", "simple")
    printer.info(f"Generating animation ({mode} mode): {args.prompt}")

    with spinner("Generating animation"):
        if mode == "advanced":
            result = await ctx.pipeline.generate_advanced(prompt=args.prompt, params=params)
        else:
            result = await ctx.pipeline.generate(prompt=args.prompt, params=params)

    printer.animation_result(result.model_dump())
    return 0


async def cmd_edit(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    params = _params_from_args(args)
    printer.info(f"Editing render {args.render_id}: {args.instructions}")

    with spinner("Editing animation"):
        result = await ctx.pipeline.edit(
            render_id=args.render_id,
            instructions=args.instructions,
            params=params,
        )

    printer.animation_result(result.model_dump())
    return 0


async def cmd_list(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    status_enum = RenderStatus(args.status) if args.status else None
    limit = min(max(args.limit, 1), 100)

    with spinner("Fetching renders"):
        renders = await ctx.tracker.list_renders(
            limit=limit, offset=args.offset, status=status_enum,
        )

    data = [r.model_dump() for r in renders]
    printer.render_list(data, len(data))
    return 0


async def cmd_get(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    with spinner("Fetching render"):
        metadata = await ctx.tracker.get_render(args.render_id)

    detail = metadata.model_dump()
    if metadata.s3_object_key:
        presigned = await ctx.storage.generate_presigned_url(metadata.s3_object_key)
        detail["presigned_url"] = presigned

    printer.render_detail(detail)
    return 0


async def cmd_delete(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    if not args.yes:
        try:
            answer = input(f"Delete render {args.render_id}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            printer.info("Cancelled.")
            return 0
        if answer not in ("y", "yes"):
            printer.info("Cancelled.")
            return 0

    with spinner("Deleting render"):
        metadata = await ctx.tracker.get_render(args.render_id)
        if metadata.s3_object_key:
            await ctx.storage.delete_object(metadata.s3_object_key)
            thumb_key = metadata.s3_object_key.rsplit("/", 1)[0] + "/thumbnail.png"
            await ctx.storage.delete_object(thumb_key)
        await ctx.tracker.delete_render(args.render_id)

    printer.success(f"Deleted render {args.render_id}")
    return 0


async def cmd_prompt(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    from manim_mcp.cli.agent import run_agent_loop

    return await run_agent_loop(
        prompt=args.prompt,
        ctx=ctx,
        printer=printer,
        max_turns=args.max_turns,
    )


async def cmd_index(
    args: argparse.Namespace,
    ctx: AppContext,
    printer: Printer,
) -> int:
    from manim_mcp.cli.indexer import cmd_index as indexer_cmd
    return await indexer_cmd(args, ctx, printer)


# ── Helpers ───────────────────────────────────────────────────────────

def _params_from_args(args: argparse.Namespace) -> RenderParams:
    return RenderParams(
        quality=RenderQuality(getattr(args, "quality", "medium")),
        fmt=OutputFormat(getattr(args, "format", "mp4")),
        resolution=getattr(args, "resolution", None),
        fps=getattr(args, "fps", None),
        background_color=getattr(args, "background_color", None),
        transparent=getattr(args, "transparent", False),
        save_last_frame=getattr(args, "save_last_frame", False),
        audio=getattr(args, "audio", False),
        voice=getattr(args, "voice", None),
    )


_HANDLERS = {
    "generate": cmd_generate,
    "gen": cmd_generate,
    "edit": cmd_edit,
    "list": cmd_list,
    "ls": cmd_list,
    "get": cmd_get,
    "delete": cmd_delete,
    "rm": cmd_delete,
    "prompt": cmd_prompt,
    "index": cmd_index,
}
