"""Shared initialization context used by both the MCP server and the CLI."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.llm import GeminiClient
from manim_mcp.core.pipeline import AnimationPipeline
from manim_mcp.core.renderer import ManimRenderer
from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.core.storage import S3Storage
from manim_mcp.core.tracker import RenderTracker
from manim_mcp.exceptions import ManimNotInstalledError

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    config: ManimMCPConfig
    pipeline: AnimationPipeline
    renderer: ManimRenderer
    storage: S3Storage
    tracker: RenderTracker
    sandbox: CodeSandbox
    scene_parser: SceneParser
    llm: GeminiClient


@asynccontextmanager
async def app_context(config: ManimMCPConfig | None = None) -> AsyncIterator[AppContext]:
    """Bootstrap all application components and yield an AppContext.

    Used by both the MCP server lifespan and the CLI commands.
    """
    config = config or ManimMCPConfig()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    # Startup checks
    if not shutil.which(config.manim_executable):
        raise ManimNotInstalledError(
            f"Manim executable '{config.manim_executable}' not found on PATH. "
            "Install with: pip install manim"
        )

    if not config.gemini_api_key:
        raise RuntimeError("MANIM_MCP_GEMINI_API_KEY must be set")

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found on PATH — some renders may fail")

    if not shutil.which("latex"):
        logger.warning("LaTeX not found on PATH — LLM will avoid LaTeX-dependent objects")
        config.latex_available = False
    else:
        config.latex_available = True

    # Initialize components
    storage = S3Storage(config)
    await storage.initialize()

    tracker = RenderTracker(config)
    await tracker.initialize()

    sandbox = CodeSandbox(config)
    scene_parser = SceneParser()
    llm = GeminiClient(config)
    renderer = ManimRenderer(config, scene_parser)
    pipeline = AnimationPipeline(config, llm, renderer, sandbox, scene_parser, tracker, storage)

    ctx = AppContext(
        config=config,
        pipeline=pipeline,
        renderer=renderer,
        storage=storage,
        tracker=tracker,
        sandbox=sandbox,
        scene_parser=scene_parser,
        llm=llm,
    )

    logger.info(
        "manim-mcp started (S3: %s, max concurrent: %d, Gemini model: %s)",
        "available" if storage.available else "degraded",
        config.max_concurrent_renders,
        config.gemini_model,
    )

    try:
        yield ctx
    finally:
        await tracker.close()
        logger.info("manim-mcp stopped")
