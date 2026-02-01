"""Shared initialization context used by both the MCP server and the CLI."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from manim_mcp.config import ManimMCPConfig
from manim_mcp.core.llm import BaseLLMClient, create_llm_client
from manim_mcp.core.pipeline import AnimationPipeline
from manim_mcp.core.rag import ChromaDBService
from manim_mcp.core.renderer import ManimRenderer
from manim_mcp.core.sandbox import CodeSandbox
from manim_mcp.core.scene_parser import SceneParser
from manim_mcp.core.storage import S3Storage
from manim_mcp.core.tracker import RenderTracker
from manim_mcp.core.tts import GeminiTTSService
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
    llm: BaseLLMClient
    rag: ChromaDBService


def _check_llm_api_key(config: ManimMCPConfig) -> None:
    """Validate that the required LLM API key is set."""
    provider = config.llm_provider.lower()
    if provider == "claude":
        if not config.claude_api_key:
            raise RuntimeError(
                "MANIM_MCP_CLAUDE_API_KEY must be set when using Claude provider. "
                "Get an API key at: https://console.anthropic.com/"
            )
    elif provider == "deepseek":
        if not config.deepseek_api_key:
            raise RuntimeError(
                "MANIM_MCP_DEEPSEEK_API_KEY must be set when using DeepSeek provider. "
                "Get an API key at: https://platform.deepseek.com/"
            )
    else:  # gemini (default)
        if not config.gemini_api_key:
            raise RuntimeError(
                "MANIM_MCP_GEMINI_API_KEY must be set when using Gemini provider. "
                "Get an API key at: https://ai.google.dev/"
            )


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
            "Install with: pip install manimgl"
        )

    _check_llm_api_key(config)

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found on PATH — some renders may fail")

    if not shutil.which("latex"):
        logger.warning("LaTeX not found on PATH — LLM will avoid LaTeX-dependent objects")
        config.latex_available = False
    else:
        config.latex_available = True

    # Initialize components with error handling
    storage = S3Storage(config)
    await storage.initialize()

    tracker = RenderTracker(config)
    await tracker.initialize()

    rag = ChromaDBService(config)
    await rag.initialize()

    sandbox = CodeSandbox(config)
    scene_parser = SceneParser()

    # Create LLM client based on provider
    try:
        llm = create_llm_client(config)
    except ImportError as e:
        logger.error("Failed to initialize LLM client: %s", e)
        raise RuntimeError(f"LLM initialization failed: {e}")

    renderer = ManimRenderer(config, scene_parser)

    # Initialize TTS service if Gemini API key is available
    tts = None
    if config.gemini_api_key:
        try:
            tts = GeminiTTSService(config)
            logger.info("TTS service initialized (voice: %s)", config.tts_voice)
        except Exception as e:
            logger.warning("Failed to initialize TTS service: %s", e)

    pipeline = AnimationPipeline(
        config, llm, renderer, sandbox, scene_parser, tracker, storage, rag, tts
    )

    ctx = AppContext(
        config=config,
        pipeline=pipeline,
        renderer=renderer,
        storage=storage,
        tracker=tracker,
        sandbox=sandbox,
        scene_parser=scene_parser,
        llm=llm,
        rag=rag,
    )

    # Determine which model name to show
    if config.llm_provider == "claude":
        model_name = config.claude_model
    elif config.llm_provider == "deepseek":
        model_name = config.deepseek_model
    else:
        model_name = config.gemini_model

    logger.info(
        "manim-mcp started (LLM: %s/%s, S3: %s, RAG: %s, max concurrent: %d)",
        config.llm_provider,
        model_name,
        "available" if storage.available else "degraded",
        "available" if rag.available else "degraded",
        config.max_concurrent_renders,
    )

    try:
        yield ctx
    finally:
        await tracker.close()
        logger.info("manim-mcp stopped")
