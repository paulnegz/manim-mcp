"""Async Manim render pipeline — takes validated code, produces a video file."""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manim_mcp.exceptions import (
    ConcurrencyLimitError,
    CodeValidationError,
    FileSizeLimitError,
    OutputNotFoundError,
    RenderError,
    RenderTimeoutError,
)
from manim_mcp.models import RenderSceneInput

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig
    from manim_mcp.core.scene_parser import SceneParser

logger = logging.getLogger(__name__)


@dataclass
class RenderOutput:
    """Raw render output — the pipeline decides what to do with it."""
    scene_name: str
    local_path: str
    file_size_bytes: int
    render_time_seconds: float
    width: int | None
    height: int | None
    format: str
    quality: str

    # Filled in by pipeline after upload
    s3_url: str | None = None
    s3_object_key: str | None = None
    url: str | None = None
    thumbnail_url: str | None = None

    @property
    def resolution(self) -> str | None:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None


class ManimRenderer:
    def __init__(
        self,
        config: ManimMCPConfig,
        scene_parser: SceneParser,
    ) -> None:
        self.config = config
        self.scene_parser = scene_parser
        self._semaphore = asyncio.Semaphore(config.max_concurrent_renders)

    async def render_scene(self, input: RenderSceneInput) -> RenderOutput:
        """Render validated Manim code. Returns local file path + metadata."""
        scenes = self.scene_parser.parse_scenes(input.code)
        scene_name = self._resolve_scene_name(scenes, input.scene_name)

        if self._semaphore._value == 0:
            raise ConcurrencyLimitError(
                f"Maximum concurrent renders ({self.config.max_concurrent_renders}) reached. Try again later."
            )

        async with self._semaphore:
            return await self._execute_render(input, scene_name)

    def _resolve_scene_name(self, scenes, requested_name: str | None) -> str:
        if not scenes:
            raise CodeValidationError("No Scene subclass found in the provided code")
        if len(scenes) == 1:
            return scenes[0].name
        if requested_name:
            for s in scenes:
                if s.name == requested_name:
                    return s.name
            names = ", ".join(s.name for s in scenes)
            raise CodeValidationError(f"Scene '{requested_name}' not found. Available: {names}")
        names = ", ".join(s.name for s in scenes)
        raise CodeValidationError(f"Multiple scenes found. Please specify scene_name. Available: {names}")

    async def _execute_render(self, input: RenderSceneInput, scene_name: str) -> RenderOutput:
        tmp_dir = tempfile.mkdtemp(prefix="manim_mcp_")
        start_time = time.monotonic()

        try:
            scene_file = os.path.join(tmp_dir, "scene.py")
            with open(scene_file, "w") as f:
                f.write(input.code)

            cmd = self._build_command(input, scene_name, tmp_dir, scene_file)
            logger.info("Rendering: %s", " ".join(cmd))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.render_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RenderTimeoutError(f"Render timed out after {self.config.render_timeout}s")

            if proc.returncode != 0:
                stderr_text = stderr.decode(errors="replace")[-2000:]
                raise RenderError(f"Manim exited with code {proc.returncode}: {stderr_text}")

            output_path = self._find_output(tmp_dir, scene_name, input)
            if not output_path:
                stderr_text = stderr.decode(errors="replace")[-2000:]
                raise OutputNotFoundError(f"Output file not found. stderr: {stderr_text}")

            file_size = os.path.getsize(output_path)
            max_bytes = self.config.s3_max_upload_size_mb * 1024 * 1024
            if file_size > max_bytes:
                raise FileSizeLimitError(
                    f"Output ({file_size} bytes) exceeds {self.config.s3_max_upload_size_mb}MB limit"
                )

            render_time = round(time.monotonic() - start_time, 2)
            ext = os.path.splitext(output_path)[1].lstrip(".")
            width, height = self._parse_resolution(input)

            return RenderOutput(
                scene_name=scene_name,
                local_path=output_path,
                file_size_bytes=file_size,
                render_time_seconds=render_time,
                width=width,
                height=height,
                format=ext,
                quality=input.quality.value,
            )

        except (ConcurrencyLimitError, RenderTimeoutError, RenderError,
                OutputNotFoundError, FileSizeLimitError, CodeValidationError):
            raise
        except Exception as e:
            raise RenderError(f"Unexpected render failure: {e}") from e

    def _build_command(
        self, input: RenderSceneInput, scene_name: str, tmp_dir: str, scene_file: str,
    ) -> list[str]:
        cmd = [
            self.config.manim_executable, "render",
            f"-q{input.quality.flag}",
            "--format", input.format.value,
            "--media_dir", os.path.join(tmp_dir, "media"),
            "--disable_caching",
        ]
        if input.fps:
            cmd.extend(["--fps", str(input.fps)])
        if input.background_color:
            cmd.extend(["-c", input.background_color])
        if input.transparent:
            cmd.append("-t")
        if input.resolution:
            cmd.extend(["-r", input.resolution])
        if input.save_last_frame:
            cmd.append("-s")
        cmd.extend([scene_file, scene_name])
        return cmd

    def _find_output(self, tmp_dir: str, scene_name: str, input: RenderSceneInput) -> str | None:
        media_dir = os.path.join(tmp_dir, "media")
        safe_name = re.sub(r'[^\w]', '', scene_name)
        ext = input.format.value

        if input.save_last_frame:
            for pattern in [
                os.path.join(media_dir, "images", "scene", f"{scene_name}*.png"),
                os.path.join(media_dir, "images", "**", f"{scene_name}*.png"),
                os.path.join(media_dir, "images", "**", f"{safe_name}*.png"),
            ]:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    return matches[0]

        for pattern in [
            os.path.join(media_dir, "videos", "scene", "**", f"{scene_name}.{ext}"),
            os.path.join(media_dir, "videos", "**", f"{scene_name}.{ext}"),
            os.path.join(media_dir, "videos", "**", f"{safe_name}.{ext}"),
            os.path.join(media_dir, "**", f"*.{ext}"),
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]

        for root, _, files in os.walk(media_dir):
            for f in files:
                if f.endswith(f".{ext}") or (input.save_last_frame and f.endswith(".png")):
                    return os.path.join(root, f)
        return None

    def _parse_resolution(self, input: RenderSceneInput) -> tuple[int | None, int | None]:
        if input.resolution:
            parts = input.resolution.lower().split("x")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        defaults = {
            "low": (854, 480), "medium": (1280, 720), "high": (1920, 1080),
            "production": (2560, 1440), "fourk": (3840, 2160),
        }
        return defaults.get(input.quality.value, (None, None))
