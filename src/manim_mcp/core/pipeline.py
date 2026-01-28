"""Text-to-video pipeline: prompt → Gemini → validate → render → upload."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from manim_mcp.exceptions import CodeValidationError, ManimMCPError
from manim_mcp.models import (
    AnimationResult,
    OutputFormat,
    RenderQuality,
    RenderSceneInput,
    RenderStatus,
)

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig
    from manim_mcp.core.llm import GeminiClient
    from manim_mcp.core.renderer import ManimRenderer, RenderOutput
    from manim_mcp.core.sandbox import CodeSandbox
    from manim_mcp.core.scene_parser import SceneParser
    from manim_mcp.core.storage import S3Storage
    from manim_mcp.core.tracker import RenderTracker

logger = logging.getLogger(__name__)


@dataclass
class RenderParams:
    """Extra render parameters passed through from the tool layer."""
    quality: RenderQuality = RenderQuality.medium
    fmt: OutputFormat = OutputFormat.mp4
    resolution: str | None = None
    fps: int | None = None
    background_color: str | None = None
    transparent: bool = False
    save_last_frame: bool = False


class AnimationPipeline:
    def __init__(
        self,
        config: ManimMCPConfig,
        llm: GeminiClient,
        renderer: ManimRenderer,
        sandbox: CodeSandbox,
        scene_parser: SceneParser,
        tracker: RenderTracker,
        storage: S3Storage,
    ) -> None:
        self.config = config
        self.llm = llm
        self.renderer = renderer
        self.sandbox = sandbox
        self.scene_parser = scene_parser
        self.tracker = tracker
        self.storage = storage

    # ── Public API ─────────────────────────────────────────────────────

    async def generate(self, prompt: str, params: RenderParams | None = None) -> AnimationResult:
        params = params or RenderParams()
        render_id = uuid.uuid4().hex[:12]
        await self.tracker.create_render(
            render_id, quality=params.quality.value, fmt=params.fmt.value, original_prompt=prompt,
        )

        try:
            # 1. Generate code via Gemini
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            code = await self._generate_and_validate(prompt)

            # 2. Store generated code
            scenes = self.scene_parser.parse_scenes(code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                render_id, source_code=code, scene_name=scene_name,
            )

            # 3. Render + upload
            result = await self._render_and_upload(render_id, code, scene_name, params)

            return AnimationResult(
                render_id=render_id,
                status=RenderStatus.completed,
                url=result.url or result.local_path,
                thumbnail_url=result.thumbnail_url,
                format=result.format,
                quality=result.quality,
                file_size_bytes=result.file_size_bytes,
                resolution=result.resolution,
                render_time_seconds=result.render_time_seconds,
                prompt=prompt,
                source_code=code,
                message="Animation created successfully",
            )

        except Exception as e:
            await self.tracker.update_render(
                render_id, status=RenderStatus.failed, error_message=str(e)[:2000],
            )
            raise

    async def edit(
        self,
        render_id: str,
        instructions: str,
        params: RenderParams | None = None,
    ) -> AnimationResult:
        params = params or RenderParams()

        # 1. Get original render
        original = await self.tracker.get_render(render_id)
        if not original.source_code:
            raise ManimMCPError(f"Render '{render_id}' has no source code to edit")

        # 2. Create new render record linked to parent
        new_id = uuid.uuid4().hex[:12]
        await self.tracker.create_render(
            new_id,
            parent_render_id=render_id,
            quality=params.quality.value,
            fmt=params.fmt.value,
            original_prompt=original.original_prompt,
            edit_instructions=instructions,
        )

        try:
            # 3. Edit code via Gemini
            await self.tracker.update_render(new_id, status=RenderStatus.generating)
            edited_code = await self._edit_and_validate(original.source_code, instructions)

            # 4. Store edited code
            scenes = self.scene_parser.parse_scenes(edited_code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                new_id, source_code=edited_code, scene_name=scene_name,
            )

            # 5. Render + upload
            result = await self._render_and_upload(new_id, edited_code, scene_name, params)

            return AnimationResult(
                render_id=new_id,
                status=RenderStatus.completed,
                url=result.url or result.local_path,
                thumbnail_url=result.thumbnail_url,
                format=result.format,
                quality=result.quality,
                file_size_bytes=result.file_size_bytes,
                resolution=result.resolution,
                render_time_seconds=result.render_time_seconds,
                prompt=original.original_prompt,
                source_code=edited_code,
                message=f"Animation edited successfully (from {render_id})",
            )

        except Exception as e:
            await self.tracker.update_render(
                new_id, status=RenderStatus.failed, error_message=str(e)[:2000],
            )
            raise

    # ── Internal ───────────────────────────────────────────────────────

    async def _render_and_upload(
        self,
        render_id: str,
        code: str,
        scene_name: str | None,
        params: RenderParams,
    ):
        """Render code, upload to S3, update tracker, return RenderOutput."""
        from manim_mcp.core.renderer import RenderOutput

        await self.tracker.update_render(render_id, status=RenderStatus.rendering)

        render_input = RenderSceneInput(
            code=code,
            scene_name=scene_name,
            quality=params.quality,
            format=params.fmt,
            resolution=params.resolution,
            fps=params.fps,
            background_color=params.background_color,
            transparent=params.transparent,
            save_last_frame=params.save_last_frame,
        )
        output: RenderOutput = await self.renderer.render_scene(render_input)

        # Upload to S3
        if self.storage.available:
            await self.tracker.update_render(render_id, status=RenderStatus.uploading)
            filename = os.path.basename(output.local_path)
            s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
            try:
                output.s3_url = await self.storage.upload_file(output.local_path, s3_key)
                output.s3_object_key = s3_key
                output.url = await self.storage.generate_presigned_url(s3_key)
                output.thumbnail_url = await self._generate_thumbnail(
                    output.local_path, render_id,
                )
            except Exception as e:
                logger.warning("S3 upload failed, keeping local path: %s", e)

        # Update tracker
        await self.tracker.update_render(
            render_id,
            status=RenderStatus.completed,
            s3_url=output.s3_url,
            s3_object_key=output.s3_object_key,
            file_size_bytes=output.file_size_bytes,
            render_time_seconds=output.render_time_seconds,
            local_path=output.local_path,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        return output

    async def _generate_and_validate(self, prompt: str) -> str:
        code = await self.llm.generate_code(prompt)
        return await self._validate_with_retries(code)

    async def _edit_and_validate(self, original_code: str, instructions: str) -> str:
        code = await self.llm.edit_code(original_code, instructions)
        return await self._validate_with_retries(code)

    async def _validate_with_retries(self, code: str) -> str:
        for attempt in range(1, self.config.gemini_max_retries + 1):
            result = self.sandbox.validate(code)
            if result.valid:
                scenes = self.scene_parser.parse_scenes(code)
                if not scenes:
                    if attempt < self.config.gemini_max_retries:
                        code = await self.llm.fix_code(
                            code, ["No Scene subclass found. Code must define a class inheriting from Scene."]
                        )
                        continue
                    raise CodeValidationError("Generated code has no Scene subclass after retries")
                return code

            logger.warning("Validation failed (attempt %d/%d): %s",
                           attempt, self.config.gemini_max_retries, result.errors)
            if attempt < self.config.gemini_max_retries:
                code = await self.llm.fix_code(code, result.errors)
            else:
                raise CodeValidationError(
                    f"Code failed validation after {self.config.gemini_max_retries} attempts: "
                    + "; ".join(result.errors)
                )

        return code

    async def _generate_thumbnail(self, video_path: str, render_id: str) -> str | None:
        if not shutil.which("ffmpeg") or not self.storage.available:
            return None

        if video_path.endswith(".png"):
            thumb_key = f"{self.config.s3_prefix}{render_id}/thumbnail.png"
            try:
                await self.storage.upload_file(video_path, thumb_key, "image/png")
                return await self.storage.generate_presigned_url(thumb_key)
            except Exception:
                return None

        import tempfile
        thumb_path = os.path.join(tempfile.gettempdir(), f"{render_id}_thumb.png")
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-sseof", "-1", "-i", video_path,
                "-vframes", "1", "-q:v", "2", thumb_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)

            if os.path.exists(thumb_path):
                thumb_key = f"{self.config.s3_prefix}{render_id}/thumbnail.png"
                await self.storage.upload_file(thumb_path, thumb_key, "image/png")
                return await self.storage.generate_presigned_url(thumb_key)
        except Exception as e:
            logger.debug("Thumbnail generation failed: %s", e)
        finally:
            if os.path.exists(thumb_path):
                os.unlink(thumb_path)

        return None
