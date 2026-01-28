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
    from manim_mcp.core.rag import ChromaDBService
    from manim_mcp.core.renderer import ManimRenderer, RenderOutput
    from manim_mcp.core.sandbox import CodeSandbox
    from manim_mcp.core.scene_parser import SceneParser
    from manim_mcp.core.storage import S3Storage
    from manim_mcp.core.tracker import RenderTracker
    from manim_mcp.core.tts import GeminiTTSService

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
    audio: bool = False
    voice: str | None = None


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
        rag: ChromaDBService | None = None,
        tts: GeminiTTSService | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.renderer = renderer
        self.sandbox = sandbox
        self.scene_parser = scene_parser
        self.tracker = tracker
        self.storage = storage
        self.rag = rag
        self.tts = tts
        self._orchestrator = None  # Lazy initialized

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def orchestrator(self):
        """Lazy-initialize the agent orchestrator."""
        if self._orchestrator is None:
            from manim_mcp.agents.orchestrator import AgentOrchestrator
            self._orchestrator = AgentOrchestrator(self.llm, self.rag, self.config)
        return self._orchestrator

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

            # 4. Generate audio narration if requested
            has_audio = False
            if params.audio and result.local_path:
                try:
                    audio_path = await self._generate_audio(prompt, render_id, params.voice)
                    if audio_path:
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path
                        )
                        result.local_path = mixed_path
                        has_audio = True

                        # Re-upload the audio-mixed video to S3
                        if self.storage.available:
                            filename = os.path.basename(mixed_path)
                            s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                            try:
                                result.s3_url = await self.storage.upload_file(mixed_path, s3_key)
                                result.s3_object_key = s3_key
                                result.url = await self.storage.generate_presigned_url(s3_key)
                                result.file_size_bytes = os.path.getsize(mixed_path)
                                logger.info("Uploaded audio-mixed video to S3: %s", s3_key)
                            except Exception as e:
                                logger.warning("Failed to upload audio-mixed video: %s", e)
                except Exception as e:
                    logger.warning("Audio generation failed, continuing without audio: %s", e)

            # Self-index successful render for RAG
            await self._on_successful_render(code, prompt, render_id)

            message = "Animation created successfully"
            if has_audio:
                message += " with audio narration"

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
                message=message,
            )

        except Exception as e:
            await self.tracker.update_render(
                render_id, status=RenderStatus.failed, error_message=str(e)[:2000],
            )
            raise

    async def generate_advanced(
        self,
        prompt: str,
        params: RenderParams | None = None,
    ) -> AnimationResult:
        """Multi-agent generation with RAG context.

        Uses the 4-agent pipeline:
        1. ConceptAnalyzer → domain, complexity, concepts
        2. ScenePlanner → structure, timing, RAG examples
        3. CodeGenerator → code with RAG few-shot context
        4. CodeReviewer → quality check and fixes

        Falls back to simple generation if agents fail.

        Args:
            prompt: Text description of the animation
            params: Render parameters (quality, format, etc.)

        Returns:
            AnimationResult with render details
        """
        params = params or RenderParams()
        render_id = uuid.uuid4().hex[:12]
        await self.tracker.create_render(
            render_id,
            quality=params.quality.value,
            fmt=params.fmt.value,
            original_prompt=prompt,
        )

        try:
            # 1. Run multi-agent pipeline
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            pipeline_result = await self.orchestrator.generate_advanced(prompt)
            code = pipeline_result.generated_code

            # 2. Validate with sandbox (agents may have pre-validated)
            code = await self._validate_with_retries(code)

            # 3. Store generated code
            scenes = self.scene_parser.parse_scenes(code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                render_id, source_code=code, scene_name=scene_name,
            )

            # 4. Render + upload
            result = await self._render_and_upload(render_id, code, scene_name, params)

            # 5. Generate audio narration if requested
            has_audio = False
            if params.audio and result.local_path:
                try:
                    audio_path = await self._generate_audio(prompt, render_id, params.voice)
                    if audio_path:
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path
                        )
                        result.local_path = mixed_path
                        has_audio = True

                        # Re-upload the audio-mixed video to S3
                        if self.storage.available:
                            filename = os.path.basename(mixed_path)
                            s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                            try:
                                result.s3_url = await self.storage.upload_file(mixed_path, s3_key)
                                result.s3_object_key = s3_key
                                result.url = await self.storage.generate_presigned_url(s3_key)
                                result.file_size_bytes = os.path.getsize(mixed_path)
                                logger.info("Uploaded audio-mixed video to S3: %s", s3_key)
                            except Exception as e:
                                logger.warning("Failed to upload audio-mixed video: %s", e)
                except Exception as e:
                    logger.warning("Audio generation failed, continuing without audio: %s", e)

            # Self-index successful render
            await self._on_successful_render(code, prompt, render_id)

            message = "Animation created successfully (advanced mode)"
            if has_audio:
                message += " with audio narration"

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
                message=message,
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

    async def _validate_with_retries(self, code: str, original_code: str | None = None) -> str:
        for attempt in range(1, self.config.gemini_max_retries + 1):
            result = self.sandbox.validate(code)
            if result.valid:
                scenes = self.scene_parser.parse_scenes(code)
                if not scenes:
                    if attempt < self.config.gemini_max_retries:
                        error = "No Scene subclass found. Code must define a class inheriting from Scene."
                        fixed_code = await self.llm.fix_code(code, [error])
                        # Index error pattern for self-learning
                        await self._on_validation_fix(code, error, fixed_code)
                        code = fixed_code
                        continue
                    raise CodeValidationError("Generated code has no Scene subclass after retries")
                return code

            logger.warning("Validation failed (attempt %d/%d): %s",
                           attempt, self.config.gemini_max_retries, result.errors)
            if attempt < self.config.gemini_max_retries:
                fixed_code = await self.llm.fix_code(code, result.errors)
                # Index error pattern for self-learning
                await self._on_validation_fix(code, "; ".join(result.errors), fixed_code)
                code = fixed_code
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

    # ── Self-Indexing Hooks ────────────────────────────────────────────

    async def _on_successful_render(
        self,
        code: str,
        prompt: str,
        render_id: str,
    ) -> None:
        """Index successful render for RAG improvement.

        Called after a render completes successfully. The code is indexed
        so future similar prompts can benefit from this example.
        """
        if not self.rag or not self.rag.available:
            return

        try:
            await self.rag.index_manim_code(
                code,
                metadata={
                    "prompt": prompt[:500],  # Truncate long prompts
                    "render_id": render_id,
                    "source": "self_indexed",
                },
            )
            logger.debug("Self-indexed successful render: %s", render_id)
        except Exception as e:
            # Non-critical, just log
            logger.debug("Failed to self-index render: %s", e)

    async def _on_validation_fix(
        self,
        original_code: str,
        error: str,
        fixed_code: str,
    ) -> None:
        """Index error pattern and fix for self-learning.

        Called when validation fails and the LLM fixes the code.
        This allows the system to learn from its mistakes.
        """
        if not self.rag or not self.rag.available:
            return

        try:
            await self.rag.index_error_pattern(
                error=error,
                fix=fixed_code[:2000],  # Truncate long code
                original_code=original_code,
            )
            logger.debug("Self-indexed error pattern: %s", error[:50])
        except Exception as e:
            # Non-critical, just log
            logger.debug("Failed to self-index error pattern: %s", e)

    # ── Audio Generation ──────────────────────────────────────────────

    async def _generate_audio(
        self, prompt: str, render_id: str, voice: str | None = None
    ) -> str | None:
        """Generate narration audio for the animation.

        Args:
            prompt: The original animation prompt
            render_id: Render ID for temp file naming
            voice: Optional voice override

        Returns:
            Path to the generated WAV file, or None if TTS unavailable
        """
        if not self.tts:
            logger.warning("TTS service not available, skipping audio generation")
            return None

        # Override voice if specified
        if voice:
            original_voice = self.tts.voice
            self.tts.voice = voice

        try:
            logger.info("Generating audio narration for render %s", render_id)
            audio_data = await self.tts.generate_full_narration(prompt)

            # Save to temp file
            import tempfile
            audio_dir = os.path.join(tempfile.gettempdir(), f"manim_mcp_{render_id}")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, "narration.wav")

            with open(audio_path, "wb") as f:
                f.write(audio_data)

            logger.info("Audio narration saved to %s", audio_path)
            return audio_path

        finally:
            # Restore original voice if overridden
            if voice:
                self.tts.voice = original_voice

    async def _mix_audio_video(self, video_path: str, audio_path: str) -> str:
        """Mix audio track into video using ffmpeg.

        Args:
            video_path: Path to the rendered video
            audio_path: Path to the audio narration

        Returns:
            Path to the output video with audio
        """
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found, cannot mix audio")
            return video_path

        output_path = video_path.replace(".mp4", "_with_audio.mp4")
        if output_path == video_path:
            # Handle non-.mp4 extensions
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_with_audio{ext}"

        logger.info("Mixing audio into video: %s + %s -> %s", video_path, audio_path, output_path)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",  # End when shortest stream ends
            output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("ffmpeg failed: %s", stderr.decode() if stderr else "unknown error")
            return video_path  # Return original on failure

        logger.info("Audio mixed successfully: %s", output_path)
        return output_path
