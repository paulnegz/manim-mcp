"""Text-to-video pipeline: prompt → Gemini → validate → render → upload."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Awaitable

# Type alias for progress callback: async fn(stage: str, percent: int) -> None
ProgressCallback = Callable[[str, int], Awaitable[None]]

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
    video_speed: float = 0.50  # Slow down video before mixing with audio (0.50 = 50% speed)


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

    async def generate(
        self,
        prompt: str,
        params: RenderParams | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnimationResult:
        params = params or RenderParams()
        render_id = uuid.uuid4().hex[:12]
        await self.tracker.create_render(
            render_id, quality=params.quality.value, fmt=params.fmt.value, original_prompt=prompt,
        )

        async def report(stage: str, pct: int):
            if progress_callback:
                await progress_callback(stage, pct)

        try:
            # 1. Generate code via LLM (no narration constraint - video drives audio)
            await report("generating", 5)
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            await report("generating", 10)
            code = await self._generate_and_validate(prompt, narration_script=None)
            await report("validating", 20)

            # 2. Store generated code (20-25%)
            scenes = self.scene_parser.parse_scenes(code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                render_id, source_code=code, scene_name=scene_name,
            )
            await report("preparing", 25)

            # 3. Render video AND generate audio in PARALLEL (25-80%)
            await report("rendering", 30)

            # Start audio generation in parallel if requested
            audio_task = None
            if params.audio and self.tts:
                audio_task = asyncio.create_task(
                    self._generate_audio_from_code(code, prompt, render_id, params.voice)
                )
                logger.info("Started parallel: video render + audio generation")

            # Render video
            result = await self._render_with_retries(render_id, code, scene_name, params, prompt)
            await report("uploading", 60)

            # Wait for audio and mix (60-90%)
            has_audio = False
            if audio_task and result.local_path:
                try:
                    await report("generating_audio", 65)
                    audio_segments, narration_script = await audio_task

                    if audio_segments:
                        await report("mixing_audio", 75)

                        # Get video duration and stitch audio to match
                        video_duration = await self._get_media_duration(result.local_path)
                        target_audio_duration = video_duration / params.video_speed if params.video_speed > 0 else video_duration
                        logger.info(
                            "Video: %.1fs, target audio (after slowdown): %.1fs",
                            video_duration, target_audio_duration
                        )

                        # Stitch audio segments paced to video duration
                        audio_data, subtitle_timings = self.tts.stitch_audio_for_duration(
                            audio_segments, target_audio_duration
                        )

                        # Save audio and generate SRT subtitles
                        import tempfile
                        audio_dir = os.path.join(tempfile.gettempdir(), f"manim_mcp_{render_id}")
                        os.makedirs(audio_dir, exist_ok=True)
                        audio_path = os.path.join(audio_dir, "narration.wav")
                        with open(audio_path, "wb") as f:
                            f.write(audio_data)

                        # Generate SRT subtitles from narration script and timings
                        srt_path = None
                        if narration_script and subtitle_timings:
                            srt_content = self.tts.generate_srt(narration_script, subtitle_timings)
                            srt_path = os.path.join(audio_dir, "subtitles.srt")
                            with open(srt_path, "w", encoding="utf-8") as f:
                                f.write(srt_content)
                            logger.info("Generated SRT subtitles: %s", srt_path)

                        await report("mixing_audio", 80)
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path, params.video_speed, srt_path
                        )
                        result.local_path = mixed_path
                        has_audio = True
                        await report("uploading_audio", 85)

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

            # Always slow down video (even without audio)
            if not has_audio and result.local_path and params.video_speed != 1.0:
                await report("processing_video", 85)
                slowed_path = await self._slow_down_video(result.local_path, params.video_speed)
                if slowed_path != result.local_path:
                    result.local_path = slowed_path
                    # Re-upload slowed video to S3
                    if self.storage.available:
                        filename = os.path.basename(slowed_path)
                        s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                        try:
                            result.s3_url = await self.storage.upload_file(slowed_path, s3_key)
                            result.s3_object_key = s3_key
                            result.url = await self.storage.generate_presigned_url(s3_key)
                            result.file_size_bytes = os.path.getsize(slowed_path)
                            logger.info("Uploaded slowed video to S3: %s", s3_key)
                        except Exception as e:
                            logger.warning("Failed to upload slowed video: %s", e)
            elif not has_audio and result.local_path and params.video_speed == 1.0:
                # No audio, no speed change - still need mobile transcoding
                await report("processing_video", 85)
                mobile_path = await self._transcode_for_mobile(result.local_path)
                if mobile_path != result.local_path:
                    result.local_path = mobile_path
                    # Re-upload mobile-transcoded video to S3
                    if self.storage.available:
                        filename = os.path.basename(mobile_path)
                        s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                        try:
                            result.s3_url = await self.storage.upload_file(mobile_path, s3_key)
                            result.s3_object_key = s3_key
                            result.url = await self.storage.generate_presigned_url(s3_key)
                            result.file_size_bytes = os.path.getsize(mobile_path)
                            logger.info("Uploaded mobile-transcoded video to S3: %s", s3_key)
                        except Exception as e:
                            logger.warning("Failed to upload mobile-transcoded video: %s", e)

            # Generate thumbnail from FINAL video and embed it
            await report("generating_thumbnail", 92)
            if result.local_path and os.path.exists(result.local_path):
                thumb_url, thumb_key, thumb_path = await self._generate_thumbnail(
                    result.local_path, render_id
                )
                if thumb_path and os.path.exists(thumb_path):
                    # Embed thumbnail into video file
                    await self._embed_thumbnail_in_video(result.local_path, thumb_path)
                    # Clean up thumbnail file
                    try:
                        os.unlink(thumb_path)
                    except Exception:
                        pass
                    # Re-upload video with embedded thumbnail
                    if self.storage.available and result.s3_object_key:
                        try:
                            await self.storage.upload_file(result.local_path, result.s3_object_key)
                            result.url = await self.storage.generate_presigned_url(result.s3_object_key)
                            result.file_size_bytes = os.path.getsize(result.local_path)
                            logger.info("Re-uploaded video with embedded thumbnail")
                        except Exception as e:
                            logger.warning("Failed to re-upload video with thumbnail: %s", e)
                result.thumbnail_url = thumb_url
                result.thumbnail_s3_key = thumb_key

            await report("finalizing", 95)

            # Self-index successful render for RAG
            await self._on_successful_render(code, prompt, render_id)

            message = "Animation created successfully"
            if has_audio:
                message += " with audio narration"

            return AnimationResult(
                render_id=render_id,
                status=RenderStatus.completed,
                url=result.url or result.local_path,
                s3_object_key=result.s3_object_key,
                thumbnail_url=result.thumbnail_url,
                thumbnail_s3_key=result.thumbnail_s3_key,
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
        progress_callback: ProgressCallback | None = None,
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
            progress_callback: Optional callback for progress updates

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

        async def report(stage: str, pct: int):
            if progress_callback:
                await progress_callback(stage, pct)

        try:
            # 1. Run multi-agent pipeline (no narration constraint - video drives audio)
            await report("analyzing", 5)
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            await report("planning", 10)
            pipeline_result = await self.orchestrator.generate_advanced(prompt, narration_script=None)
            code = pipeline_result.generated_code
            await report("generating", 15)

            # 2. Validate with sandbox (agents may have pre-validated)
            code = await self._validate_with_retries(code)
            await report("validating", 20)

            # 3. Store generated code
            scenes = self.scene_parser.parse_scenes(code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                render_id, source_code=code, scene_name=scene_name,
            )
            await report("preparing", 25)

            # 4. Render video AND generate audio in PARALLEL (25-80%)
            await report("rendering", 30)

            # Start audio generation in parallel if requested
            audio_task = None
            if params.audio and self.tts:
                audio_task = asyncio.create_task(
                    self._generate_audio_from_code(code, prompt, render_id, params.voice)
                )
                logger.info("Started parallel: video render + audio generation")

            # Render video
            result = await self._render_and_upload(render_id, code, scene_name, params)
            await report("uploading", 60)

            # Wait for audio and mix (60-90%)
            has_audio = False
            if audio_task and result.local_path:
                try:
                    await report("generating_audio", 65)
                    audio_segments, narration_script = await audio_task

                    if audio_segments:
                        await report("mixing_audio", 75)

                        # Get video duration and stitch audio to match
                        video_duration = await self._get_media_duration(result.local_path)
                        target_audio_duration = video_duration / params.video_speed if params.video_speed > 0 else video_duration
                        logger.info(
                            "Video: %.1fs, target audio (after slowdown): %.1fs",
                            video_duration, target_audio_duration
                        )

                        # Stitch audio segments paced to video duration
                        audio_data, subtitle_timings = self.tts.stitch_audio_for_duration(
                            audio_segments, target_audio_duration
                        )

                        # Save audio and generate SRT subtitles
                        import tempfile
                        audio_dir = os.path.join(tempfile.gettempdir(), f"manim_mcp_{render_id}")
                        os.makedirs(audio_dir, exist_ok=True)
                        audio_path = os.path.join(audio_dir, "narration.wav")
                        with open(audio_path, "wb") as f:
                            f.write(audio_data)

                        # Generate SRT subtitles from narration script and timings
                        srt_path = None
                        if narration_script and subtitle_timings:
                            srt_content = self.tts.generate_srt(narration_script, subtitle_timings)
                            srt_path = os.path.join(audio_dir, "subtitles.srt")
                            with open(srt_path, "w", encoding="utf-8") as f:
                                f.write(srt_content)
                            logger.info("Generated SRT subtitles: %s", srt_path)

                        await report("mixing_audio", 80)
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path, params.video_speed, srt_path
                        )
                        result.local_path = mixed_path
                        has_audio = True
                        await report("uploading_audio", 85)

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

            # Always slow down video (even without audio)
            if not has_audio and result.local_path and params.video_speed != 1.0:
                await report("processing_video", 85)
                slowed_path = await self._slow_down_video(result.local_path, params.video_speed)
                if slowed_path != result.local_path:
                    result.local_path = slowed_path
                    # Re-upload slowed video to S3
                    if self.storage.available:
                        filename = os.path.basename(slowed_path)
                        s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                        try:
                            result.s3_url = await self.storage.upload_file(slowed_path, s3_key)
                            result.s3_object_key = s3_key
                            result.url = await self.storage.generate_presigned_url(s3_key)
                            result.file_size_bytes = os.path.getsize(slowed_path)
                            logger.info("Uploaded slowed video to S3: %s", s3_key)
                        except Exception as e:
                            logger.warning("Failed to upload slowed video: %s", e)
            elif not has_audio and result.local_path and params.video_speed == 1.0:
                # No audio, no speed change - still need mobile transcoding
                await report("processing_video", 85)
                mobile_path = await self._transcode_for_mobile(result.local_path)
                if mobile_path != result.local_path:
                    result.local_path = mobile_path
                    # Re-upload mobile-transcoded video to S3
                    if self.storage.available:
                        filename = os.path.basename(mobile_path)
                        s3_key = f"{self.config.s3_prefix}{render_id}/{filename}"
                        try:
                            result.s3_url = await self.storage.upload_file(mobile_path, s3_key)
                            result.s3_object_key = s3_key
                            result.url = await self.storage.generate_presigned_url(s3_key)
                            result.file_size_bytes = os.path.getsize(mobile_path)
                            logger.info("Uploaded mobile-transcoded video to S3: %s", s3_key)
                        except Exception as e:
                            logger.warning("Failed to upload mobile-transcoded video: %s", e)

            # Generate thumbnail from FINAL video and embed it
            await report("generating_thumbnail", 92)
            if result.local_path and os.path.exists(result.local_path):
                thumb_url, thumb_key, thumb_path = await self._generate_thumbnail(
                    result.local_path, render_id
                )
                if thumb_path and os.path.exists(thumb_path):
                    # Embed thumbnail into video file
                    await self._embed_thumbnail_in_video(result.local_path, thumb_path)
                    # Clean up thumbnail file
                    try:
                        os.unlink(thumb_path)
                    except Exception:
                        pass
                    # Re-upload video with embedded thumbnail
                    if self.storage.available and result.s3_object_key:
                        try:
                            await self.storage.upload_file(result.local_path, result.s3_object_key)
                            result.url = await self.storage.generate_presigned_url(result.s3_object_key)
                            result.file_size_bytes = os.path.getsize(result.local_path)
                            logger.info("Re-uploaded video with embedded thumbnail")
                        except Exception as e:
                            logger.warning("Failed to re-upload video with thumbnail: %s", e)
                result.thumbnail_url = thumb_url
                result.thumbnail_s3_key = thumb_key

            await report("finalizing", 95)

            # Self-index successful render
            await self._on_successful_render(code, prompt, render_id)

            message = "Animation created successfully (advanced mode)"
            if has_audio:
                message += " with audio narration"

            return AnimationResult(
                render_id=render_id,
                status=RenderStatus.completed,
                url=result.url or result.local_path,
                s3_object_key=result.s3_object_key,
                thumbnail_url=result.thumbnail_url,
                thumbnail_s3_key=result.thumbnail_s3_key,
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
        progress_callback: ProgressCallback | None = None,
    ) -> AnimationResult:
        params = params or RenderParams()

        async def report(stage: str, pct: int):
            if progress_callback:
                await progress_callback(stage, pct)

        # 1. Get original render
        await report("loading", 5)
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
        await report("preparing", 10)

        try:
            # 3. Edit code via Gemini (10-30%)
            await report("editing", 15)
            await self.tracker.update_render(new_id, status=RenderStatus.generating)
            edited_code = await self._edit_and_validate(original.source_code, instructions)
            await report("validating", 30)

            # 4. Store edited code
            scenes = self.scene_parser.parse_scenes(edited_code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                new_id, source_code=edited_code, scene_name=scene_name,
            )
            await report("storing", 35)

            # 5. Render + upload with retry on runtime errors (35-90%)
            await report("rendering", 40)
            result = await self._render_with_retries(
                new_id, edited_code, scene_name, params, original.original_prompt or instructions
            )
            await report("uploading", 90)

            # Transcode for mobile compatibility
            if result.local_path:
                await report("processing_video", 92)
                mobile_path = await self._transcode_for_mobile(result.local_path)
                if mobile_path != result.local_path:
                    result.local_path = mobile_path
                    # Re-upload mobile-transcoded video to S3
                    if self.storage.available:
                        filename = os.path.basename(mobile_path)
                        s3_key = f"{self.config.s3_prefix}{new_id}/{filename}"
                        try:
                            result.s3_url = await self.storage.upload_file(mobile_path, s3_key)
                            result.s3_object_key = s3_key
                            result.url = await self.storage.generate_presigned_url(s3_key)
                            result.file_size_bytes = os.path.getsize(mobile_path)
                            logger.info("Uploaded mobile-transcoded video to S3: %s", s3_key)
                        except Exception as e:
                            logger.warning("Failed to upload mobile-transcoded video: %s", e)

            # Generate thumbnail from FINAL video and embed it
            await report("generating_thumbnail", 93)
            if result.local_path and os.path.exists(result.local_path):
                thumb_url, thumb_key, thumb_path = await self._generate_thumbnail(
                    result.local_path, new_id
                )
                if thumb_path and os.path.exists(thumb_path):
                    # Embed thumbnail into video file
                    await self._embed_thumbnail_in_video(result.local_path, thumb_path)
                    # Clean up thumbnail file
                    try:
                        os.unlink(thumb_path)
                    except Exception:
                        pass
                    # Re-upload video with embedded thumbnail
                    if self.storage.available and result.s3_object_key:
                        try:
                            await self.storage.upload_file(result.local_path, result.s3_object_key)
                            result.url = await self.storage.generate_presigned_url(result.s3_object_key)
                            result.file_size_bytes = os.path.getsize(result.local_path)
                            logger.info("Re-uploaded video with embedded thumbnail")
                        except Exception as e:
                            logger.warning("Failed to re-upload video with thumbnail: %s", e)
                result.thumbnail_url = thumb_url
                result.thumbnail_s3_key = thumb_key

            await report("finalizing", 95)

            return AnimationResult(
                render_id=new_id,
                status=RenderStatus.completed,
                url=result.url or result.local_path,
                s3_object_key=result.s3_object_key,
                thumbnail_url=result.thumbnail_url,
                thumbnail_s3_key=result.thumbnail_s3_key,
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

    async def _get_rag_fix_hints(self, error_msg: str) -> list[str]:
        """Query RAG for error patterns + API signatures to help fix."""
        if not self.rag:
            return []

        hints = []
        import re

        # Extract class/method names from error - handle multiple patterns
        # IMPORTANT: Order matters! More specific patterns (AttributeError, TypeError)
        # must come BEFORE generic patterns to avoid matching traceback lines like "sys.exit("
        # NOTE: For TypeError with unexpected keyword, we need ClassName.method pattern
        patterns = [
            # Specific error patterns first
            r"'(\w+)' object has no attribute '(\w+)'",  # AttributeError - most specific
            # TypeError: ClassName.method() got an unexpected keyword argument 'param'
            # Capture: group(1)=ClassName, group(2)=method, group(3)=invalid_param
            r"(\w+)\.(\w+)\(\) got an unexpected keyword argument '(\w+)'",
            r"(\w+)\(\) got an unexpected keyword argument '(\w+)'",  # TypeError with method only
            r"(\w+)\(.*unexpected.*argument.*'(\w+)'",  # unexpected argument variant
            r"(\w+)\.__init__\(\)",  # __init__ error
            # Generic pattern last (can match traceback noise like "sys.exit(")
            # Only use if nothing else matched
        ]

        class_name = None
        method_name = None
        invalid_param = None
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                if match.lastindex >= 3:
                    # Pattern with ClassName.method(invalid_param)
                    class_name = match.group(1)
                    method_name = match.group(2)
                    invalid_param = match.group(3)
                elif match.lastindex >= 2:
                    class_name = match.group(1)
                    method_name = match.group(2)
                else:
                    class_name = match.group(1)
                break

        # If no specific pattern matched, try the generic one but filter out common false positives
        if not class_name:
            generic_match = re.search(r"(\w+)\.(\w+)\(", error_msg)
            if generic_match:
                potential_class = generic_match.group(1)
                # Filter out common false positives from tracebacks
                if potential_class not in ("sys", "os", "re", "ast", "json", "logging", "asyncio", "self"):
                    class_name = potential_class
                    method_name = generic_match.group(2)

        if class_name:
            # Build search query: prefer ClassName.method if both available
            search_query = f"{class_name}.{method_name}" if method_name else class_name
            logger.info("[RAG-RETRY] Looking up API for: %s", search_query)
            sigs = await self.rag.search_api_signatures(search_query, n_results=3)
            for sig in sigs:
                meta = sig.get("metadata", {})
                # Try parameter_names first (from AST), then parameters, then valid_params
                params = meta.get("parameter_names", "") or meta.get("parameters", "") or meta.get("valid_params", "")
                # Get required params - check both 'required' and 'required_params' keys
                required = meta.get("required_params", "") or meta.get("required", [])
                if isinstance(required, str):
                    required = [r.strip() for r in required.split(",") if r.strip()]
                if params:
                    # Build helpful hint - emphasize REQUIRED params first!
                    if required:
                        required_str = ", ".join(required) if isinstance(required, list) else str(required)
                        # Split params into required vs optional
                        param_list = [p.strip() for p in params.split(",")]
                        optional = [p for p in param_list if p not in required]
                        optional_str = ", ".join(optional[:10])  # Limit optional list
                        hint = f"[API] {search_query} REQUIRED: {required_str}"
                        if optional_str:
                            hint += f" | Optional: {optional_str}"
                    else:
                        hint = f"[API] {search_query} valid params: {params[:300]}"
                    # If we detected an invalid param, explicitly call it out
                    if invalid_param and invalid_param not in params:
                        hint += f"\n  ❌ '{invalid_param}' is NOT a valid parameter!"
                        # Suggest similar valid param if exists
                        for valid_p in params.split(","):
                            valid_p = valid_p.strip()
                            if valid_p and (invalid_param in valid_p or valid_p in invalid_param):
                                hint += f"\n  ✓ Did you mean '{valid_p}'?"
                    hints.append(hint)
                    logger.info("[RAG-RETRY] Found API signature for %s", search_query)
                    break

        # Search error patterns for known fixes
        similar = await self.rag.search_error_patterns(error_msg[:500], n_results=2)
        logger.info("[RAG-RETRY] Found %d similar error patterns", len(similar))
        for pattern in similar:
            content = pattern.get("content", "")
            # Extract fix from document content (stored after "FIX:" marker)
            if "FIX:" in content and pattern.get("metadata", {}).get("has_fix"):
                import re
                fix_match = re.search(r"FIX:\s*```python\s*(.*?)```", content, re.DOTALL)
                if fix_match:
                    fix = fix_match.group(1).strip()
                    hints.append(f"[FIX] Similar error fixed by:\n{fix[:500]}")
                    logger.info("[RAG-RETRY] Found fix hint!")
                    break

        # If no hints found, add a generic hint about CONFIG pattern
        if not hints and "has no attribute" in error_msg:
            hints.append("[HINT] Don't use CONFIG dict pattern. Define variables directly in construct().")
            logger.info("[RAG-RETRY] Added CONFIG pattern hint")

        return hints

    async def _render_with_retries(
        self,
        render_id: str,
        code: str,
        scene_name: str | None,
        params: RenderParams,
        prompt: str,
    ):
        """Render with retry on runtime errors (e.g., Manim crashes)."""
        from manim_mcp.core.code_bridge import bridge_code

        last_error = None
        current_code = code
        previous_error = None  # Track error for learning
        previous_code = None   # Track code that failed

        for attempt in range(1, self.config.gemini_max_retries + 1):
            # Transform code using CE→manimgl bridge (for error tracking)
            # Pass RAG for dynamic API validation against 1,652+ indexed signatures
            transformed_code = bridge_code(current_code, rag=self.rag)

            try:
                result = await self._render_and_upload(render_id, current_code, scene_name, params)
                # If we're on a retry and it succeeded, store the error→fix pattern
                if previous_error and self.rag:
                    previous_transformed = bridge_code(previous_code, rag=self.rag) if previous_code else None
                    await self.rag.store_error_pattern(
                        error_message=previous_error[:500],
                        code=previous_code[:3000],
                        fix=current_code[:3000],  # The working code
                        prompt=prompt[:200],
                        transformed_code=previous_transformed[:3000] if previous_transformed else None,
                    )
                    logger.info("[RAG] Stored successful error fix pattern")
                return result
            except Exception as e:
                last_error = e
                error_msg = str(e)

                # Check if this is a Manim runtime error worth retrying
                if "Manim exited" not in error_msg and "render" not in error_msg.lower():
                    # Not a render error, don't retry
                    raise

                if attempt < self.config.gemini_max_retries:
                    logger.warning(
                        "Render failed (attempt %d/%d): %s",
                        attempt, self.config.gemini_max_retries, error_msg[:1000]
                    )
                    previous_error = error_msg
                    previous_code = current_code

                    # Enrich error with RAG hints if available
                    errors = [error_msg[:1500]]
                    errors.extend(await self._get_rag_fix_hints(error_msg))

                    await self.tracker.update_render(render_id, status=RenderStatus.generating)
                    current_code = await self.llm.fix_code(current_code, errors)

                    # Re-validate and update scene name
                    current_code = await self._validate_with_retries(current_code)
                    scenes = self.scene_parser.parse_scenes(current_code)
                    scene_name = scenes[0].name if scenes else None
                    await self.tracker.update_render(
                        render_id, source_code=current_code, scene_name=scene_name,
                    )
                else:
                    logger.error(
                        "Render failed after %d attempts: %s",
                        self.config.gemini_max_retries, error_msg[:1500]
                    )
                    # Store error pattern for future learning (with both original and transformed)
                    if self.rag:
                        await self.rag.store_error_pattern(
                            error_message=error_msg[:1000],
                            code=current_code[:3000],
                            fix=None,  # No fix available yet
                            prompt=prompt[:200] if prompt else None,
                            transformed_code=transformed_code[:3000],
                        )
                    raise

        raise last_error  # Should not reach here

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
                output.thumbnail_url, output.thumbnail_s3_key, _ = await self._generate_thumbnail(
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
            thumbnail_s3_key=output.thumbnail_s3_key,
            file_size_bytes=output.file_size_bytes,
            render_time_seconds=output.render_time_seconds,
            local_path=output.local_path,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        return output

    async def _get_preflight_warnings(self, prompt: str) -> list[str]:
        """Get warnings based on similar failed prompts from error patterns.

        Queries the error patterns collection for prompts similar to the current one
        that have failed before, and extracts actionable warnings.

        Args:
            prompt: The user's animation prompt

        Returns:
            List of warning messages about common pitfalls for similar animations
        """
        if not self.rag or not self.rag.available:
            return []

        warnings = []
        try:
            # Query error patterns for similar prompts
            similar_errors = await self.rag.search_error_patterns(prompt, n_results=5)

            for error in similar_errors:
                metadata = error.get("metadata", {})
                # Only warn about patterns with low success rate or no fix
                success_rate = metadata.get("success_rate", 0)
                has_fix = metadata.get("has_fix", False)

                if success_rate < 0.5 or not has_fix:
                    error_reason = metadata.get("error_reason", "")
                    error_type = metadata.get("error_type", "")

                    if error_reason and error_reason not in warnings:
                        # Format warning for LLM
                        warning = f"- {error_reason}"
                        if error_type:
                            warning = f"- [{error_type}] {error_reason}"
                        warnings.append(warning)

            # Limit to most relevant warnings
            warnings = warnings[:5]

            if warnings:
                logger.info("[PRE-FLIGHT] Found %d warnings for similar prompts", len(warnings))

        except Exception as e:
            logger.debug("[PRE-FLIGHT] Failed to get warnings: %s", e)

        return warnings

    async def _generate_and_validate(self, prompt: str, narration_script: list[str] | None = None) -> str:
        """Generate code with optional RAG context and validate.

        Uses RAG to find similar scenes for few-shot context when available.
        If narration_script is provided, code will be generated to match it.
        """
        # Query RAG for similar scenes if available
        enhanced_prompt = prompt
        if self.rag and self.rag.available:
            try:
                similar_scenes = await self.rag.search_similar_scenes(
                    query=prompt,
                    n_results=3,
                    prioritize_3b1b=True,
                )
                if similar_scenes:
                    logger.info("[RAG] Found %d similar scenes for context", len(similar_scenes))
                    rag_context = self._build_rag_context(similar_scenes)
                    enhanced_prompt = f"{prompt}\n\nHere are some similar animations for reference:\n{rag_context}"
            except Exception as e:
                logger.warning("[RAG] Failed to query similar scenes: %s", e)

        # Get pre-flight warnings from similar failed prompts
        preflight_warnings = await self._get_preflight_warnings(prompt)
        if preflight_warnings:
            warnings_text = "\n".join(preflight_warnings)
            enhanced_prompt = f"""{enhanced_prompt}

## KNOWN PITFALLS (from similar animations that failed):
{warnings_text}

IMPORTANT: Avoid these patterns in your implementation."""
            logger.info("[PRE-FLIGHT] Injected %d warnings into prompt", len(preflight_warnings))

        # If narration script provided, include it to sync code with audio
        if narration_script:
            script_text = "\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(narration_script))
            enhanced_prompt = f"""{enhanced_prompt}

IMPORTANT - NARRATION SCRIPT TO FOLLOW:
The animation MUST match this narration script exactly. Each numbered sentence corresponds to a visual step.

{script_text}

Generate code where each self.play() or self.wait() corresponds to one narration sentence in order.
The timing and visuals must sync with this script."""
            logger.info("Code generation guided by %d-sentence narration script", len(narration_script))

        code = await self.llm.generate_code(enhanced_prompt)
        return await self._validate_with_retries(code)

    def _build_rag_context(self, similar_scenes: list[dict]) -> str:
        """Format RAG results as context for generation.

        Shows full code for high-similarity matches to enable close following of patterns.
        """
        lines = []
        for i, scene in enumerate(similar_scenes[:3], 1):
            meta = scene.get("metadata", {})
            similarity = scene.get("similarity_score", 0)
            prompt_hint = meta.get("prompt", "")[:100]
            if prompt_hint:
                lines.append(f"\nExample {i} (prompt: {prompt_hint}, similarity: {similarity:.2f}):")
            else:
                lines.append(f"\nExample {i} (similarity: {similarity:.2f}):")

            # Show more code for high-similarity matches (up to 3000 chars)
            # This allows LLM to follow working patterns closely instead of inventing
            max_chars = 3000 if similarity > 0.7 else 1500
            code = scene.get("content", "")[:max_chars]
            if code:
                lines.append(f"```python\n{code}\n```")
                if similarity > 0.8:
                    lines.append("**IMPORTANT: This is a highly relevant example. Follow this pattern closely.**")
        return "\n".join(lines)

    async def _edit_and_validate(self, original_code: str, instructions: str) -> str:
        code = await self.llm.edit_code(original_code, instructions)
        return await self._validate_with_retries(code)

    async def _validate_with_retries(self, code: str, original_code: str | None = None) -> str:
        from manim_mcp.core.code_bridge import sanitize_latex_strings

        for attempt in range(1, self.config.gemini_max_retries + 1):
            # Sanitize LaTeX strings BEFORE AST validation to fix common LLM errors
            # like unterminated strings, unbalanced braces, etc.
            try:
                code, latex_fixes = sanitize_latex_strings(code)
                if latex_fixes:
                    logger.info("[VALIDATION] Pre-sanitized %d LaTeX issues", len(latex_fixes))
            except Exception as e:
                logger.debug("[VALIDATION] LaTeX sanitization failed: %s", e)

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
                # Enhance errors with specific context for syntax errors
                enhanced_errors = self._enhance_error_context(code, result.errors)
                fixed_code = await self.llm.fix_code(code, enhanced_errors)
                # Index error pattern for self-learning
                await self._on_validation_fix(code, "; ".join(result.errors), fixed_code)
                code = fixed_code
            else:
                raise CodeValidationError(
                    f"Code failed validation after {self.config.gemini_max_retries} attempts: "
                    + "; ".join(result.errors)
                )

        return code

    def _enhance_error_context(self, code: str, errors: list[str]) -> list[str]:
        """Enhance error messages with specific context to help LLM fix them.

        For syntax errors, extracts:
        - The exact line that caused the error
        - Hints about likely causes (apostrophes, mixed quotes, etc.)

        Args:
            code: The code that failed validation
            errors: List of error messages from validator

        Returns:
            Enhanced error list with more context
        """
        import re
        enhanced = []
        lines = code.split('\n')

        for error in errors:
            enhanced_error = error

            # Check for syntax error with line/column info
            # Format: "Syntax error at line X, column Y: message"
            syntax_match = re.search(r'[Ll]ine\s*(\d+)(?:,\s*column\s*(\d+))?', error)
            if syntax_match:
                line_num = int(syntax_match.group(1))
                col_num = int(syntax_match.group(2)) if syntax_match.group(2) else None

                # Get the problematic line
                if 1 <= line_num <= len(lines):
                    problem_line = lines[line_num - 1]
                    enhanced_error += f"\n\nPROBLEMATIC LINE {line_num}:\n{problem_line}"

                    # Add column pointer if available
                    if col_num:
                        pointer = ' ' * (col_num - 1) + '^'
                        enhanced_error += f"\n{pointer}"

                    # Analyze the line for common issues and add hints
                    hints = self._analyze_line_for_hints(problem_line, error)
                    if hints:
                        enhanced_error += f"\n\nLIKELY CAUSE: {hints}"

            # Check for unterminated string literal
            if 'unterminated string' in error.lower():
                # Add specific guidance
                enhanced_error += "\n\nFIX: Check for mismatched quotes (\" vs ') or apostrophes in text like \"Newton's Law\""

            enhanced.append(enhanced_error)

        return enhanced

    def _analyze_line_for_hints(self, line: str, error: str) -> str:
        """Analyze a problematic line to provide specific fix hints.

        Args:
            line: The line that caused the error
            error: The error message

        Returns:
            Hint string about likely cause, or empty string
        """
        hints = []

        # Check for apostrophe in single-quoted string
        # Pattern: 'something's or 's anywhere in single-quoted context
        if "'" in line:
            # Check if there's a word with apostrophe-s pattern
            import re
            if re.search(r"'[^']*\w's[^']*'", line) or re.search(r"'\w+'s", line):
                hints.append("Contains apostrophe (like \"Newton's\") in single-quoted string - USE DOUBLE QUOTES instead")

            # Check for mixed quotes (started with one, ended with other)
            if '"' in line:
                # Count quote types to detect mismatch
                double_count = line.count('"')
                single_count = line.count("'")
                if double_count % 2 == 1 and single_count % 2 == 1:
                    hints.append("Mixed quote types detected - string may start with \" but end with ' (or vice versa)")

        # Check for common LaTeX issues
        if 'unterminated' in error.lower() and ('Tex' in line or 'r"' in line or "r'" in line):
            if line.count('{') != line.count('}'):
                hints.append("Unbalanced LaTeX braces { }")

        return "; ".join(hints) if hints else ""

    async def _generate_thumbnail(self, video_path: str, render_id: str) -> tuple[str | None, str | None, str | None]:
        """Generate thumbnail and return (presigned_url, s3_key, local_path).

        The local_path is kept for embedding into the video file.
        """
        if not shutil.which("ffmpeg") or not self.storage.available:
            return None, None, None

        if video_path.endswith(".png"):
            thumb_key = f"{self.config.s3_prefix}{render_id}/thumbnail.png"
            try:
                await self.storage.upload_file(video_path, thumb_key, "image/png")
                url = await self.storage.generate_presigned_url(thumb_key)
                return url, thumb_key, video_path
            except Exception:
                return None, None, None

        import tempfile
        thumb_path = os.path.join(tempfile.gettempdir(), f"{render_id}_thumb.png")
        try:
            # Get video duration using ffprobe
            duration = 0.0
            if shutil.which("ffprobe"):
                probe_proc = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", video_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(probe_proc.communicate(), timeout=10)
                try:
                    duration = float(stdout.decode().strip())
                except (ValueError, AttributeError):
                    duration = 0.0

            # Extract middle frame (or fallback to 1s from end if duration unknown)
            if duration > 0:
                middle_time = str(duration / 2)
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-ss", middle_time, "-i", video_path,
                    "-vframes", "1", "-q:v", "2", thumb_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
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
                url = await self.storage.generate_presigned_url(thumb_key)
                # Return local path too - don't delete yet, needed for embedding
                return url, thumb_key, thumb_path
        except Exception as e:
            logger.debug("Thumbnail generation failed: %s", e)
            if os.path.exists(thumb_path):
                os.unlink(thumb_path)

        return None, None, None

    async def _embed_thumbnail_in_video(self, video_path: str, thumb_path: str) -> str:
        """Embed thumbnail as cover art/attached picture in MP4 file.

        This makes the thumbnail visible when video is downloaded and viewed
        in file browsers, video players, etc.

        Returns path to video with embedded thumbnail (or original if failed).
        """
        if not os.path.exists(thumb_path) or not os.path.exists(video_path):
            return video_path

        if not video_path.endswith(".mp4"):
            # Only MP4 supports attached pictures well
            return video_path

        output_path = video_path.replace(".mp4", "_with_thumb.mp4")

        try:
            # Embed thumbnail as attached picture (cover art)
            # -map 0 = all streams from video
            # -map 1 = thumbnail image
            # -c copy = copy all streams without re-encoding
            # -c:v:1 mjpeg = encode thumbnail as mjpeg (required for attached_pic)
            # -disposition:v:1 attached_pic = mark second video stream as cover art
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", thumb_path,
                "-map", "0",
                "-map", "1",
                "-c", "copy",
                "-c:v:1", "mjpeg",
                "-disposition:v:1", "attached_pic",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode == 0 and os.path.exists(output_path):
                # Replace original with thumbnail-embedded version
                os.replace(output_path, video_path)
                logger.info("Embedded thumbnail into video: %s", video_path)
                return video_path
            else:
                logger.warning("Failed to embed thumbnail: %s", stderr.decode()[:200] if stderr else "unknown error")
                if os.path.exists(output_path):
                    os.unlink(output_path)
        except Exception as e:
            logger.warning("Thumbnail embedding failed: %s", e)
            if os.path.exists(output_path):
                os.unlink(output_path)

        return video_path

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
                    "source": "generated",  # Mark as generated (filtered out in RAG to prioritize 3b1b)
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
        self, prompt: str, render_id: str, voice: str | None = None, script: list[str] | None = None
    ) -> str | None:
        """Generate narration audio for the animation.

        Args:
            prompt: The original animation prompt
            render_id: Render ID for temp file naming
            voice: Optional voice override
            script: Pre-generated narration script (for code-audio sync)

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
            audio_data = await self.tts.generate_full_narration(prompt, script)

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

    async def _generate_audio_from_script(
        self, script: list[str], render_id: str, voice: str | None = None
    ) -> str | None:
        """Generate narration audio from a pre-generated script.

        This is used for parallel TTS generation - the script is already
        generated, so we skip script generation and go straight to TTS.
        TTS internally parallelizes across sentences.

        Args:
            script: Pre-generated narration script (list of sentences)
            render_id: Render ID for temp file naming
            voice: Optional voice override

        Returns:
            Path to the generated WAV file, or None if TTS unavailable
        """
        if not self.tts:
            logger.warning("TTS service not available, skipping audio generation")
            return None

        if not script:
            logger.warning("No script provided for audio generation")
            return None

        # Override voice if specified
        original_voice = None
        if voice:
            original_voice = self.tts.voice
            self.tts.voice = voice

        try:
            logger.info("Generating audio from %d-sentence script (parallel TTS)", len(script))

            # Generate audio for all sentences in parallel
            audio_segments = await self.tts.generate_parallel(script)

            successful = sum(1 for s in audio_segments if not isinstance(s, Exception))
            logger.info("TTS completed: %d/%d sentences successful", successful, len(script))

            # Stitch audio segments together
            audio_data = self.tts.stitch_audio(audio_segments)

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
            if original_voice:
                self.tts.voice = original_voice

    async def _generate_audio_from_code(
        self, code: str, prompt: str, render_id: str, voice: str | None = None
    ) -> tuple[list, list[str]]:
        """Generate narration and TTS from code in parallel with video render.

        Args:
            code: Generated Manim code with comments
            prompt: Original animation prompt
            render_id: Render ID for logging
            voice: Optional voice override

        Returns:
            Tuple of (audio_segments, narration_script) - segments not yet stitched
        """
        if not self.tts:
            return [], []

        original_voice = None
        if voice:
            original_voice = self.tts.voice
            self.tts.voice = voice

        try:
            # Generate narration script from code (no duration constraint yet)
            logger.info("Generating narration script from code comments")
            narration_script = await self.tts.generate_narration_for_duration(
                prompt, target_duration=30.0, code=code  # Estimate ~30s, will adjust later
            )
            logger.info("Generated %d-sentence narration from code", len(narration_script))

            if not narration_script:
                return [], []

            # Generate TTS for all sentences in parallel
            logger.info("Generating TTS for %d sentences in parallel", len(narration_script))
            audio_segments = await self.tts.generate_parallel(narration_script)

            successful = sum(1 for s in audio_segments if not isinstance(s, Exception))
            logger.info("TTS completed: %d/%d sentences", successful, len(narration_script))

            return audio_segments, narration_script

        finally:
            if original_voice:
                self.tts.voice = original_voice

    async def _generate_audio_for_duration(
        self, script: list[str], render_id: str, target_duration: float, voice: str | None = None
    ) -> str | None:
        """Generate narration audio paced to fit target video duration.

        Args:
            script: Narration script (list of sentences)
            render_id: Render ID for temp file naming
            target_duration: Target duration in seconds
            voice: Optional voice override

        Returns:
            Path to the generated WAV file, or None if TTS unavailable
        """
        if not self.tts:
            logger.warning("TTS service not available, skipping audio generation")
            return None

        if not script:
            logger.warning("No script provided for audio generation")
            return None

        original_voice = None
        if voice:
            original_voice = self.tts.voice
            self.tts.voice = voice

        try:
            logger.info("Generating audio for %d sentences, target %.1fs", len(script), target_duration)

            # Generate audio for all sentences in parallel
            audio_segments = await self.tts.generate_parallel(script)

            successful = sum(1 for s in audio_segments if not isinstance(s, Exception))
            logger.info("TTS completed: %d/%d sentences successful", successful, len(script))

            # Stitch audio adjusted to fit target duration
            audio_data, _ = self.tts.stitch_audio_for_duration(audio_segments, target_duration)

            # Save to temp file
            import tempfile
            audio_dir = os.path.join(tempfile.gettempdir(), f"manim_mcp_{render_id}")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, "narration.wav")

            with open(audio_path, "wb") as f:
                f.write(audio_data)

            logger.info("Audio narration (duration-adjusted) saved to %s", audio_path)
            return audio_path

        finally:
            if original_voice:
                self.tts.voice = original_voice

    async def _mix_audio_video(
        self, video_path: str, audio_path: str, video_speed: float = 0.70, srt_path: str | None = None
    ) -> str:
        """Mix audio track into video using ffmpeg, optionally burning in subtitles.

        Handles duration mismatch without looping:
        - If audio is longer: freeze on last frame until audio ends
        - If video is longer: pad audio with silence

        Args:
            video_path: Path to the rendered video
            audio_path: Path to the audio narration
            video_speed: Speed factor for video (0.75 = 75% speed, slower). Default 0.75.
            srt_path: Optional path to SRT subtitle file to burn into video

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

        # Get durations of both streams
        video_duration = await self._get_media_duration(video_path)
        audio_duration = await self._get_media_duration(audio_path)

        # Apply video speed adjustment (0.75 = 75% speed = 1.33x longer duration)
        # setpts=PTS/speed slows down the video
        speed_filter = ""
        adjusted_video_duration = video_duration
        if video_speed != 1.0 and video_speed > 0:
            adjusted_video_duration = video_duration / video_speed
            speed_filter = f"setpts=PTS/{video_speed},"
            logger.info(
                "Slowing video to %.0f%% speed: %.1fs -> %.1fs",
                video_speed * 100, video_duration, adjusted_video_duration
            )

        # Build subtitle filter if SRT provided
        # Add bottom padding for subtitle area so text doesn't overlap with rendered content
        subtitle_filter = ""
        padding_filter = ""
        if srt_path and os.path.exists(srt_path):
            # Escape path for ffmpeg filter (colons and backslashes need escaping)
            escaped_path = srt_path.replace("\\", "\\\\").replace(":", "\\:")
            # Add 60px black bar at bottom for subtitle area
            padding_filter = "pad=iw:ih+60:0:0:black,"
            # Subtitles sit in the padded area with small margin from bottom edge
            subtitle_filter = f",subtitles='{escaped_path}':force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Shadow=1,MarginV=12'"
            logger.info("Burning subtitles from: %s (with 60px bottom padding)", srt_path)

        logger.info(
            "Mixing audio into video: %s (%.1fs adj) + %s (%.1fs) -> %s",
            video_path, adjusted_video_duration, audio_path, audio_duration, output_path
        )

        # Handle duration mismatch - NEVER loop
        # - Freeze last frame if audio is longer
        # - Pad audio with silence if video is longer
        if audio_duration > adjusted_video_duration and adjusted_video_duration > 0:
            # Audio is longer: freeze on last frame (tpad with stop_mode=clone)
            pad_duration = audio_duration - adjusted_video_duration
            logger.info("Freezing last frame for %.1fs to match audio duration", pad_duration)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex",
                f"[0:v]{speed_filter}{padding_filter}tpad=stop_mode=clone:stop_duration={pad_duration}{subtitle_filter}[v]",
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast",
                "-profile:v", "main", "-level", "3.1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-shortest",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Video is longer or equal: apply speed filter and pad audio with silence
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex",
                f"[0:v]{speed_filter}{padding_filter}null{subtitle_filter}[v];[1:a]apad=whole_dur={adjusted_video_duration}[a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast",
                "-profile:v", "main", "-level", "3.1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("ffmpeg failed: %s", stderr.decode() if stderr else "unknown error")
            return video_path  # Return original on failure

        logger.info("Audio mixed successfully (mobile-compatible): %s", output_path)
        return output_path

    async def _slow_down_video(self, video_path: str, video_speed: float = 0.60) -> str:
        """Slow down video without audio using ffmpeg.

        Args:
            video_path: Path to the rendered video
            video_speed: Speed factor (0.60 = 60% speed, slower). Default 0.60.

        Returns:
            Path to the slowed video
        """
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found, cannot slow down video")
            return video_path

        if video_speed == 1.0 or video_speed <= 0:
            return video_path

        output_path = video_path.replace(".mp4", "_slowed.mp4")
        if output_path == video_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_slowed{ext}"

        logger.info(
            "Slowing video to %.0f%% speed: %s -> %s",
            video_speed * 100, video_path, output_path
        )

        # Apply speed filter: setpts=PTS/speed slows down the video
        # Include mobile-compatible encoding settings
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter:v", f"setpts=PTS/{video_speed}",
            "-an",  # No audio
            "-c:v", "libx264", "-preset", "fast",
            "-profile:v", "main", "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("ffmpeg slowdown failed: %s", stderr.decode() if stderr else "unknown error")
            return video_path  # Return original on failure

        logger.info("Video slowed successfully (mobile-compatible): %s", output_path)
        return output_path

    async def _transcode_for_mobile(self, video_path: str) -> str:
        """Transcode video to mobile-compatible H.264 format.

        Ensures the video uses:
        - H.264 Main profile (iOS Safari compatible)
        - Level 3.1 (good mobile compatibility)
        - yuv420p pixel format (required for mobile)
        - faststart flag (moov atom at beginning for streaming)

        Args:
            video_path: Path to the rendered video

        Returns:
            Path to the mobile-compatible video
        """
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found, cannot transcode for mobile")
            return video_path

        output_path = video_path.replace(".mp4", "_mobile.mp4")
        if output_path == video_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_mobile{ext}"

        logger.info("Transcoding for mobile compatibility: %s -> %s", video_path, output_path)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264", "-preset", "fast",
            "-profile:v", "main", "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",  # Also ensure audio is AAC if present
            output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("Mobile transcode failed: %s", stderr.decode() if stderr else "unknown error")
            return video_path  # Return original on failure

        logger.info("Video transcoded for mobile: %s", output_path)
        return output_path

    async def _get_media_duration(self, path: str) -> float:
        """Get duration of a media file using ffprobe."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            return float(stdout.decode().strip())
        except Exception as e:
            logger.warning("Failed to get duration for %s: %s", path, e)
            return 0.0
