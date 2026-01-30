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
            # 0. Generate narration script FIRST if audio requested (for code-audio sync)
            narration_script = None
            audio_task = None
            if params.audio and self.tts:
                await report("generating", 3)
                logger.info("Generating narration script first for code-audio sync")
                narration_script = await self.tts.generate_narration_script(prompt)
                logger.info("Generated %d-sentence narration script", len(narration_script))

                # Start TTS generation in parallel with code generation
                # TTS internally parallelizes across sentences
                audio_task = asyncio.create_task(
                    self._generate_audio_from_script(narration_script, render_id, params.voice)
                )
                logger.info("Started parallel TTS generation")

            # 1. Generate code via LLM, guided by narration script (0-20%)
            # Runs in parallel with TTS if audio requested
            await report("generating", 5)
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            await report("generating", 10)
            code = await self._generate_and_validate(prompt, narration_script)
            await report("validating", 20)

            # 2. Store generated code (20-25%)
            scenes = self.scene_parser.parse_scenes(code)
            scene_name = scenes[0].name if scenes else None
            await self.tracker.update_render(
                render_id, source_code=code, scene_name=scene_name,
            )
            await report("preparing", 25)

            # 3. Render video (25-60%)
            await report("rendering", 30)
            result = await self._render_with_retries(render_id, code, scene_name, params, prompt)
            await report("uploading", 60)

            # 4. Wait for parallel TTS and mix audio (60-90%)
            has_audio = False
            if audio_task and result.local_path:
                try:
                    await report("generating_audio", 65)
                    # Await the parallel TTS task
                    audio_path = await audio_task
                    await report("mixing_audio", 80)
                    if audio_path:
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path
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
            # 0. Generate narration script FIRST if audio requested (for code-audio sync)
            narration_script = None
            audio_task = None
            if params.audio and self.tts:
                await report("generating", 3)
                logger.info("Generating narration script first for code-audio sync")
                narration_script = await self.tts.generate_narration_script(prompt)
                logger.info("Generated %d-sentence narration script", len(narration_script))

                # Start TTS generation in parallel with code generation
                # TTS internally parallelizes across sentences
                audio_task = asyncio.create_task(
                    self._generate_audio_from_script(narration_script, render_id, params.voice)
                )
                logger.info("Started parallel TTS generation")

            # 1. Run multi-agent pipeline (0-20%)
            # Runs in parallel with TTS if audio requested
            await report("analyzing", 5)
            await self.tracker.update_render(render_id, status=RenderStatus.generating)
            await report("planning", 10)
            pipeline_result = await self.orchestrator.generate_advanced(prompt, narration_script)
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

            # 4. Render video (25-60%)
            await report("rendering", 30)
            result = await self._render_and_upload(render_id, code, scene_name, params)
            await report("uploading", 60)

            # 5. Wait for parallel TTS and mix audio (60-90%)
            has_audio = False
            if audio_task and result.local_path:
                try:
                    await report("generating_audio", 65)
                    # Await the parallel TTS task
                    audio_path = await audio_task
                    await report("mixing_audio", 80)
                    if audio_path:
                        mixed_path = await self._mix_audio_video(
                            result.local_path, audio_path
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

            await report("finalizing", 95)

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
            transformed_code = bridge_code(current_code)

            try:
                result = await self._render_and_upload(render_id, current_code, scene_name, params)
                # If we're on a retry and it succeeded, store the error→fix pattern
                if previous_error and self.rag:
                    previous_transformed = bridge_code(previous_code) if previous_code else None
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
                        attempt, self.config.gemini_max_retries, error_msg[:200]
                    )
                    # Track error for learning (before fixing)
                    previous_error = error_msg
                    previous_code = current_code
                    # Ask LLM to fix the code based on runtime error
                    await self.tracker.update_render(render_id, status=RenderStatus.generating)
                    current_code = await self.llm.fix_code(current_code, [error_msg[:1500]])

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
                        self.config.gemini_max_retries, error_msg[:500]
                    )
                    # Store error pattern for future learning (with both original and transformed)
                    if self.rag:
                        await self.rag.store_error_pattern(
                            error_message=error_msg[:500],
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
        """Format RAG results as context for generation."""
        lines = []
        for i, scene in enumerate(similar_scenes[:3], 1):
            meta = scene.get("metadata", {})
            prompt_hint = meta.get("prompt", "")[:100]
            if prompt_hint:
                lines.append(f"\nExample {i} (prompt: {prompt_hint}):")
            else:
                lines.append(f"\nExample {i}:")
            # Show code snippet (first 800 chars)
            code = scene.get("content", "")[:800]
            if code:
                lines.append(f"```python\n{code}\n```")
        return "\n".join(lines)

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

    async def _mix_audio_video(self, video_path: str, audio_path: str) -> str:
        """Mix audio track into video using ffmpeg.

        Uses the LONGER of video/audio duration to avoid clipping either:
        - If audio is longer: loops video to match audio length
        - If video is longer: pads audio with silence

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

        # Get durations of both streams
        video_duration = await self._get_media_duration(video_path)
        audio_duration = await self._get_media_duration(audio_path)

        logger.info(
            "Mixing audio into video: %s (%.1fs) + %s (%.1fs) -> %s",
            video_path, video_duration, audio_path, audio_duration, output_path
        )

        # Use filter_complex to handle duration mismatch
        # - Loop video if audio is longer
        # - Pad audio with silence if video is longer
        if audio_duration > video_duration and video_duration > 0:
            # Audio is longer: loop video to match audio duration
            loop_count = int(audio_duration / video_duration) + 1
            logger.info("Looping video %dx to match audio duration", loop_count)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-stream_loop", str(loop_count - 1),
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                "-t", str(audio_duration),  # Trim to exact audio duration
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Video is longer or equal: pad audio with silence
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-filter_complex", f"[1:a]apad=whole_dur={video_duration}[a]",
                "-map", "0:v",
                "-map", "[a]",
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
