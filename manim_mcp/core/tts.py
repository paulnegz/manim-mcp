"""Text-to-speech service using Gemini's native audio generation."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from typing import TYPE_CHECKING

from pydub import AudioSegment

if TYPE_CHECKING:
    from google import genai
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)


class GeminiTTSService:
    """Text-to-speech using Gemini's native audio generation."""

    def __init__(self, config: ManimMCPConfig) -> None:
        from google import genai as genai_module

        self.client = genai_module.Client(api_key=config.gemini_api_key)
        self.model = config.tts_model  # For TTS audio generation
        self.narration_model = config.gemini_model  # For narration script generation
        self.voice = config.tts_voice
        self.pause_ms = config.tts_pause_ms
        self.max_concurrent = config.tts_max_concurrent

    async def generate_narration_script(self, prompt: str) -> list[str]:
        """Generate narration script from animation prompt, split into sentences.

        This script will be used to guide code generation - code should match this narration.
        """
        response = await self.client.aio.models.generate_content(
            model=self.narration_model,
            contents=f"""Create an educational narration script (6-10 sentences) for a math animation video about:

{prompt}

The narration should:
1. Start with an introduction to the topic
2. Describe each visual step that should appear (e.g., "First, let's draw a coordinate plane")
3. Explain the mathematical concepts as they would appear visually
4. End with a summary or key takeaway

Rules:
- Write as if narrating a 3Blue1Brown-style video
- Each sentence describes ONE visual step or concept
- Be specific about what should appear (axes, vectors, equations, etc.)
- Keep sentences under 25 words each
- Use educational, engaging language

Return ONLY the narration text, one sentence per line. Each sentence will correspond to a scene in the animation.""",
        )
        text = response.text.strip()
        # Split into sentences, filter empty
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        # Ensure each sentence ends with punctuation for natural speech
        sentences = [s if s.endswith((".", "!", "?")) else s + "." for s in sentences]
        return sentences[:10]  # Max 10 sentences

    async def generate_narration_for_duration(
        self, prompt: str, target_duration: float, code: str | None = None
    ) -> list[str]:
        """Generate narration script from code and comments, paced to fit video duration.

        Args:
            prompt: Original animation prompt
            target_duration: Target duration in seconds to fit narration into
            code: Generated Manim code with comments (primary source for narration)

        Returns:
            List of sentences paced to fit the duration
        """
        # Estimate ~150 words per minute for natural speech = 2.5 words/second
        # With pauses between sentences, aim for ~2 words/second
        target_word_count = int(target_duration * 2.0)

        # Calculate sentence count (avg 12 words/sentence)
        sentence_count = max(2, min(10, target_word_count // 12))

        if code:
            # Generate narration from code structure and comments
            response = await self.client.aio.models.generate_content(
                model=self.narration_model,
                contents=f"""You are narrating a 3Blue1Brown-style math animation video.

ANIMATION CODE:
```python
{code}
```

ORIGINAL PROMPT: {prompt}

YOUR TASK:
Read the code above carefully. Based on:
1. The comments in the code (they describe what's happening)
2. The animation calls (self.play, ShowCreation, Transform, etc.)
3. The objects being created and manipulated

Generate a narration script that explains what the viewer SEES on screen.

TIMING CONSTRAINT:
- Video duration: {target_duration:.1f} seconds
- Target: ~{target_word_count} words ({sentence_count} sentences)
- Each sentence matches one visual moment in the animation

RULES:
- Follow the ORDER of animations in the code
- Use comments as hints but write natural spoken sentences
- Describe what appears, moves, transforms on screen
- Educational, engaging 3Blue1Brown style
- Don't mention code, variables, or technical implementation

Return ONLY the narration, one sentence per line.""",
            )
        else:
            # Fallback: generate from prompt only
            response = await self.client.aio.models.generate_content(
                model=self.narration_model,
                contents=f"""Create an educational narration script for a math animation video about:

{prompt}

TIMING CONSTRAINT:
- Video duration: {target_duration:.1f} seconds
- Target: ~{target_word_count} words ({sentence_count} sentences)

Rules:
- Write as if narrating a 3Blue1Brown-style video
- Each sentence describes what's happening visually
- Natural speaking pace
- Educational but concise

Return ONLY the narration text, one sentence per line.""",
            )

        text = response.text.strip()
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        sentences = [s if s.endswith((".", "!", "?")) else s + "." for s in sentences]
        return sentences[:sentence_count]

    async def text_to_speech(self, text: str) -> tuple[bytes, str]:
        """Convert single sentence to audio bytes using Gemini TTS.

        Returns:
            Tuple of (audio_data, mime_type)
        """
        from google.genai import types

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        # Audio data is in the response
        inline_data = response.candidates[0].content.parts[0].inline_data
        audio_data = inline_data.data
        mime_type = getattr(inline_data, "mime_type", "audio/wav")

        logger.debug("TTS returned mime_type: %s, data length: %d", mime_type, len(audio_data) if audio_data else 0)

        # Decode if base64 encoded (string), otherwise use as-is (bytes)
        if isinstance(audio_data, str):
            audio_data = base64.b64decode(audio_data)
        return audio_data, mime_type

    async def generate_parallel(self, sentences: list[str]) -> list[tuple[bytes, str] | Exception]:
        """Generate audio for sentences in parallel with concurrency limit.

        Returns:
            List of (audio_data, mime_type) tuples or Exceptions
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_tts(text: str) -> tuple[bytes, str]:
            async with semaphore:
                return await self.text_to_speech(text)

        tasks = [limited_tts(s) for s in sentences]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(results)

    def _parse_mime_type(self, mime_type: str) -> tuple[str, int]:
        """Parse MIME type and extract format and sample rate.

        Returns:
            Tuple of (format_string, sample_rate)
        """
        # Extract base mime type (before semicolon)
        base_mime = mime_type.split(";")[0].strip()

        # Extract sample rate from parameters if present
        sample_rate = 24000  # Default for Gemini TTS
        if "rate=" in mime_type:
            try:
                rate_part = [p for p in mime_type.split(";") if "rate=" in p][0]
                sample_rate = int(rate_part.split("=")[1].strip())
            except (IndexError, ValueError):
                pass

        mime_map = {
            "audio/wav": "wav",
            "audio/wave": "wav",
            "audio/x-wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/opus": "ogg",
            "audio/webm": "webm",
            "audio/flac": "flac",
            "audio/aac": "aac",
            "audio/L16": "raw",  # Raw PCM 16-bit
            "audio/pcm": "raw",
        }
        return mime_map.get(base_mime, "wav"), sample_rate

    def stitch_audio_for_duration(
        self, segments: list[tuple[bytes, str] | Exception], target_duration: float
    ) -> bytes:
        """Combine audio segments and adjust to fit target duration.

        If audio is shorter than target: add pauses between sentences
        If audio is longer than target: speed up slightly (max 1.3x)

        Args:
            segments: Audio segments from parallel TTS
            target_duration: Target duration in seconds

        Returns:
            WAV bytes adjusted to approximately match target duration
        """
        # First, stitch without duration constraint to get base audio
        combined = AudioSegment.empty()

        valid_segments = []
        for i, segment_data in enumerate(segments):
            if isinstance(segment_data, Exception):
                logger.warning("Skipping failed audio segment %d: %s", i, segment_data)
                continue
            try:
                audio_bytes, mime_type = segment_data
                fmt, sample_rate = self._parse_mime_type(mime_type)

                if fmt == "raw":
                    audio = AudioSegment.from_raw(
                        io.BytesIO(audio_bytes),
                        sample_width=2,
                        frame_rate=sample_rate,
                        channels=1,
                    )
                else:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)

                valid_segments.append(audio)
            except Exception as e:
                logger.warning("Failed to process audio segment %d: %s", i, e)

        if not valid_segments:
            raise ValueError("No valid audio segments to stitch")

        # Calculate total audio duration without pauses
        total_audio_ms = sum(len(seg) for seg in valid_segments)
        target_ms = target_duration * 1000

        # Calculate pause duration to distribute between segments
        if len(valid_segments) > 1:
            available_pause_time = target_ms - total_audio_ms
            pause_per_gap = max(1300, min(2000, available_pause_time / (len(valid_segments) - 1)))
        else:
            pause_per_gap = 0

        # If audio is too long even with minimal pauses, speed it up
        min_total = total_audio_ms + (len(valid_segments) - 1) * 1300  # 1300ms min pause
        if min_total > target_ms:
            # Need to speed up - calculate required factor (max 1.3x)
            speed_factor = min(1.3, min_total / target_ms)
            logger.info("Speeding up audio by %.2fx to fit target duration", speed_factor)
            # Speed up by changing frame rate
            valid_segments = [
                seg._spawn(seg.raw_data, overrides={
                    "frame_rate": int(seg.frame_rate * speed_factor)
                }).set_frame_rate(seg.frame_rate)
                for seg in valid_segments
            ]
            # Recalculate pause time
            total_audio_ms = sum(len(seg) for seg in valid_segments)
            if len(valid_segments) > 1:
                available_pause_time = target_ms - total_audio_ms
                pause_per_gap = max(1300, available_pause_time / (len(valid_segments) - 1))

        # Build final audio with calculated pauses
        silence = AudioSegment.silent(duration=int(pause_per_gap))
        for i, audio in enumerate(valid_segments):
            if i > 0:
                combined += silence
            combined += audio

        # If still shorter than target, add silence at end
        if len(combined) < target_ms:
            combined += AudioSegment.silent(duration=int(target_ms - len(combined)))

        logger.info(
            "Audio adjusted: %d segments, %.1fs target, %.1fs result, %.0fms pauses",
            len(valid_segments), target_duration, len(combined) / 1000, pause_per_gap
        )

        output = io.BytesIO()
        combined.export(output, format="wav")
        return output.getvalue()

    def stitch_audio(self, segments: list[tuple[bytes, str] | Exception]) -> bytes:
        """Combine audio segments with pauses, return WAV bytes."""
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=self.pause_ms)

        for i, segment_data in enumerate(segments):
            if isinstance(segment_data, Exception):
                logger.warning("Skipping failed audio segment %d: %s", i, segment_data)
                continue
            try:
                audio_bytes, mime_type = segment_data
                fmt, sample_rate = self._parse_mime_type(mime_type)
                logger.debug("Processing segment %d: mime=%s, format=%s, rate=%d, size=%d",
                             i, mime_type, fmt, sample_rate, len(audio_bytes))

                # For raw PCM, we need to specify parameters
                if fmt == "raw":
                    audio = AudioSegment.from_raw(
                        io.BytesIO(audio_bytes),
                        sample_width=2,  # 16-bit (L16)
                        frame_rate=sample_rate,
                        channels=1,
                    )
                else:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)

                if i > 0 and len(combined) > 0:
                    combined += silence
                combined += audio
            except Exception as e:
                logger.warning("Failed to process audio segment %d: %s", i, e)
                continue

        if len(combined) == 0:
            raise ValueError("No valid audio segments to stitch")

        # Export as WAV
        output = io.BytesIO()
        combined.export(output, format="wav")
        return output.getvalue()

    async def generate_full_narration(self, prompt: str, script: list[str] | None = None) -> bytes:
        """Full pipeline: prompt → script → parallel TTS → stitched audio.

        Args:
            prompt: Animation prompt (used if script not provided)
            script: Pre-generated narration script (if already generated for code sync)
        """
        if script:
            sentences = script
            logger.info("Using provided narration script with %d sentences", len(sentences))
        else:
            logger.info("Generating narration script for prompt")
            sentences = await self.generate_narration_script(prompt)
        logger.info("Generated %d sentences for narration", len(sentences))

        logger.info("Generating audio for %d sentences in parallel", len(sentences))
        audio_segments = await self.generate_parallel(sentences)

        successful = sum(1 for s in audio_segments if not isinstance(s, Exception))
        logger.info("Successfully generated %d/%d audio segments", successful, len(sentences))

        return self.stitch_audio(audio_segments)
