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
        self.model = config.tts_model
        self.voice = config.tts_voice
        self.pause_ms = config.tts_pause_ms
        self.max_concurrent = config.tts_max_concurrent

    async def generate_narration_script(self, prompt: str) -> list[str]:
        """Generate narration script from animation prompt, split into sentences."""
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Create a short educational narration script (5-8 sentences) for this animation:

{prompt}

Rules:
- Write as if narrating a video explanation
- Each sentence should be clear and standalone
- Keep sentences under 20 words each
- Use simple, educational language

Return ONLY the narration text, one sentence per line.""",
        )
        text = response.text.strip()
        # Split into sentences, filter empty
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        # Ensure each sentence ends with punctuation for natural speech
        sentences = [s if s.endswith((".", "!", "?")) else s + "." for s in sentences]
        return sentences[:10]  # Max 10 sentences

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

    async def generate_full_narration(self, prompt: str) -> bytes:
        """Full pipeline: prompt → script → parallel TTS → stitched audio."""
        logger.info("Generating narration script for prompt")
        sentences = await self.generate_narration_script(prompt)
        logger.info("Generated %d sentences for narration", len(sentences))

        logger.info("Generating audio for %d sentences in parallel", len(sentences))
        audio_segments = await self.generate_parallel(sentences)

        successful = sum(1 for s in audio_segments if not isinstance(s, Exception))
        logger.info("Successfully generated %d/%d audio segments", successful, len(sentences))

        return self.stitch_audio(audio_segments)
