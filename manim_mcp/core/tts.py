"""Text-to-speech service using Gemini's native audio generation."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from typing import TYPE_CHECKING

from pydub import AudioSegment

from manim_mcp.prompts import (
    get_tts_narration_script,
    get_tts_narration_from_code,
    get_tts_narration_fallback,
)

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
            contents=get_tts_narration_script(prompt=prompt),
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
                contents=get_tts_narration_from_code(
                    code=code,
                    prompt=prompt,
                    target_duration=target_duration,
                    target_word_count=target_word_count,
                    sentence_count=sentence_count,
                ),
            )
        else:
            # Fallback: generate from prompt only
            response = await self.client.aio.models.generate_content(
                model=self.narration_model,
                contents=get_tts_narration_fallback(
                    prompt=prompt,
                    target_duration=target_duration,
                    target_word_count=target_word_count,
                    sentence_count=sentence_count,
                ),
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
    ) -> tuple[bytes, list[tuple[int, int]]]:
        """Combine audio segments and adjust to fit target duration.

        If audio is shorter than target: add pauses between sentences
        If audio is longer than target: speed up slightly (max 1.3x)

        Args:
            segments: Audio segments from parallel TTS
            target_duration: Target duration in seconds

        Returns:
            Tuple of (WAV bytes, subtitle_timings, valid_indices) where:
            - timings are list of (start_ms, end_ms)
            - valid_indices are original indices of segments that succeeded
        """
        # 1 second initial offset before audio/subtitles start
        initial_offset_ms = 1000

        valid_segments = []
        valid_indices = []  # Track original indices for subtitle correlation
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
                valid_indices.append(i)
            except Exception as e:
                logger.warning("Failed to process audio segment %d: %s", i, e)

        if not valid_segments:
            raise ValueError("No valid audio segments to stitch")

        # Calculate total audio duration without pauses (excluding initial offset)
        total_audio_ms = sum(len(seg) for seg in valid_segments)
        target_ms = target_duration * 1000

        # Account for initial offset in pause calculations
        available_for_content = target_ms - initial_offset_ms

        # Calculate pause duration to distribute between segments
        if len(valid_segments) > 1:
            available_pause_time = available_for_content - total_audio_ms
            pause_per_gap = max(1300, min(2000, available_pause_time / (len(valid_segments) - 1)))
        else:
            pause_per_gap = 0

        # If audio is too long even with minimal pauses, speed it up
        min_total = total_audio_ms + (len(valid_segments) - 1) * 1300  # 1300ms min pause
        if min_total > available_for_content:
            # Need to speed up - calculate required factor (max 1.3x)
            speed_factor = min(1.3, min_total / available_for_content)
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
                available_pause_time = available_for_content - total_audio_ms
                pause_per_gap = max(1300, available_pause_time / (len(valid_segments) - 1))

        # Build final audio with initial silence and calculated pauses
        # Also track subtitle timings
        combined = AudioSegment.silent(duration=initial_offset_ms)
        subtitle_timings: list[tuple[int, int]] = []
        current_position_ms = initial_offset_ms

        silence = AudioSegment.silent(duration=int(pause_per_gap))
        for i, audio in enumerate(valid_segments):
            if i > 0:
                combined += silence
                current_position_ms += int(pause_per_gap)

            # Record timing for this segment
            start_ms = current_position_ms
            end_ms = current_position_ms + len(audio)
            subtitle_timings.append((start_ms, end_ms))

            combined += audio
            current_position_ms = end_ms

        # If still shorter than target, add silence at end
        if len(combined) < target_ms:
            combined += AudioSegment.silent(duration=int(target_ms - len(combined)))

        logger.info(
            "Audio adjusted: %d segments, %.1fs target, %.1fs result, %.0fms pauses, %dms initial offset",
            len(valid_segments), target_duration, len(combined) / 1000, pause_per_gap, initial_offset_ms
        )

        output = io.BytesIO()
        combined.export(output, format="wav")
        return output.getvalue(), subtitle_timings, valid_indices

    @staticmethod
    def generate_srt(sentences: list[str], timings: list[tuple[int, int]], chunk_size: int = 4) -> str:
        """Generate SRT subtitle content with chunked display.

        Creates chunked subtitles where groups of words are shown together,
        then entirely replaced with the next chunk for easier reading.
        Each chunk displays for at least 1.2 seconds to ensure readability.

        Args:
            sentences: List of narration sentences
            timings: List of (start_ms, end_ms) tuples for each sentence
            chunk_size: Number of words per chunk (default 4)

        Returns:
            SRT-formatted subtitle string with chunked timing

        Example output (for "The quick brown fox jumps over" with chunk_size=4):
            1
            00:00:01,000 --> 00:00:02,200
            The quick brown fox

            2
            00:00:02,200 --> 00:00:03,400
            jumps over
        """
        # Minimum display time per chunk (ms) - ~300ms per word for comfortable reading
        MIN_CHUNK_DURATION_MS = 1200

        def ms_to_srt_time(ms: int) -> str:
            """Convert milliseconds to SRT timestamp format HH:MM:SS,mmm"""
            hours = ms // 3600000
            minutes = (ms % 3600000) // 60000
            seconds = (ms % 60000) // 1000
            millis = ms % 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        srt_lines = []
        entry_num = 1
        count = min(len(sentences), len(timings))

        for i in range(count):
            start_ms, end_ms = timings[i]
            text = sentences[i].strip()
            words = text.split()

            if not words:
                continue

            # Calculate timing per word within this sentence
            duration_ms = end_ms - start_ms
            num_chunks = (len(words) + chunk_size - 1) // chunk_size  # ceiling division

            # Calculate time per chunk, ensuring minimum readable duration
            time_per_chunk = max(MIN_CHUNK_DURATION_MS, duration_ms / num_chunks)

            chunk_start_ms = start_ms
            for j in range(num_chunks):
                chunk_start_word = j * chunk_size
                chunk_end_word = min((j + 1) * chunk_size, len(words))

                # End time: start + duration, but don't exceed sentence end
                chunk_end_ms = min(int(chunk_start_ms + time_per_chunk), end_ms)

                # Get the words for this chunk
                chunk_words = words[chunk_start_word:chunk_end_word]
                chunk_text = " ".join(chunk_words)

                srt_lines.append(str(entry_num))
                srt_lines.append(f"{ms_to_srt_time(int(chunk_start_ms))} --> {ms_to_srt_time(chunk_end_ms)}")
                srt_lines.append(chunk_text)
                srt_lines.append("")
                entry_num += 1

                # Next chunk starts where this one ends
                chunk_start_ms = chunk_end_ms

        return "\n".join(srt_lines)

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
