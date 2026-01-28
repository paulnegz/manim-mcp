"""Multi-provider LLM client for Manim code generation.

Supports:
- Google Gemini (default)
- Anthropic Claude
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    gemini = "gemini"
    claude = "claude"


# ── System Prompts ────────────────────────────────────────────────────

_GENERATE_BASE = """\
You are an expert Manim Community Edition animator. Given a text description,
generate complete, executable Python code that creates a beautiful mathematical
or educational animation.

Requirements:
- Use Manim Community Edition: `from manim import *`
- Create exactly ONE Scene subclass with a descriptive CamelCase name
- Implement the `construct(self)` method with all animation logic
{latex_instructions}
- Include smooth animations: Create, Write, FadeIn, FadeOut, Transform, ReplacementTransform, etc.
- Add self.wait() calls between sections for pacing (0.5-2 seconds)
- Use color effectively: BLUE, RED, YELLOW, GREEN, WHITE, GOLD, TEAL, PURPLE
- Structure: introduction → core explanation → conclusion
- Target 10-30 seconds total duration
- Only import from manim, numpy, and math

CRITICAL - Avoid overlapping text and elements:
- Use small font sizes: font_size=28 for body text, font_size=36 for titles
- ALWAYS use .scale_to_fit_width(config.frame_width - 1) for wide text
- ALWAYS FadeOut old elements BEFORE showing new ones in the same area
- Use VGroup with .arrange(DOWN, buff=0.4) to stack text vertically
- Position titles at .to_edge(UP, buff=0.5), content at ORIGIN or below
- Use .next_to(other, DOWN, buff=0.3) to place elements below each other
- For multiple items, use .arrange(DOWN, buff=0.3, aligned_edge=LEFT)
- Keep equations centered, explanatory text to the sides
- Maximum 3-4 lines of text on screen at once
- Clear the screen between major sections: self.play(FadeOut(*self.mobjects))

Layout patterns:
- Title at top: title.to_edge(UP, buff=0.5)
- Main content centered: content.move_to(ORIGIN)
- Labels below objects: label.next_to(obj, DOWN, buff=0.2)
- Side-by-side: VGroup(left, right).arrange(RIGHT, buff=1.0)
- Stacked list: VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT).scale_to_fit_height(5)

Common patterns:
- Title card: Text("Title", font_size=36).to_edge(UP, buff=0.5)
{latex_patterns}
- Graph: axes.plot(lambda x: f(x), color=BLUE)
- Cleanup: self.play(FadeOut(*self.mobjects)) before new sections

Return ONLY the Python code. No markdown fences. No explanations. No comments
outside the code."""

_LATEX_INSTRUCTIONS = """\
- Use proper LaTeX with raw strings: MathTex(r"E = mc^2")
- Use Text() for non-math text, Tex() for mixed LaTeX text"""

_NO_LATEX_INSTRUCTIONS = """\
- IMPORTANT: LaTeX is NOT installed. Do NOT use MathTex, Tex, or any LaTeX-dependent class.
- Use Text() for ALL text, including math: Text("E = mc²", font_size=36)
- Use Unicode symbols for math: ², ³, ≠, ≤, ≥, ∑, ∫, π, θ, Δ, →, ∞, √, ±
- For formulas, spell them out or use Unicode: Text("F = ma"), Text("a² + b² = c²")"""

_LATEX_PATTERNS = """\
- Equation reveal: Write(equation) or FadeIn(equation)
- Step-by-step: Transform old_eq into new_eq to show derivation steps
- Highlight: Indicate(obj, color=YELLOW) or SurroundingRectangle(obj)"""

_NO_LATEX_PATTERNS = """\
- Equation reveal: Write(Text("E = mc²")) or FadeIn(text_obj)
- Step-by-step: Transform old_text into new_text to show derivation steps
- Highlight: Indicate(obj, color=YELLOW) or SurroundingRectangle(obj)"""


def _build_generate_system(latex_available: bool) -> str:
    if latex_available:
        return _GENERATE_BASE.format(
            latex_instructions=_LATEX_INSTRUCTIONS,
            latex_patterns=_LATEX_PATTERNS,
        )
    return _GENERATE_BASE.format(
        latex_instructions=_NO_LATEX_INSTRUCTIONS,
        latex_patterns=_NO_LATEX_PATTERNS,
    )


EDIT_SYSTEM = """\
You are an expert Manim Community Edition animator. You will receive existing
Manim animation code and edit instructions. Modify the code to fulfill the
instructions while preserving everything else that works.

Rules:
- Keep the same Scene class name and overall structure
- Make only the changes requested — do not rewrite unrelated parts
- Preserve working animations and transitions
- Maintain imports, spacing, and code quality
- Only import from manim, numpy, and math

Return ONLY the modified Python code. No markdown fences. No explanations."""

FIX_SYSTEM = """\
You are an expert Manim Community Edition developer. The following code has
validation errors. Fix the code to resolve them while preserving the intended
animation.

Return ONLY the fixed Python code. No markdown fences. No explanations."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wraps its output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ── Abstract Base Client ──────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: ManimMCPConfig) -> None:
        self.config = config
        self._generate_system = _build_generate_system(config.latex_available)

    @abstractmethod
    async def generate_code(self, prompt: str) -> str:
        """Generate Manim code from a text description."""
        ...

    @abstractmethod
    async def edit_code(self, original_code: str, instructions: str) -> str:
        """Edit existing Manim code based on instructions."""
        ...

    @abstractmethod
    async def fix_code(self, code: str, errors: list[str]) -> str:
        """Fix code that failed validation."""
        ...

    async def _retry_with_backoff(
        self,
        coro_func,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """Retry an async operation with exponential backoff.

        Uses progressive backoff: 1s, 2s, 4s, etc.
        Handles rate limits specially with longer delays.
        """
        from manim_mcp.exceptions import LLMMaxRetriesError, LLMRateLimitError

        last_error = None
        last_error_msg = None

        for attempt in range(max_retries):
            try:
                return await coro_func()
            except Exception as e:
                last_error = e
                last_error_msg = str(e)

                # Check for rate limit errors (common across providers)
                is_rate_limit = any(x in str(e).lower() for x in [
                    "rate_limit", "rate limit", "429", "quota", "too many requests"
                ])

                if is_rate_limit:
                    # Longer delay for rate limits
                    delay = base_delay * (3 ** attempt) + 5
                    logger.warning(
                        "LLM rate limit hit (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1, max_retries, delay, e
                    )
                elif attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, max_retries, delay, e
                    )
                else:
                    # Last attempt, don't sleep
                    break

                await asyncio.sleep(delay)

        # Wrap in our custom exception for better error handling
        raise LLMMaxRetriesError(
            f"LLM call failed after {max_retries} attempts",
            attempts=max_retries,
            last_error=last_error_msg,
        ) from last_error


# ── Gemini Client ─────────────────────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client."""

    def __init__(self, config: ManimMCPConfig) -> None:
        super().__init__(config)
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_name = config.gemini_model

    async def generate_code(self, prompt: str) -> str:
        """Generate Manim code from a text description."""
        async def _call():
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=self._generate_system,
                ),
            )
            return _strip_fences(response.text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)

    async def edit_code(self, original_code: str, instructions: str) -> str:
        """Edit existing Manim code based on instructions."""
        user_prompt = f"ORIGINAL CODE:\n{original_code}\n\nEDIT INSTRUCTIONS:\n{instructions}"

        async def _call():
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=EDIT_SYSTEM,
                ),
            )
            return _strip_fences(response.text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)

    async def fix_code(self, code: str, errors: list[str]) -> str:
        """Fix code that failed validation."""
        error_text = "\n".join(f"- {e}" for e in errors)
        user_prompt = f"CODE:\n{code}\n\nERRORS:\n{error_text}"

        async def _call():
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=FIX_SYSTEM,
                ),
            )
            return _strip_fences(response.text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)


# ── Claude Client ─────────────────────────────────────────────────────

class ClaudeClient(BaseLLMClient):
    """Anthropic Claude LLM client."""

    def __init__(self, config: ManimMCPConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.claude_api_key)
            self.model_name = config.claude_model
            self._available = True
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    async def generate_code(self, prompt: str) -> str:
        """Generate Manim code from a text description."""
        async def _call():
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                system=self._generate_system,
                messages=[{"role": "user", "content": prompt}],
            )
            return _strip_fences(response.content[0].text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)

    async def edit_code(self, original_code: str, instructions: str) -> str:
        """Edit existing Manim code based on instructions."""
        user_prompt = f"ORIGINAL CODE:\n{original_code}\n\nEDIT INSTRUCTIONS:\n{instructions}"

        async def _call():
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                system=EDIT_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return _strip_fences(response.content[0].text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)

    async def fix_code(self, code: str, errors: list[str]) -> str:
        """Fix code that failed validation."""
        error_text = "\n".join(f"- {e}" for e in errors)
        user_prompt = f"CODE:\n{code}\n\nERRORS:\n{error_text}"

        async def _call():
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                system=FIX_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return _strip_fences(response.content[0].text)

        return await self._retry_with_backoff(_call, self.config.llm_max_retries)


# ── Factory ───────────────────────────────────────────────────────────

def create_llm_client(config: ManimMCPConfig) -> BaseLLMClient:
    """Create the appropriate LLM client based on configuration.

    Args:
        config: Application configuration

    Returns:
        LLM client instance (GeminiClient or ClaudeClient)

    Raises:
        ValueError: If configured provider is unknown or API key is missing
    """
    provider = LLMProvider(config.llm_provider)

    if provider == LLMProvider.claude:
        if not config.claude_api_key:
            raise ValueError(
                "MANIM_MCP_CLAUDE_API_KEY must be set when using Claude provider"
            )
        logger.info("Using Claude LLM provider (model: %s)", config.claude_model)
        return ClaudeClient(config)

    # Default to Gemini
    if not config.gemini_api_key:
        raise ValueError(
            "MANIM_MCP_GEMINI_API_KEY must be set when using Gemini provider"
        )
    logger.info("Using Gemini LLM provider (model: %s)", config.gemini_model)
    return GeminiClient(config)
