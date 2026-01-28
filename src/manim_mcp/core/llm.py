"""Gemini LLM client for Manim code generation."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)

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
- Use VGroup to organize related objects
- Position elements clearly: UP, DOWN, LEFT, RIGHT, ORIGIN, with buff spacing
- For equations, use .next_to(), .shift(), or .move_to() for clear layout
- For graphs, use Axes() or NumberPlane() with proper labels

Common patterns:
- Title card: Text("Title", font_size=48).to_edge(UP)
{latex_patterns}
- Graph: axes.plot(lambda x: f(x), color=BLUE)
- Cleanup: FadeOut(*self.mobjects) before new sections

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


class GeminiClient:
    def __init__(self, config: ManimMCPConfig) -> None:
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_name = config.gemini_model
        self._generate_system = _build_generate_system(config.latex_available)

    async def generate_code(self, prompt: str) -> str:
        """Generate Manim code from a text description."""
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._generate_system,
            ),
        )
        return _strip_fences(response.text)

    async def edit_code(self, original_code: str, instructions: str) -> str:
        """Edit existing Manim code based on instructions."""
        user_prompt = f"ORIGINAL CODE:\n{original_code}\n\nEDIT INSTRUCTIONS:\n{instructions}"
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=EDIT_SYSTEM,
            ),
        )
        return _strip_fences(response.text)

    async def fix_code(self, code: str, errors: list[str]) -> str:
        """Fix code that failed validation."""
        error_text = "\n".join(f"- {e}" for e in errors)
        user_prompt = f"CODE:\n{code}\n\nERRORS:\n{error_text}"
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=FIX_SYSTEM,
            ),
        )
        return _strip_fences(response.text)
