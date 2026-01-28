"""CodeGeneratorAgent: Generates Manim code from scene plans with RAG context."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan

logger = logging.getLogger(__name__)


def _build_system_prompt(latex_available: bool) -> str:
    """Build the code generation system prompt."""
    latex_section = ""
    if latex_available:
        latex_section = """\
- Use proper LaTeX with raw strings: MathTex(r"E = mc^2")
- Use Text() for non-math text, Tex() for mixed LaTeX text"""
    else:
        latex_section = """\
- IMPORTANT: LaTeX is NOT installed. Do NOT use MathTex, Tex, or any LaTeX-dependent class.
- Use Text() for ALL text, including math: Text("E = mc²", font_size=28)
- Use Unicode symbols for math: ², ³, ≠, ≤, ≥, ∑, ∫, π, θ, Δ, →, ∞, √, ±"""

    return f"""\
You are an expert Manim Community Edition animator. Generate complete, executable
Python code based on the provided scene plan.

Requirements:
- Use Manim Community Edition: `from manim import *`
- Create exactly ONE Scene subclass with a descriptive CamelCase name
- Implement the `construct(self)` method with all animation logic
- Follow the scene plan's segments for structure and timing
{latex_section}
- Include smooth animations: Create, Write, FadeIn, FadeOut, Transform, ReplacementTransform
- Add self.wait() calls between sections for pacing
- Use color effectively: BLUE, RED, YELLOW, GREEN, WHITE, GOLD, TEAL, PURPLE
- Only import from manim, numpy, and math

CRITICAL - Prevent overlapping text and elements:
- Use small font sizes: font_size=28 for body, font_size=36 for titles
- ALWAYS call self.play(FadeOut(*self.mobjects)) before showing new content in same area
- Use .scale_to_fit_width(config.frame_width - 1) for any wide text
- Stack text with VGroup(*items).arrange(DOWN, buff=0.4)
- Place titles at .to_edge(UP, buff=0.5)
- Use .next_to(other, DOWN, buff=0.3) for vertical stacking
- Maximum 3-4 text elements visible at once
- Clear screen between sections

Layout rules:
- Title: text.to_edge(UP, buff=0.5)
- Content: group.move_to(ORIGIN) or group.next_to(title, DOWN, buff=0.5)
- Side by side: VGroup(a, b).arrange(RIGHT, buff=1.0)
- Vertical list: VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

Quality guidelines:
- Match segment durations with appropriate wait() calls
- Use the suggested mobjects and animations from the plan
- Create smooth transitions between segments
- End with FadeOut or a clean final frame

Return ONLY the Python code. No markdown fences. No explanations."""


class CodeGeneratorAgent(BaseAgent):
    """Generates Manim code from scene plans, using RAG for few-shot examples."""

    name = "code_generator"

    async def process(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
    ) -> str:
        """Generate Manim code based on the scene plan.

        Args:
            prompt: Original user prompt
            analysis: Concept analysis from first agent
            plan: Scene plan from second agent

        Returns:
            Generated Manim Python code
        """
        logger.debug("Generating code for: %s (%d segments)",
                     plan.title, len(plan.segments))

        # Get RAG examples for few-shot context
        rag_context = ""
        if self.rag_available and plan.rag_examples:
            rag_context = await self._get_rag_examples(prompt, analysis)

        # Build generation prompt
        gen_prompt = self._build_prompt(prompt, analysis, plan, rag_context)

        # Generate code
        system = _build_system_prompt(self.config.latex_available)
        code = await self._llm_call(gen_prompt, system)

        return self._strip_fences(code)

    async def _get_rag_examples(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> str:
        """Retrieve relevant code examples from RAG."""
        if not self.rag_available:
            return ""

        # Search for similar scenes
        results = await self.rag.search_similar_scenes(
            query=f"{analysis.domain.value}: {prompt}",
            n_results=2,
        )

        if not results:
            return ""

        examples = []
        for i, result in enumerate(results[:2], 1):
            code = result.get("content", "")
            if len(code) > 1500:
                code = code[:1500] + "\n# ... (truncated)"
            examples.append(f"Example {i}:\n```python\n{code}\n```")

        return "\n\n".join(examples)

    def _build_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
        rag_context: str,
    ) -> str:
        """Build the code generation prompt with all context."""
        parts = [
            f"Create a Manim animation: {prompt}",
            "",
            f"Scene: {plan.title}",
            f"Total duration: ~{plan.total_duration:.1f} seconds",
            "",
            "Segments:",
        ]

        for i, seg in enumerate(plan.segments, 1):
            parts.append(f"{i}. {seg.name} ({seg.duration:.1f}s)")
            parts.append(f"   - {seg.description}")
            if seg.mobjects:
                parts.append(f"   - Objects: {', '.join(seg.mobjects)}")
            if seg.animations:
                parts.append(f"   - Animations: {', '.join(seg.animations)}")

        if analysis.visual_elements:
            parts.extend([
                "",
                f"Suggested visual elements: {', '.join(analysis.visual_elements)}",
            ])

        if rag_context:
            parts.extend([
                "",
                "Reference examples (use as inspiration, not copy):",
                rag_context,
            ])

        return "\n".join(parts)
