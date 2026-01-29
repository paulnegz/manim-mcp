"""CodeGeneratorAgent: Generates Manim code from scene plans with RAG context."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan

logger = logging.getLogger(__name__)

# Threshold for "very high quality" match - use code directly
DIRECT_USE_THRESHOLD = 0.06


def _build_system_prompt(latex_available: bool) -> str:
    """Build the code generation system prompt for manimgl (3b1b style)."""
    return """\
You are Grant Sanderson (3Blue1Brown), creating a mathematical animation that builds intuition
through elegant visual storytelling. Generate complete, executable manimgl code.

Requirements:
- Use manimgl (3b1b's library): `from manimlib import *`
- Create exactly ONE Scene subclass with a descriptive CamelCase name
- Implement the `construct(self)` method with all animation logic
- Follow the scene plan's segments for structure and timing
- Use proper LaTeX: Tex(r"E = mc^2"), TexText("Hello")
- Only import from manimlib, numpy, and math

SCENE STRUCTURE (3Blue1Brown arc):
1. ESTABLISH (2-3s): Show what we're looking at
2. BUILD (main): Progressive reveal - never jump to the answer
3. INSIGHT (slow): Highlight the key moment
4. RESOLVE (2s+): Let the final state breathe

3BLUE1BROWN ANIMATION PATTERNS:
- Staggered reveals: LaggedStartMap(FadeIn, elements, lag_ratio=0.25)
- Show relationships: TransformFromCopy(source, target) - non-destructive
- Smooth transitions: FadeTransform(old, new) for text changes
- Synchronized: self.play(a.animate.shift(LEFT), b.animate.set_color(RED))
- Emphasis: Indicate(obj, color=YELLOW), FlashAround(result)

COLOR SEMANTICS (colors carry meaning):
- BLUE: Primary input, what we start with
- TEAL: Supporting elements
- GREEN: Transformation, the operation
- YELLOW/GOLD: Result, insight, "pay attention"
- RED: Constraint, warning
- GREY: Scaffolding (axes, labels)

PACING:
- Vary self.wait(): 0.5s for quick transitions, 1-2s for insights
- Use run_time=2+ for important transforms
- End with self.wait(2) - let it register
- Camera: frame.animate.reorient() for 3D, moves slowly

Return ONLY the Python code. No markdown fences. No explanations."""


class CodeGeneratorAgent(BaseAgent):
    """Generates Manim code from scene plans, using RAG for few-shot examples."""

    name = "code_generator"

    async def process(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
    ) -> tuple[str, str | None]:
        """Generate Manim code based on the scene plan.

        Args:
            prompt: Original user prompt
            analysis: Concept analysis from first agent
            plan: Scene plan from second agent

        Returns:
            Tuple of (generated_code, original_template_code_or_none)
        """
        logger.debug("Generating code for: %s (%d segments)",
                     plan.title, len(plan.segments))

        # Get RAG examples for few-shot context
        rag_context = ""
        high_quality_template = None
        if self.rag_available and plan.rag_examples:
            rag_context, high_quality_template = await self._get_rag_examples(prompt, analysis)

        # Store template code for reference
        template_code = high_quality_template.get("content") if high_quality_template else None

        # For VERY HIGH quality matches, use 3b1b code DIRECTLY (no conversion!)
        # manimgl will run it natively
        if high_quality_template:
            score = high_quality_template.get("similarity_score", 0)
            if score >= DIRECT_USE_THRESHOLD:
                code = self._use_directly(high_quality_template, plan)
                logger.info(
                    "[DIRECT-USE] Using 3b1b code directly for score=%.3f (%d chars)",
                    score, len(code)
                )
                return code, template_code

        # Build generation prompt
        gen_prompt = self._build_prompt(prompt, analysis, plan, rag_context, high_quality_template)

        # Generate code
        system = _build_system_prompt(self.config.latex_available)
        code = await self._llm_call(gen_prompt, system)

        return self._strip_fences(code), template_code

    def _use_directly(self, template: dict, plan: ScenePlan) -> str:
        """Use high-quality 3b1b code directly without conversion.

        manimgl will run it natively - no conversion needed!
        Just update the scene class name.
        """
        code = template.get("content", "")

        # Update scene class name to match the plan title
        import re
        words = re.sub(r"[^a-zA-Z0-9\s]", "", plan.title).split()
        class_name = "".join(word.capitalize() for word in words) or "GeneratedScene"

        # Replace the existing class name (handle various Scene types)
        code = re.sub(
            r"class\s+(\w+)\s*\(\s*(\w*Scene)\s*\)",
            f"class {class_name}(\\2)",
            code,
            count=1,
        )

        return code

    async def _get_rag_examples(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> tuple[str, dict | None]:
        """Retrieve relevant code examples AND library docs from RAG.

        Returns:
            Tuple of (rag_context_string, high_quality_match_or_none)
            If a high-quality 3b1b match is found (score > 0.02), returns it separately
            for direct adaptation rather than just inspiration.
        """
        if not self.rag_available:
            logger.debug("[RAG] RAG not available for code examples")
            return "", None

        # Search for similar scenes (3b1b prioritized)
        search_query = f"{analysis.domain.value}: {prompt}"
        logger.info("[RAG] Code generator searching: %s", search_query[:100])

        # ALSO search library documentation for correct API usage
        doc_results = await self.rag.search_documentation(
            query=f"manimgl {' '.join(analysis.visual_elements or [])} Mobject Scene animation",
            n_results=3,
        )
        if doc_results:
            logger.info("[RAG] Found %d library docs for API reference", len(doc_results))

        results = await self.rag.search_similar_scenes(
            query=search_query,
            n_results=3,
            prioritize_3b1b=True,
        )

        if not results:
            logger.info("[RAG] No code examples found")
            return "", None

        logger.info(
            "[RAG] Found %d code examples (scores: %s)",
            len(results),
            [f"{r.get('similarity_score', 0):.3f}" for r in results],
        )

        # Check for high-quality 3b1b match
        high_quality_match = None
        for result in results:
            score = result.get("similarity_score", 0)
            content = result.get("content", "")
            is_3b1b = "manim_imports_ext" in content or "OldTex" in content

            # High match threshold: similarity > 0.02 (distance < 0.98) for 3b1b code
            if is_3b1b and score > 0.02:
                logger.info(
                    "[RAG] HIGH QUALITY 3b1b match found! score=%.3f - will use as template",
                    score,
                )
                high_quality_match = result
                break

        examples = []
        for i, result in enumerate(results[:2], 1):
            code = result.get("content", "")
            meta = result.get("metadata", {})
            score = result.get("similarity_score", 0)
            is_3b1b = "manim_imports_ext" in code or "OldTex" in code

            logger.debug(
                "[RAG] Example %d: source=%s, is_3b1b=%s, len=%d, score=%.3f",
                i, meta.get("source", "?"), is_3b1b, len(code), score,
            )
            if len(code) > 2000:
                code = code[:2000] + "\n# ... (truncated)"

            source_label = " (3blue1brown original)" if is_3b1b else ""
            examples.append(f"Example {i}{source_label}:\n```python\n{code}\n```")

        # Add library documentation for correct API usage
        if doc_results:
            examples.append("\n" + "=" * 60)
            examples.append("MANIMGL LIBRARY REFERENCE (use these exact APIs):")
            examples.append("=" * 60)
            for doc in doc_results[:3]:
                doc_content = doc.get("content", "")
                doc_meta = doc.get("metadata", {})
                category = doc_meta.get("category", "")
                name = doc_meta.get("name", "")
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "\n# ... (truncated)"
                examples.append(f"\n# {category}/{name}:\n```python\n{doc_content}\n```")

        return "\n\n".join(examples), high_quality_match

    def _build_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
        rag_context: str,
        high_quality_template: dict | None = None,
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

        # If we have a high-quality 3b1b template, use it with API conversion
        if high_quality_template:
            template_code = high_quality_template.get("content", "")
            parts.extend([
                "",
                "=" * 60,
                "3BLUE1BROWN REFERENCE CODE - ADAPT THIS!",
                "=" * 60,
                "",
                "```python",
                template_code,
                "```",
                "",
                "MANDATORY API CONVERSIONS (apply ALL of these):",
                "- from manim_imports_ext import * → from manim import *",
                "- OldTex(...) → MathTex(r'...')",
                "- OldTexText(...) → Text(...)",
                "- .move_to(x, point_to_align=Y) → .move_to(x, aligned_edge=Y)",
                "- self.play(obj.method, val) → self.play(obj.animate.method(val))",
                "- RightAngle(triangle, ...) → Elbow() or just skip right angle marks",
                "- SIDE_COLORS → [RED, GREEN, BLUE]",
                "- compass_directions(4) → [UP, RIGHT, DOWN, LEFT]",
                "- get_corner(DL) works the same",
                "- Polygon, Square, Line, VGroup, MathTex, Text all work the same",
                "",
                "Keep the STRUCTURE, TIMING, and VISUAL QUALITY of the reference.",
                "The animation logic is excellent - just fix the API calls.",
            ])
        elif rag_context:
            parts.extend([
                "",
                "Reference examples (study their style and patterns):",
                rag_context,
            ])

        return "\n".join(parts)
