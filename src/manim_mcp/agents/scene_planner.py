"""ScenePlannerAgent: Plans animation structure, timing, and transitions."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan, SceneSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Grant Sanderson (3Blue1Brown), planning a mathematical animation that builds
intuition through elegant visual storytelling.

Create a detailed scene plan with:

1. **title**: A descriptive title for the scene
2. **segments**: List of animation segments following the 3b1b arc

MANDATORY 4-PHASE STRUCTURE (3Blue1Brown style):

Phase 1: ESTABLISH (2-3 seconds)
- Show what we're looking at
- Display title or introduce the main object
- Mobjects: Text/title, initial shapes
- Animations: Write, FadeIn, Create

Phase 2: BUILD (main content, 5-15 seconds)
- Progressive reveal - NEVER jump to the answer
- Show intermediate steps, relationships
- Use LaggedStart for staggered reveals
- Use TransformFromCopy to show relationships
- Split into 2-4 sub-segments

Phase 3: INSIGHT (2-4 seconds)
- Highlight the key moment/result
- Slow down for emphasis
- Mobjects: Key result, highlight boxes
- Animations: Indicate, FlashAround, Transform (slower run_time)

Phase 4: RESOLVE (2-3 seconds)
- Let the final state breathe
- Clean up or show final arrangement
- End with longer wait (2+ seconds)
- Animations: FadeOut (old elements), final positioning

Each segment needs:
- name: Short segment name
- description: What happens (be specific about the visual)
- duration: Duration in seconds (0.5-10)
- mobjects: Manim objects (Circle, MathTex, Axes, Arrow, VGroup, etc.)
- animations: 3b1b-style animations (LaggedStart, TransformFromCopy, FadeTransform, Indicate, etc.)

Keep total duration between 12-30 seconds. Use smooth transitions.

Respond in JSON format with keys: title, segments (array), total_duration."""


class ScenePlannerAgent(BaseAgent):
    """Plans the structure and timing of Manim animations.

    Uses RAG to find similar successful scenes and incorporates them
    into the planning process.
    """

    name = "scene_planner"

    async def process(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> ScenePlan:
        """Create a scene plan based on the prompt and concept analysis.

        Args:
            prompt: Original user prompt
            analysis: Output from ConceptAnalyzerAgent

        Returns:
            ScenePlan with segments, timing, and RAG examples
        """
        logger.debug("Planning scene for domain=%s, complexity=%s",
                     analysis.domain.value, analysis.complexity.value)

        # Query RAG for similar scenes
        rag_examples = []
        rag_context = ""
        if self.rag_available:
            logger.info("[RAG] Searching similar scenes for: %s", prompt[:100])
            similar_scenes = await self.rag.search_similar_scenes(
                query=prompt,
                n_results=3,
                prioritize_3b1b=True,
            )
            if similar_scenes:
                rag_examples = [s["document_id"] for s in similar_scenes]
                rag_context = self._build_rag_context(similar_scenes)
                logger.info(
                    "[RAG] Found %d similar scenes (scores: %s)",
                    len(similar_scenes),
                    [f"{s.get('similarity_score', 0):.3f}" for s in similar_scenes],
                )
                for i, s in enumerate(similar_scenes[:3], 1):
                    meta = s.get("metadata", {})
                    logger.debug(
                        "[RAG] Result %d: source=%s, file=%s, score=%.3f",
                        i, meta.get("source", "?"), meta.get("filename", "?"),
                        s.get("similarity_score", 0),
                    )
            else:
                logger.info("[RAG] No similar scenes found")

        # Build planning prompt
        planning_prompt = self._build_prompt(prompt, analysis, rag_context)

        try:
            result = await self._llm_call_json(
                prompt=planning_prompt,
                system=SYSTEM_PROMPT,
            )

            segments = []
            for seg_data in result.get("segments", []):
                segment = SceneSegment(
                    name=seg_data.get("name", "Segment"),
                    description=seg_data.get("description", ""),
                    duration=min(max(float(seg_data.get("duration", 2.0)), 0.5), 15.0),
                    mobjects=seg_data.get("mobjects", [])[:10],
                    animations=seg_data.get("animations", [])[:10],
                )
                segments.append(segment)

            # Ensure at least one segment
            if not segments:
                segments = [self._default_segment()]

            total_duration = sum(s.duration for s in segments)

            return ScenePlan(
                title=result.get("title", "Animation"),
                segments=segments,
                total_duration=total_duration,
                rag_examples=rag_examples,
            )

        except Exception as e:
            logger.warning("Scene planning failed, using default 3b1b arc: %s", e)
            # Store error for learning
            await self._store_error(prompt, str(e))
            default_segments = self._default_segments()
            return ScenePlan(
                title="Animation",
                segments=default_segments,
                total_duration=sum(s.duration for s in default_segments),
                rag_examples=rag_examples,
            )

    async def _store_error(self, prompt: str, error: str) -> None:
        """Store planning error for self-learning."""
        if not self.rag_available:
            return
        try:
            await self.rag.store_error_pattern(
                error_message=f"[scene_planner] {error}"[:500],
                code="",
                fix=None,
                prompt=prompt[:200],
            )
        except Exception:
            pass  # Non-critical

    def _build_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        rag_context: str,
    ) -> str:
        """Build the planning prompt with all context."""
        parts = [
            f"User request: {prompt}",
            "",
            "Concept analysis:",
            f"- Domain: {analysis.domain.value}",
            f"- Complexity: {analysis.complexity.value}",
            f"- Key concepts: {', '.join(analysis.key_concepts) or 'general'}",
            f"- Suggested elements: {', '.join(analysis.visual_elements) or 'basic shapes'}",
            f"- Target duration: ~{analysis.suggested_duration} seconds",
        ]

        if rag_context:
            parts.extend([
                "",
                "Similar successful animations for reference:",
                rag_context,
            ])

        return "\n".join(parts)

    def _build_rag_context(self, similar_scenes: list[dict]) -> str:
        """Format RAG results as context for the planner."""
        lines = []
        for i, scene in enumerate(similar_scenes[:3], 1):
            meta = scene.get("metadata", {})
            prompt_hint = meta.get("prompt", "")[:100]
            if prompt_hint:
                lines.append(f"{i}. Prompt: {prompt_hint}")
            # Show code snippet (first 500 chars)
            code = scene.get("content", "")[:500]
            if code:
                lines.append(f"   Code preview: {code[:200]}...")
        return "\n".join(lines)

    def _default_segments(self) -> list[SceneSegment]:
        """Create default segments following 3b1b arc when planning fails."""
        return [
            SceneSegment(
                name="Establish",
                description="Show title and introduce the main concept",
                duration=2.5,
                mobjects=["Text"],
                animations=["Write", "FadeIn"],
            ),
            SceneSegment(
                name="Build",
                description="Progressive reveal of the main content",
                duration=8.0,
                mobjects=["Circle", "Text", "VGroup"],
                animations=["Create", "LaggedStart", "Transform"],
            ),
            SceneSegment(
                name="Insight",
                description="Highlight the key result",
                duration=3.0,
                mobjects=["SurroundingRectangle"],
                animations=["Indicate", "FlashAround"],
            ),
            SceneSegment(
                name="Resolve",
                description="Let the final state breathe",
                duration=2.5,
                mobjects=[],
                animations=["FadeOut", "wait"],
            ),
        ]

    def _default_segment(self) -> SceneSegment:
        """Create a single default segment (legacy fallback)."""
        return SceneSegment(
            name="Main",
            description="Create and animate the main content",
            duration=10.0,
            mobjects=["Text", "Circle"],
            animations=["Create", "Write", "FadeOut"],
        )
