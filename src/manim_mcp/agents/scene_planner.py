"""ScenePlannerAgent: Plans animation structure, timing, and transitions."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan, SceneSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Manim animation planner. Given a concept analysis and user prompt,
create a detailed scene plan with:

1. **title**: A descriptive title for the scene
2. **segments**: List of animation segments, each with:
   - name: Short segment name
   - description: What happens in this segment
   - duration: Duration in seconds (0.5-10)
   - mobjects: List of Manim objects used (Circle, Text, Axes, etc.)
   - animations: List of animations used (Create, Write, FadeIn, Transform, etc.)

Structure the animation as:
- Introduction (show title, set context)
- Main content (build up the core concept step by step)
- Conclusion (summarize or show final result)

Keep total duration between 10-45 seconds. Use smooth transitions between segments.

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
            similar_scenes = await self.rag.search_similar_scenes(
                query=prompt,
                n_results=3,
            )
            if similar_scenes:
                rag_examples = [s["document_id"] for s in similar_scenes]
                rag_context = self._build_rag_context(similar_scenes)

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
            logger.warning("Scene planning failed, using default: %s", e)
            return ScenePlan(
                title="Animation",
                segments=[self._default_segment()],
                total_duration=10.0,
                rag_examples=rag_examples,
            )

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

    def _default_segment(self) -> SceneSegment:
        """Create a default segment when planning fails."""
        return SceneSegment(
            name="Main",
            description="Create and animate the main content",
            duration=10.0,
            mobjects=["Text", "Circle"],
            animations=["Create", "Write", "FadeOut"],
        )
