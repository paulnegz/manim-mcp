"""ConceptAnalyzerAgent: Analyzes prompts for math domain, complexity, and key concepts."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import Complexity, ConceptAnalysis, MathDomain

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert at analyzing mathematical and educational content. Given a prompt
for creating a Manim animation, analyze it and extract:

1. **domain**: The mathematical domain (algebra, calculus, geometry, linear_algebra,
   probability, physics, or general)
2. **complexity**: How complex the visualization should be (simple, moderate, complex)
3. **key_concepts**: List of 2-5 key concepts to visualize
4. **visual_elements**: Suggested Manim objects/elements to use (e.g., Circle, Arrow,
   Axes, NumberPlane, VGroup, etc.)
5. **suggested_duration**: Recommended animation duration in seconds (10-60)

Respond in JSON format with these exact keys."""


class ConceptAnalyzerAgent(BaseAgent):
    """Analyzes user prompts to extract domain, complexity, and key concepts.

    This is the first agent in the pipeline - it parses the user's intent
    and provides structured guidance for subsequent agents.
    """

    name = "concept_analyzer"

    async def process(self, prompt: str) -> ConceptAnalysis:
        """Analyze a user prompt and return structured concept analysis.

        Args:
            prompt: The user's animation request

        Returns:
            ConceptAnalysis with domain, complexity, concepts, etc.
        """
        logger.debug("Analyzing concepts for: %s", prompt[:100])

        try:
            result = await self._llm_call_json(
                prompt=f"Analyze this animation request:\n\n{prompt}",
                system=SYSTEM_PROMPT,
            )

            # Parse domain
            domain_str = result.get("domain", "general").lower()
            try:
                domain = MathDomain(domain_str)
            except ValueError:
                domain = MathDomain.general

            # Parse complexity
            complexity_str = result.get("complexity", "moderate").lower()
            try:
                complexity = Complexity(complexity_str)
            except ValueError:
                complexity = Complexity.moderate

            return ConceptAnalysis(
                domain=domain,
                complexity=complexity,
                key_concepts=result.get("key_concepts", [])[:5],
                visual_elements=result.get("visual_elements", [])[:10],
                suggested_duration=min(max(result.get("suggested_duration", 15), 5), 60),
            )

        except Exception as e:
            logger.warning("Concept analysis failed, using defaults: %s", e)
            # Store error for learning
            await self._store_error(prompt, str(e))
            return ConceptAnalysis(
                domain=MathDomain.general,
                complexity=Complexity.moderate,
                key_concepts=[],
                visual_elements=[],
                suggested_duration=15,
            )

    async def _store_error(self, prompt: str, error: str) -> None:
        """Store analysis error for self-learning."""
        if not self.rag_available:
            return
        try:
            await self.rag.store_error_pattern(
                error_message=f"[concept_analyzer] {error}"[:500],
                code="",
                fix=None,
                prompt=prompt[:200],
            )
        except Exception:
            pass  # Non-critical
