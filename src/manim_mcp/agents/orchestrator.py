"""AgentOrchestrator: Coordinates the multi-agent pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from manim_mcp.agents.code_generator import CodeGeneratorAgent
from manim_mcp.agents.code_reviewer import CodeReviewerAgent
from manim_mcp.agents.concept_analyzer import ConceptAnalyzerAgent
from manim_mcp.agents.scene_planner import ScenePlannerAgent
from manim_mcp.models import AgentPipelineResult

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig
    from manim_mcp.core.llm import GeminiClient
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates the multi-agent pipeline for advanced code generation.

    Pipeline:
    1. ConceptAnalyzer → Parse prompt, identify domain/complexity
    2. ScenePlanner → Design structure, query RAG for examples
    3. CodeGenerator → Generate code with RAG context
    4. CodeReviewer → Validate and fix before sandbox

    If any agent fails, the orchestrator falls back gracefully.
    """

    def __init__(
        self,
        llm: GeminiClient,
        rag: ChromaDBService | None,
        config: ManimMCPConfig,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.config = config

        # Initialize agents
        self.concept_analyzer = ConceptAnalyzerAgent(llm, rag, config)
        self.scene_planner = ScenePlannerAgent(llm, rag, config)
        self.code_generator = CodeGeneratorAgent(llm, rag, config)
        self.code_reviewer = CodeReviewerAgent(llm, rag, config)

    async def generate_advanced(
        self,
        prompt: str,
        max_review_iterations: int = 2,
    ) -> AgentPipelineResult:
        """Run the full multi-agent pipeline.

        Args:
            prompt: User's animation request
            max_review_iterations: Max times to iterate on code fixes

        Returns:
            AgentPipelineResult with all intermediate outputs
        """
        logger.info("Starting advanced generation pipeline for: %s", prompt[:100])

        result = AgentPipelineResult()

        # 1. Concept Analysis
        logger.debug("Step 1: Concept analysis")
        try:
            result.concept_analysis = await self.concept_analyzer.process(prompt)
            logger.debug(
                "Analyzed: domain=%s, complexity=%s, concepts=%s",
                result.concept_analysis.domain.value,
                result.concept_analysis.complexity.value,
                result.concept_analysis.key_concepts,
            )
        except Exception as e:
            logger.warning("Concept analysis failed: %s", e)
            # Continue without analysis

        # 2. Scene Planning
        logger.debug("Step 2: Scene planning")
        try:
            analysis = result.concept_analysis
            if analysis:
                result.scene_plan = await self.scene_planner.process(prompt, analysis)
                result.rag_context_used = bool(result.scene_plan.rag_examples)
                logger.debug(
                    "Planned: %s (%d segments, %.1fs)",
                    result.scene_plan.title,
                    len(result.scene_plan.segments),
                    result.scene_plan.total_duration,
                )
        except Exception as e:
            logger.warning("Scene planning failed: %s", e)
            # Continue without plan

        # 3. Code Generation
        logger.debug("Step 3: Code generation")
        try:
            if result.concept_analysis and result.scene_plan:
                result.generated_code = await self.code_generator.process(
                    prompt,
                    result.concept_analysis,
                    result.scene_plan,
                )
            else:
                # Fallback to simple generation
                logger.info("Falling back to simple generation (no plan)")
                result.generated_code = await self.llm.generate_code(prompt)
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            raise

        # 4. Code Review (with iteration)
        logger.debug("Step 4: Code review")
        code = result.generated_code
        for iteration in range(max_review_iterations):
            try:
                review = await self.code_reviewer.process(code, result.scene_plan)
                result.review_result = review

                if review.approved:
                    logger.debug("Code approved on iteration %d", iteration + 1)
                    break

                if review.fixed_code:
                    logger.debug(
                        "Review found %d issues, applying fix (iteration %d)",
                        len(review.issues),
                        iteration + 1,
                    )
                    code = review.fixed_code
                    result.generated_code = code
                else:
                    logger.warning(
                        "Review found issues but no fix provided: %s",
                        review.issues,
                    )
                    break

            except Exception as e:
                logger.warning("Code review failed: %s", e)
                break

        logger.info(
            "Pipeline complete: %d chars, reviewed=%s, rag_used=%s",
            len(result.generated_code),
            result.review_result.approved if result.review_result else "skipped",
            result.rag_context_used,
        )

        return result

    async def generate_simple(self, prompt: str) -> str:
        """Simple generation without the full pipeline.

        This is equivalent to the original single-LLM-call approach.

        Args:
            prompt: User's animation request

        Returns:
            Generated Manim code
        """
        return await self.llm.generate_code(prompt)
