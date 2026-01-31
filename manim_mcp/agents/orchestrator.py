"""AgentOrchestrator: Coordinates the multi-agent pipeline."""

from __future__ import annotations

import asyncio
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

# Per-agent timeout configuration (seconds)
AGENT_TIMEOUTS = {
    "concept_analyzer": 40,   # Quick analysis
    "scene_planner": 55,      # May need RAG lookups
    "code_generator": 100,    # LLM generation + RAG
    "code_reviewer": 55,      # Static + LLM review
}


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
        narration_script: list[str] | None = None,
        max_review_iterations: int = 2,
    ) -> AgentPipelineResult:
        """Run the full multi-agent pipeline.

        Args:
            prompt: User's animation request
            narration_script: Pre-generated narration script (for code-audio sync)
            max_review_iterations: Max times to iterate on code fixes

        Returns:
            AgentPipelineResult with all intermediate outputs
        """
        logger.info("Starting advanced generation pipeline for: %s", prompt[:100])
        if narration_script:
            logger.info("Code generation will follow %d-sentence narration script", len(narration_script))

        result = AgentPipelineResult()

        # 1. Concept Analysis (with timeout)
        logger.debug("Step 1: Concept analysis")
        try:
            async with asyncio.timeout(AGENT_TIMEOUTS["concept_analyzer"]):
                result.concept_analysis = await self.concept_analyzer.process(prompt)
            logger.debug(
                "Analyzed: domain=%s, complexity=%s, concepts=%s",
                result.concept_analysis.domain.value,
                result.concept_analysis.complexity.value,
                result.concept_analysis.key_concepts,
            )
        except asyncio.TimeoutError:
            logger.warning("Concept analysis timed out after %ds", AGENT_TIMEOUTS["concept_analyzer"])
            await self._store_agent_error("concept_analysis", "timeout", prompt=prompt)
        except Exception as e:
            logger.warning("Concept analysis failed: %s", e)
            await self._store_agent_error("concept_analysis", str(e), prompt=prompt)

        # 2. Scene Planning (with timeout)
        logger.debug("Step 2: Scene planning")
        try:
            analysis = result.concept_analysis
            if analysis:
                async with asyncio.timeout(AGENT_TIMEOUTS["scene_planner"]):
                    result.scene_plan = await self.scene_planner.process(prompt, analysis)
                result.rag_context_used = bool(result.scene_plan.rag_examples)
                logger.debug(
                    "Planned: %s (%d segments, %.1fs)",
                    result.scene_plan.title,
                    len(result.scene_plan.segments),
                    result.scene_plan.total_duration,
                )
        except asyncio.TimeoutError:
            logger.warning("Scene planning timed out after %ds", AGENT_TIMEOUTS["scene_planner"])
            await self._store_agent_error("scene_planning", "timeout", prompt=prompt)
        except Exception as e:
            logger.warning("Scene planning failed: %s", e)
            await self._store_agent_error("scene_planning", str(e), prompt=prompt)

        # 3. Code Generation (with timeout)
        logger.debug("Step 3: Code generation")
        try:
            if result.concept_analysis and result.scene_plan:
                async with asyncio.timeout(AGENT_TIMEOUTS["code_generator"]):
                    generated_code, template_code = await self.code_generator.process(
                        prompt,
                        result.concept_analysis,
                        result.scene_plan,
                        narration_script,
                    )
                result.generated_code = generated_code
                result.original_template_code = template_code
            else:
                # Fallback to simple generation with RAG context
                logger.info("Falling back to simple generation with RAG (no plan)")
                async with asyncio.timeout(AGENT_TIMEOUTS["code_generator"]):
                    result.generated_code = await self.generate_simple(prompt, narration_script)
                result.rag_context_used = self.rag is not None and self.rag.available
        except asyncio.TimeoutError:
            logger.error("Code generation timed out after %ds", AGENT_TIMEOUTS["code_generator"])
            await self._store_agent_error("code_generation", "timeout", prompt=prompt)
            raise
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            await self._store_agent_error("code_generation", str(e), prompt=prompt)
            raise

        # 4. Code Review (with timeout and iteration)
        logger.debug("Step 4: Code review")
        code = result.generated_code
        for iteration in range(max_review_iterations):
            try:
                async with asyncio.timeout(AGENT_TIMEOUTS["code_reviewer"]):
                    review = await self.code_reviewer.process(
                        code,
                        result.scene_plan,
                        original_template=result.original_template_code,
                    )
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

            except asyncio.TimeoutError:
                logger.warning("Code review timed out after %ds (iteration %d)",
                              AGENT_TIMEOUTS["code_reviewer"], iteration + 1)
                await self._store_agent_error("code_review", "timeout", prompt=prompt, code=code)
                break
            except Exception as e:
                logger.warning("Code review failed: %s", e)
                await self._store_agent_error("code_review", str(e), prompt=prompt, code=code)
                break

        logger.info(
            "Pipeline complete: %d chars, reviewed=%s, rag_used=%s",
            len(result.generated_code),
            result.review_result.approved if result.review_result else "skipped",
            result.rag_context_used,
        )

        return result

    async def generate_simple(
        self,
        prompt: str,
        narration_script: list[str] | None = None,
    ) -> str:
        """Simple generation with FULL RAG context (all sources in parallel).

        Queries all RAG collections in parallel for comprehensive context:
        - Similar scenes (3b1b examples)
        - API signatures (method constraints)
        - Animation patterns (reusable templates)
        - Documentation (library usage)
        - Error patterns (common mistakes to avoid)

        Args:
            prompt: User's animation request
            narration_script: Pre-generated narration script (for code-audio sync)

        Returns:
            Generated Manim code
        """
        if narration_script:
            logger.info("Simple generation will follow %d-sentence narration script", len(narration_script))

        # Query RAG sources with PRIORITY ordering (most important first)
        # Strategy: API signatures + scenes in parallel (critical), then patterns if needed
        scenes_context = ""
        api_context = ""
        patterns_context = ""
        docs_context = ""
        errors_context = ""

        if self.rag and self.rag.available:
            logger.info("[RAG] Querying priority sources for simple generation")
            try:
                # TIER 1: Critical queries in parallel (API signatures + similar scenes)
                # These are essential for correctness and style
                tier1_results = await asyncio.gather(
                    self.rag.search_api_signatures(query=prompt, n_results=5),
                    self.rag.search_similar_scenes(query=prompt, n_results=3, prioritize_3b1b=True),
                    return_exceptions=True,
                )

                api_sigs, similar_scenes = tier1_results

                if isinstance(api_sigs, list) and api_sigs:
                    logger.info("[RAG] Found %d API signatures", len(api_sigs))
                    api_context = self._build_api_context(api_sigs)

                if isinstance(similar_scenes, list) and similar_scenes:
                    logger.info("[RAG] Found %d similar scenes", len(similar_scenes))
                    scenes_context = self._build_rag_context(similar_scenes)

                # TIER 2: Animation patterns (only if scenes didn't provide enough context)
                # Skip if we already have good scene examples
                if not scenes_context or len(scenes_context) < 500:
                    try:
                        patterns = await self.rag.search_animation_patterns(query=prompt, n_results=3)
                        if patterns:
                            logger.info("[RAG] Found %d animation patterns", len(patterns))
                            patterns_context = self._build_patterns_context(patterns)
                    except Exception as e:
                        logger.debug("[RAG] Patterns query failed: %s", e)

                # TIER 3: Error patterns (lightweight, useful for avoiding common mistakes)
                # Only query if we have API context (errors are about API misuse)
                if api_context:
                    try:
                        errors = await self.rag.search_error_patterns(query=prompt, n_results=2)
                        if errors:
                            logger.info("[RAG] Found %d error patterns to avoid", len(errors))
                            errors_context = self._build_errors_context(errors)
                    except Exception as e:
                        logger.debug("[RAG] Errors query failed: %s", e)

                # Skip docs for simple generation - scenes provide better examples

            except Exception as e:
                logger.warning("[RAG] Failed to query RAG sources: %s", e)

        # Build enhanced prompt with all RAG context
        enhanced_prompt = prompt

        if api_context:
            enhanced_prompt = f"{enhanced_prompt}\n\n## API CONSTRAINTS - USE THESE EXACT SIGNATURES:\n{api_context}"

        if scenes_context:
            enhanced_prompt = f"{enhanced_prompt}\n\n## REFERENCE EXAMPLES (from 3Blue1Brown):\n{scenes_context}"

        if patterns_context:
            enhanced_prompt = f"{enhanced_prompt}\n\n## ANIMATION PATTERNS TO USE:\n{patterns_context}"

        if docs_context:
            enhanced_prompt = f"{enhanced_prompt}\n\n## DOCUMENTATION REFERENCE:\n{docs_context}"

        if errors_context:
            enhanced_prompt = f"{enhanced_prompt}\n\n## COMMON MISTAKES TO AVOID:\n{errors_context}"

        # Add narration script guidance if provided (code-audio sync)
        if narration_script:
            script_text = "\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(narration_script))
            enhanced_prompt = f"""{enhanced_prompt}

IMPORTANT - NARRATION SCRIPT TO FOLLOW:
The animation MUST match this narration script exactly. Each numbered sentence corresponds to a visual step.

{script_text}

Generate code where each self.play() or self.wait() corresponds to one narration sentence in order.
The timing and visuals must sync with this script."""

        return await self.llm.generate_code(enhanced_prompt)

    def _build_rag_context(self, similar_scenes: list[dict]) -> str:
        """Format RAG results as context for generation.

        Shows full code for high-similarity matches to enable close following of patterns.
        """
        lines = []
        for i, scene in enumerate(similar_scenes[:3], 1):
            meta = scene.get("metadata", {})
            similarity = scene.get("similarity_score", 0)
            prompt_hint = meta.get("prompt", "")[:100]
            if prompt_hint:
                lines.append(f"\nExample {i} (prompt: {prompt_hint}, similarity: {similarity:.2f}):")
            else:
                lines.append(f"\nExample {i} (similarity: {similarity:.2f}):")

            # Show more code for high-similarity matches (up to 3000 chars)
            # This allows LLM to follow working patterns closely instead of inventing
            max_chars = 3000 if similarity > 0.7 else 1500
            code = scene.get("content", "")[:max_chars]
            if code:
                lines.append(f"```python\n{code}\n```")
                if similarity > 0.8:
                    lines.append("**IMPORTANT: This is a highly relevant example. Follow this pattern closely.**")
        return "\n".join(lines)

    def _build_api_context(self, api_sigs: list[dict]) -> str:
        """Format API signatures as constraints for generation."""
        lines = []
        for sig in api_sigs[:5]:
            meta = sig.get("metadata", {})
            method_name = meta.get("method_name", meta.get("full_name", ""))
            valid_params = meta.get("valid_params", meta.get("parameter_names", ""))
            if method_name:
                if valid_params:
                    lines.append(f"- {method_name}({valid_params})")
                else:
                    content = sig.get("content", "")[:200]
                    if content:
                        lines.append(f"- {content}")
        return "\n".join(lines) if lines else ""

    def _build_patterns_context(self, patterns: list[dict]) -> str:
        """Format animation patterns as templates."""
        lines = []
        for i, pattern in enumerate(patterns[:3], 1):
            meta = pattern.get("metadata", {})
            name = meta.get("name", meta.get("pattern_type", f"Pattern {i}"))
            lines.append(f"\n### {name}")
            code = pattern.get("content", "")[:600]
            if code:
                lines.append(f"```python\n{code}\n```")
        return "\n".join(lines) if lines else ""

    def _build_docs_context(self, docs: list[dict]) -> str:
        """Format documentation entries."""
        lines = []
        for doc in docs[:2]:
            content = doc.get("content", "")[:400]
            if content:
                lines.append(content)
        return "\n\n".join(lines) if lines else ""

    def _build_errors_context(self, errors: list[dict]) -> str:
        """Format error patterns as warnings."""
        lines = []
        for error in errors[:3]:
            meta = error.get("metadata", {})
            error_msg = meta.get("error_message", "")[:100]
            fix = meta.get("fix", "")[:150]
            if error_msg:
                if fix:
                    lines.append(f"- AVOID: {error_msg}\n  FIX: {fix}")
                else:
                    lines.append(f"- AVOID: {error_msg}")
        return "\n".join(lines) if lines else ""

    async def _store_agent_error(
        self,
        agent_name: str,
        error: str,
        prompt: str | None = None,
        code: str | None = None,
    ) -> None:
        """Store an agent error for self-learning.

        All errors in the pipeline are stored to the RAG error_patterns
        collection so the system can learn from failures.
        """
        if not self.rag or not self.rag.available:
            return

        try:
            error_message = f"[{agent_name}] {error}"
            await self.rag.store_error_pattern(
                error_message=error_message[:500],
                code=code[:3000] if code else "",
                fix=None,  # No fix available at this stage
                prompt=prompt[:200] if prompt else None,
            )
            logger.debug("[RAG] Stored %s error pattern", agent_name)
        except Exception as e:
            # Non-critical, just log
            logger.debug("Failed to store agent error: %s", e)
