"""CodeGeneratorAgent: Generates Manim code from scene plans with RAG context."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan
from manim_mcp.prompts import get_code_generator_system

logger = logging.getLogger(__name__)

# Threshold for "very high quality" match - use code directly
# Lower threshold = more direct use of verified 3b1b code (fewer LLM generation errors)
DIRECT_USE_THRESHOLD = 0.08  # Keep low - 3b1b code is correct, LLM generation has errors

# Minimum code length for direct use (short snippets need LLM enhancement)
DIRECT_USE_MIN_CHARS = 800


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
        error_patterns = []
        animation_patterns = []
        api_signatures = []
        if self.rag_available and plan.rag_examples:
            rag_context, high_quality_template = await self._get_rag_examples(prompt, analysis)
            # Get error patterns to avoid common mistakes
            error_patterns = await self._get_error_patterns(prompt, analysis)
            # Get animation patterns for high-quality 3b1b-style animations
            animation_patterns = await self._get_animation_patterns(prompt, analysis)
            # Get API signatures for correct parameter usage
            api_signatures = await self._get_api_signatures(prompt, analysis)

        # Store template code for reference
        template_code = high_quality_template.get("content") if high_quality_template else None

        # For VERY HIGH quality matches, use 3b1b code DIRECTLY (no conversion!)
        # manimgl will run it natively
        # Requirements: high similarity score AND sufficient code length
        if high_quality_template:
            score = high_quality_template.get("similarity_score", 0)
            template_len = len(high_quality_template.get("content", ""))
            if score >= DIRECT_USE_THRESHOLD and template_len >= DIRECT_USE_MIN_CHARS:
                code = self._use_directly(high_quality_template, plan)
                logger.info(
                    "[DIRECT-USE] Using 3b1b code directly for score=%.3f (%d chars)",
                    score, len(code)
                )
                return code, template_code
            elif score >= DIRECT_USE_THRESHOLD:
                logger.info(
                    "[DIRECT-USE] Skipping - code too short (%d chars < %d min)",
                    template_len, DIRECT_USE_MIN_CHARS
                )

        # Build generation prompt
        gen_prompt = self._build_prompt(prompt, analysis, plan, rag_context, high_quality_template, error_patterns, animation_patterns, api_signatures)

        # Generate code
        system = get_code_generator_system(self.config.latex_available)
        code = await self._llm_call(gen_prompt, system)

        return self._strip_fences(code), template_code

    def _use_directly(self, template: dict, plan: ScenePlan) -> str:
        """Use high-quality 3b1b code directly with import fixes.

        Fixes imports to work with manimlib and updates class name.
        """
        import re

        code = template.get("content", "")

        # Fix imports - replace 3b1b custom imports with standard manimlib
        import_replacements = [
            ("from manim_imports_ext import *", "from manimlib import *"),
            ("from big_ol_pile_of_manim_imports import *", "from manimlib import *"),
            ("from manimlib.imports import *", "from manimlib import *"),
            ("from manim import *", "from manimlib import *"),  # CE -> manimgl
        ]
        for old, new in import_replacements:
            code = code.replace(old, new)

        # Ensure manimlib import exists if missing entirely
        if "from manimlib import" not in code and "import manimlib" not in code:
            code = "from manimlib import *\n\n" + code

        # Update scene class name to match the plan title
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
            # Don't truncate too aggressively - 3b1b examples can be 3000-4000 chars
            if len(code) > 4000:
                code = code[:4000] + "\n# ... (truncated)"

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

    async def _get_error_patterns(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant error patterns to avoid common mistakes.

        Returns:
            List of error patterns with their fixes
        """
        if not self.rag_available:
            return []

        # Build a query that captures the visual elements being used
        visual_elements = analysis.visual_elements or []
        query_parts = [prompt]

        # Add specific method names that often have parameter issues
        problem_methods = ["get_riemann_rectangles", "get_area", "Axes", "NumberPlane"]
        for method in problem_methods:
            if any(method.lower() in elem.lower() for elem in visual_elements) or method.lower() in prompt.lower():
                query_parts.append(method)

        search_query = " ".join(query_parts)
        logger.debug("[RAG] Searching error patterns for: %s", search_query[:100])

        try:
            results = await self.rag.search_error_patterns(search_query, n_results=5)

            if results:
                logger.info("[RAG] Found %d error patterns to avoid", len(results))

            return results

        except Exception as e:
            logger.warning("[RAG] Error pattern search failed: %s", e)
            return []

    async def _get_animation_patterns(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant 3b1b animation patterns for high-quality animations.

        IMPORTANT: Always returns at least 2 patterns for multi-animation videos.

        Returns:
            List of animation patterns with code templates (minimum 2)
        """
        if not self.rag_available:
            return []

        # Build a query that captures the math concepts and animation needs
        visual_elements = analysis.visual_elements or []
        key_concepts = analysis.key_concepts or []

        query_parts = [prompt]
        query_parts.extend(visual_elements)
        query_parts.extend(key_concepts)

        # Add domain-specific keywords
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["riemann", "rectangles", "integral", "area under"]):
            query_parts.append("riemann_sum_convergence progression")
        if any(kw in prompt_lower for kw in ["derivative", "tangent", "slope"]):
            query_parts.append("tangent_line derivative calculus")
        if any(kw in prompt_lower for kw in ["series", "sum", "convergence"]):
            query_parts.append("series partial_sums convergence")
        if any(kw in prompt_lower for kw in ["matrix", "transformation", "linear"]):
            query_parts.append("matrix_transformation linear_algebra")

        search_query = " ".join(query_parts)
        logger.debug("[RAG] Searching animation patterns for: %s", search_query[:100])

        results = []
        try:
            # Get primary patterns based on prompt (request 4 to ensure variety)
            results = await self.rag.search_animation_patterns(search_query, n_results=4)

            # ENSURE at least 2 patterns - add complementary technique patterns
            if len(results) < 2:
                # Add general technique patterns that work with any animation
                complementary_queries = [
                    "lagged_start_reveal staggered sequence",
                    "value_tracker_animation continuous smooth",
                    "indicate_with_flash highlight emphasis",
                    "build_up_construction step by step",
                    "fade_transition smooth animation",
                ]
                for fallback_query in complementary_queries:
                    if len(results) >= 2:
                        break
                    fallback_results = await self.rag.search_animation_patterns(
                        fallback_query, n_results=1
                    )
                    for r in fallback_results:
                        # Avoid duplicates
                        if not any(r.get("id") == existing.get("id") for existing in results):
                            results.append(r)
                            if len(results) >= 2:
                                break

            if results:
                logger.info("[RAG] Found %d animation patterns to apply (min 2 required)", len(results))

        except Exception as e:
            logger.warning("[RAG] Animation pattern search failed: %s", e)

        return results[:4]  # Return up to 4 patterns

    async def _get_api_signatures(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant API signatures for correct parameter usage.

        Returns:
            List of API signatures with parameters and docstrings
        """
        if not self.rag_available:
            return []

        # Build query from visual elements and prompt
        visual_elements = analysis.visual_elements or []
        query_parts = []

        # Add common manimgl classes/methods from visual elements
        for elem in visual_elements:
            query_parts.append(elem)

        # Add keywords from prompt
        prompt_lower = prompt.lower()
        method_keywords = [
            "axes", "graph", "riemann", "rectangles", "area", "tangent",
            "transform", "animation", "mobject", "vgroup", "tex", "mathtex",
            "circle", "square", "line", "arrow", "dot", "polygon",
            "numberplane", "complexplane", "valuetarcker", "always_redraw"
        ]
        for kw in method_keywords:
            if kw in prompt_lower:
                query_parts.append(kw)

        if not query_parts:
            query_parts = ["Axes", "Scene", "animation"]

        search_query = " ".join(query_parts[:5])  # Limit query length
        logger.debug("[RAG] Searching API signatures for: %s", search_query[:100])

        try:
            results = await self.rag.search_api_signatures(search_query, n_results=5)

            if results:
                logger.info("[RAG] Found %d API signatures for correct parameter usage", len(results))

            return results

        except Exception as e:
            logger.warning("[RAG] API signature search failed: %s", e)
            return []

    def _build_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
        rag_context: str,
        high_quality_template: dict | None = None,
        error_patterns: list[dict] | None = None,
        animation_patterns: list[dict] | None = None,
        api_signatures: list[dict] | None = None,
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

        # Add error patterns to avoid
        if error_patterns:
            parts.extend([
                "",
                "=" * 60,
                "MANIMGL PITFALLS TO AVOID (these will cause errors!):",
                "=" * 60,
            ])
            for pattern in error_patterns[:5]:  # Limit to 5 patterns
                content = pattern.get("content", "")
                # Parse the error/fix from the document format
                if "ERROR:" in content and "FIX:" in content:
                    # Extract just the key parts
                    error_part = content.split("FIX:")[0].replace("ERROR:", "").strip()
                    fix_part = content.split("FIX:")[-1].strip()
                    # Clean up for concise display
                    if len(error_part) > 100:
                        error_part = error_part[:100] + "..."
                    if len(fix_part) > 200:
                        fix_part = fix_part[:200] + "..."
                    parts.append(f"- DON'T: {error_part}")
                    parts.append(f"  DO: {fix_part}")
                else:
                    # Fallback for other formats
                    if len(content) > 300:
                        content = content[:300] + "..."
                    parts.append(f"- {content}")

        # Add animation patterns (3b1b-style techniques)
        # IMPORTANT: Use at least 2 patterns for professional multi-animation videos
        if animation_patterns:
            n_patterns = min(len(animation_patterns), 4)
            parts.extend([
                "",
                "=" * 60,
                f"3B1B ANIMATION PATTERNS TO USE (MANDATORY: use at least 2!):",
                "=" * 60,
                "",
                "CRITICAL: Every video should use MULTIPLE animation techniques!",
                f"You have {n_patterns} patterns below - combine at least 2 in your code.",
                "",
                "Examples of multi-pattern usage:",
                "- Use 'lagged_start_reveal' for initial elements + 'value_tracker' for dynamics",
                "- Use 'build_up_construction' for setup + 'transform' for changes",
                "- Use 'zoom_to_detail' for focus + 'indicate_with_flash' for emphasis",
                "",
            ])
            for i, pattern in enumerate(animation_patterns[:4], 1):  # Up to 4 patterns
                content = pattern.get("content", "")
                meta = pattern.get("metadata", {})
                pattern_name = meta.get("name", "pattern")
                category = meta.get("category", "")

                parts.append(f"\n### PATTERN {i}: {pattern_name} ({category})")

                # Extract code template from content
                if "## Code Template" in content:
                    # Get everything after Code Template header
                    template_start = content.find("## Code Template")
                    if template_start != -1:
                        template_section = content[template_start:]
                        # Truncate if too long
                        if len(template_section) > 1200:
                            template_section = template_section[:1200] + "\n# ... (truncated)"
                        parts.append(template_section)
                else:
                    # Just use the content directly
                    if len(content) > 1200:
                        content = content[:1200] + "\n# ... (truncated)"
                    parts.append(content)

            # Reminder at the end
            parts.extend([
                "",
                "=" * 60,
                "REMEMBER: Your animation MUST use at least 2 of the above patterns!",
                "=" * 60,
            ])

        # Add API signatures for correct parameter usage
        if api_signatures:
            parts.extend([
                "",
                "=" * 60,
                "MANIMGL API SIGNATURES (use these EXACT parameters!):",
                "=" * 60,
                "",
            ])
            for sig in api_signatures[:5]:  # Limit to 5 signatures
                content = sig.get("content", "")
                meta = sig.get("metadata", {})
                method_name = meta.get("name", "")
                class_name = meta.get("class_name", "")

                if class_name and method_name:
                    parts.append(f"### {class_name}.{method_name}")
                elif method_name:
                    parts.append(f"### {method_name}")

                # Extract signature and key params
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"```python\n{content}\n```")
                parts.append("")

        return "\n".join(parts)
