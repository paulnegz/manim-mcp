"""CodeReviewerAgent: Reviews and validates generated Manim code."""

from __future__ import annotations

import logging

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import CodeReviewResult, ScenePlan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Manim code reviewer. Review the provided code for:

1. **Correctness**: Valid Python syntax, proper Manim API usage
2. **Scene structure**: Has Scene subclass with construct() method
3. **Animation quality**: Proper timing, smooth transitions, good pacing
4. **Best practices**: Proper imports, VGroup usage, clear positioning

If issues are found:
- List specific issues with line numbers if possible
- Provide the FIXED code if the issues are fixable

Respond in JSON format:
{
  "approved": true/false,
  "issues": ["list of issues found"],
  "suggestions": ["optional improvement suggestions"],
  "fixed_code": "corrected code if issues found, null if approved"
}

Only return fixed_code if there are actual issues that need fixing."""


class CodeReviewerAgent(BaseAgent):
    """Reviews generated code for quality before passing to the sandbox.

    Can use RAG to check for known error patterns and apply fixes.
    """

    name = "code_reviewer"

    async def process(
        self,
        code: str,
        plan: ScenePlan | None = None,
    ) -> CodeReviewResult:
        """Review generated Manim code and optionally fix issues.

        Args:
            code: Generated Manim code to review
            plan: Optional scene plan for context

        Returns:
            CodeReviewResult with approval status and any fixes
        """
        logger.debug("Reviewing code (%d chars)", len(code))

        # First, do static checks
        static_issues = self._static_checks(code)

        # Check RAG for similar error patterns
        rag_suggestions = []
        if self.rag_available and static_issues:
            rag_suggestions = await self._check_error_patterns(static_issues)

        # Build review prompt
        review_prompt = self._build_prompt(code, plan, static_issues, rag_suggestions)

        try:
            result = await self._llm_call_json(
                prompt=review_prompt,
                system=SYSTEM_PROMPT,
            )

            approved = result.get("approved", False)
            issues = result.get("issues", [])
            suggestions = result.get("suggestions", [])
            fixed_code = result.get("fixed_code")

            # Add static issues if LLM missed them
            for issue in static_issues:
                if issue not in issues:
                    issues.insert(0, issue)

            if issues:
                approved = False

            return CodeReviewResult(
                approved=approved,
                issues=issues,
                suggestions=suggestions,
                fixed_code=self._strip_fences(fixed_code) if fixed_code else None,
            )

        except Exception as e:
            logger.warning("Code review failed: %s", e)
            # If review fails but static checks pass, cautiously approve
            if not static_issues:
                return CodeReviewResult(approved=True)
            return CodeReviewResult(
                approved=False,
                issues=static_issues,
            )

    def _static_checks(self, code: str) -> list[str]:
        """Perform static checks on the code."""
        issues = []

        # Check for required imports
        if "from manim import" not in code and "import manim" not in code:
            issues.append("Missing Manim import (need 'from manim import *')")

        # Check for Scene class
        if "class " not in code or "(Scene)" not in code:
            if "(ThreeDScene)" not in code and "(MovingCameraScene)" not in code:
                issues.append("No Scene subclass found")

        # Check for construct method
        if "def construct(self)" not in code:
            issues.append("Missing construct(self) method")

        # Check for dangerous imports (security)
        dangerous = ["os.system", "subprocess", "eval(", "exec(", "__import__"]
        for danger in dangerous:
            if danger in code:
                issues.append(f"Potentially dangerous code: {danger}")

        # Check for common mistakes
        if "self.play()" in code and code.count("self.play()") > 0:
            # Empty play calls
            if "self.play()" in code:
                issues.append("Empty self.play() call (needs animation argument)")

        return issues

    async def _check_error_patterns(self, issues: list[str]) -> list[str]:
        """Check RAG for known fixes for these issues."""
        if not self.rag_available:
            return []

        suggestions = []
        for issue in issues[:3]:  # Check first 3 issues
            results = await self.rag.search_error_patterns(issue, n_results=1)
            if results:
                content = results[0].get("content", "")
                if "FIX:" in content:
                    fix = content.split("FIX:")[1].strip()[:200]
                    suggestions.append(f"Known fix for '{issue[:50]}': {fix}")

        return suggestions

    def _build_prompt(
        self,
        code: str,
        plan: ScenePlan | None,
        static_issues: list[str],
        rag_suggestions: list[str],
    ) -> str:
        """Build the review prompt."""
        parts = [
            "Review this Manim code:",
            "",
            "```python",
            code,
            "```",
        ]

        if plan:
            parts.extend([
                "",
                f"Expected scene: {plan.title}",
                f"Expected duration: ~{plan.total_duration:.1f}s",
                f"Segments: {len(plan.segments)}",
            ])

        if static_issues:
            parts.extend([
                "",
                "Static analysis found these issues:",
                *[f"- {issue}" for issue in static_issues],
            ])

        if rag_suggestions:
            parts.extend([
                "",
                "Known fixes from similar errors:",
                *[f"- {sug}" for sug in rag_suggestions],
            ])

        return "\n".join(parts)
