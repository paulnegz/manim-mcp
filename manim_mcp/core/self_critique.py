"""Self-critique system for multi-pass code generation with self-review.

This module implements a multi-pass generation pipeline:
1. Generate initial code (pass 1)
2. Self-critique the code for issues (pass 2)
3. Fix identified issues (pass 3)
4. Optionally verify fixes (pass 4)

The self-critique pattern helps catch common issues before runtime validation,
improving code quality and reducing sandbox retry cycles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.core.llm import BaseLLMClient
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


# ── Critique Result ────────────────────────────────────────────────────


@dataclass
class CritiqueResult:
    """Result of a code critique pass.

    Attributes:
        issues: List of identified issues in the code
        severity: Overall severity level ("none", "minor", "major", "critical")
        suggestions: List of actionable suggestions to fix issues
        pass_number: Which critique pass produced this result
        code_quality_score: Optional quality score (0-100)
    """
    issues: list[str] = field(default_factory=list)
    severity: str = "none"  # "none", "minor", "major", "critical"
    suggestions: list[str] = field(default_factory=list)
    pass_number: int = 1
    code_quality_score: int | None = None

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.issues) > 0

    def needs_fix(self) -> bool:
        """Check if issues are severe enough to warrant fixing."""
        return self.severity in ("major", "critical")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "issues": self.issues,
            "severity": self.severity,
            "suggestions": self.suggestions,
            "pass_number": self.pass_number,
            "code_quality_score": self.code_quality_score,
        }


# ── Critique Prompts ───────────────────────────────────────────────────


CRITIQUE_SYSTEM_PROMPT = """You are a Manim code reviewer specializing in 3blue1brown style animations.
Your task is to critique the given Manim code for common issues and style violations.

Focus on these specific issues:

1. VARIABLE SHADOWING
   - Variables that are assigned to themselves (e.g., `BLUE_A = BLUE_A`)
   - Redefining built-in Manim constants without purpose
   - Local variables shadowing class attributes

2. UNUSED VARIABLES
   - Variables defined but never used in animations
   - Objects created but never added to scene or animated
   - Intermediate calculations whose results are discarded

3. MISSING NARRATION SYNC COMMENTS
   - For narrated animations, there should be comments indicating sync points
   - Format like: # [NARRATION: "text here"]
   - Missing wait() calls between narration segments

4. 3BLUE1BROWN STYLE COMPLIANCE
   - Uses smooth animations (FadeIn, Transform, Write) not instant Add
   - Proper use of wait() for pacing
   - Colors should be Manim constants (BLUE, YELLOW) not hex codes
   - Mathematical notation uses Tex/MathTex properly
   - Progressive reveals of complex concepts

5. DEAD CODE / REDUNDANT DEFINITIONS
   - Unreachable code after return statements
   - Duplicate imports or definitions
   - Commented-out code blocks
   - Variables overwritten before use

6. PROPER VGROUP USAGE
   - Multiple related objects should be grouped in VGroup
   - VGroups should be used for coordinated transformations
   - Avoid animating many individual objects when VGroup would be cleaner

Respond in JSON format:
{
    "issues": ["issue 1 description", "issue 2 description", ...],
    "severity": "none" | "minor" | "major" | "critical",
    "suggestions": ["suggestion 1", "suggestion 2", ...],
    "code_quality_score": 0-100
}

Severity guidelines:
- "none": No issues found, code is clean
- "minor": Style issues that don't affect functionality
- "major": Issues that may cause unexpected behavior or are clearly wrong
- "critical": Issues that will definitely cause errors or security concerns
"""


FIX_SYSTEM_PROMPT = """You are a Manim code fixer. Your task is to fix the identified issues in the code.

Apply fixes for all reported issues while:
1. Preserving the overall structure and intent of the animation
2. Maintaining 3blue1brown style patterns
3. Ensuring all animations are smooth and well-paced
4. Keeping narration sync points if present

Output ONLY the fixed Python code, no explanations or markdown fences.
"""


VERIFY_SYSTEM_PROMPT = """You are a Manim code verifier. Your task is to verify that the previous fixes were applied correctly.

Check that:
1. All reported issues have been addressed
2. No new issues were introduced by the fixes
3. The code still produces the intended animation
4. 3blue1brown style is maintained

Respond in JSON format:
{
    "all_fixed": true | false,
    "remaining_issues": ["issue 1", ...],
    "new_issues": ["issue 1", ...],
    "verification_passed": true | false
}
"""


# ── Self-Critique Generator ────────────────────────────────────────────


class SelfCritiqueGenerator:
    """Multi-pass code generation with self-critique loop.

    This generator implements a multi-pass approach to improve code quality:
    1. Generate initial code
    2. Critique the code for issues
    3. Fix identified issues
    4. Optionally verify fixes

    The critique loop continues until:
    - No major/critical issues are found
    - Maximum passes are reached
    - Fix attempts stop improving the code

    Attributes:
        llm: The LLM client for generation and critique
        rag: Optional RAG service for retrieving examples
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        rag: ChromaDBService | None = None,
    ) -> None:
        """Initialize the self-critique generator.

        Args:
            llm_client: LLM client for code generation and critique
            rag: Optional RAG service for retrieving similar examples
        """
        self.llm = llm_client
        self.rag = rag

    async def generate_with_critique(
        self,
        prompt: str,
        narration: list[str] | None = None,
        max_passes: int = 3,
        verify_fixes: bool = False,
    ) -> tuple[str, list[CritiqueResult]]:
        """Generate code with self-critique loop.

        This method orchestrates the multi-pass generation:
        1. Generate initial code from prompt
        2. Critique the generated code
        3. If issues found, fix them
        4. Repeat until clean or max_passes reached

        Args:
            prompt: The generation prompt describing the animation
            narration: Optional list of narration text for sync checking
            max_passes: Maximum number of critique/fix cycles (default 3)
            verify_fixes: Whether to run a verification pass after fixes

        Returns:
            Tuple of (final_code, list_of_critique_results)
        """
        critiques: list[CritiqueResult] = []

        # Pass 1: Generate initial code
        logger.info("[SELF-CRITIQUE] Pass 1: Generating initial code")
        code = await self._generate_initial_code(prompt, narration)

        current_pass = 1
        while current_pass < max_passes:
            current_pass += 1

            # Critique current code
            logger.info("[SELF-CRITIQUE] Pass %d: Critiquing code", current_pass)
            rag_examples = await self._get_rag_examples(prompt) if self.rag else []
            critique = await self.critique_code(code, rag_examples, narration)
            critique.pass_number = current_pass
            critiques.append(critique)

            logger.info(
                "[SELF-CRITIQUE] Critique result: severity=%s, issues=%d, score=%s",
                critique.severity,
                len(critique.issues),
                critique.code_quality_score,
            )

            # Check if we need to fix issues
            if not critique.needs_fix():
                logger.info("[SELF-CRITIQUE] No major issues found, stopping critique loop")
                break

            # Fix identified issues
            current_pass += 1
            if current_pass > max_passes:
                logger.warning(
                    "[SELF-CRITIQUE] Max passes reached (%d), stopping with issues",
                    max_passes,
                )
                break

            logger.info("[SELF-CRITIQUE] Pass %d: Fixing %d issues", current_pass, len(critique.issues))
            code = await self.fix_issues(code, critique)

        # Optional verification pass
        if verify_fixes and critiques and critiques[-1].needs_fix():
            logger.info("[SELF-CRITIQUE] Running verification pass")
            verification = await self._verify_fixes(code, critiques[-1])
            if not verification.get("verification_passed", False):
                logger.warning(
                    "[SELF-CRITIQUE] Verification failed: remaining=%s, new=%s",
                    verification.get("remaining_issues", []),
                    verification.get("new_issues", []),
                )

        return code, critiques

    async def critique_code(
        self,
        code: str,
        rag_examples: list[dict] | None = None,
        narration: list[str] | None = None,
    ) -> CritiqueResult:
        """Critique code for common issues.

        Performs static analysis and LLM-based review of the code
        to identify issues across multiple categories.

        Args:
            code: The Manim code to critique
            rag_examples: Optional list of similar examples from RAG
            narration: Optional narration text for sync checking

        Returns:
            CritiqueResult with identified issues and suggestions
        """
        # First, run static analysis
        static_issues = self._static_analysis(code, narration)

        # Build critique prompt
        critique_prompt = self._build_critique_prompt(code, rag_examples, narration, static_issues)

        # Get LLM critique
        try:
            result = await self._llm_call_json(critique_prompt, CRITIQUE_SYSTEM_PROMPT)

            issues = result.get("issues", [])
            severity = result.get("severity", "none")
            suggestions = result.get("suggestions", [])
            score = result.get("code_quality_score")

            # Merge static issues
            for issue in static_issues:
                if issue not in issues:
                    issues.insert(0, issue)

            # Upgrade severity if static analysis found critical issues
            if static_issues:
                if any("shadowing" in i.lower() for i in static_issues):
                    severity = max(severity, "major", key=lambda x: ["none", "minor", "major", "critical"].index(x))

            return CritiqueResult(
                issues=issues,
                severity=severity,
                suggestions=suggestions,
                code_quality_score=score,
            )

        except Exception as e:
            logger.warning("[SELF-CRITIQUE] LLM critique failed: %s", e)
            # Fall back to static analysis only
            severity = "major" if static_issues else "none"
            return CritiqueResult(
                issues=static_issues,
                severity=severity,
                suggestions=["Review and fix the identified issues manually"],
            )

    async def fix_issues(
        self,
        code: str,
        critique: CritiqueResult,
    ) -> str:
        """Fix identified issues in the code.

        Applies fixes for all issues in the critique result,
        prioritizing by severity.

        Args:
            code: The original code with issues
            critique: The critique result with issues to fix

        Returns:
            Fixed code
        """
        if not critique.has_issues():
            return code

        # Build fix prompt
        fix_prompt = self._build_fix_prompt(code, critique)

        try:
            fixed_code = await self._llm_call(fix_prompt, FIX_SYSTEM_PROMPT)
            fixed_code = self._strip_fences(fixed_code)

            # Validate the fix didn't break the code structure
            if not self._validate_code_structure(fixed_code):
                logger.warning("[SELF-CRITIQUE] Fixed code has structural issues, keeping original")
                return code

            logger.debug(
                "[SELF-CRITIQUE] Fixed code: %d -> %d chars",
                len(code),
                len(fixed_code),
            )
            return fixed_code

        except Exception as e:
            logger.warning("[SELF-CRITIQUE] Fix failed: %s", e)
            return code

    async def _generate_initial_code(
        self,
        prompt: str,
        narration: list[str] | None = None,
    ) -> str:
        """Generate initial code from prompt.

        Args:
            prompt: The animation description
            narration: Optional narration for sync hints

        Returns:
            Generated code
        """
        full_prompt = prompt
        if narration:
            narration_text = "\n".join(f"[{i+1}] {text}" for i, text in enumerate(narration))
            full_prompt += f"\n\nNARRATION (sync animations to these segments):\n{narration_text}"

        code = await self.llm.generate_code(full_prompt)
        return code

    async def _get_rag_examples(self, query: str) -> list[dict]:
        """Get relevant examples from RAG.

        Args:
            query: Search query

        Returns:
            List of example dictionaries
        """
        if not self.rag or not self.rag.available:
            return []

        try:
            results = await self.rag.search_similar_scenes(query, n_results=2)
            return results
        except Exception as e:
            logger.debug("[SELF-CRITIQUE] RAG search failed: %s", e)
            return []

    async def _verify_fixes(self, code: str, last_critique: CritiqueResult) -> dict:
        """Verify that fixes were applied correctly.

        Args:
            code: The fixed code
            last_critique: The critique that prompted fixes

        Returns:
            Verification result dict
        """
        verify_prompt = f"""Verify that the following issues have been fixed in this code:

ISSUES TO CHECK:
{chr(10).join(f"- {issue}" for issue in last_critique.issues)}

CODE:
```python
{code}
```

Check each issue and report on the verification status."""

        try:
            result = await self._llm_call_json(verify_prompt, VERIFY_SYSTEM_PROMPT)
            return result
        except Exception as e:
            logger.warning("[SELF-CRITIQUE] Verification failed: %s", e)
            return {"verification_passed": False, "error": str(e)}

    def _static_analysis(
        self,
        code: str,
        narration: list[str] | None = None,
    ) -> list[str]:
        """Perform static analysis on the code.

        Checks for issues that can be detected without LLM:
        - Variable shadowing
        - Unused variables
        - Missing narration comments
        - Common mistakes

        Args:
            code: Code to analyze
            narration: Optional narration for sync checking

        Returns:
            List of issues found
        """
        issues = []

        # 1. Check for variable shadowing (BLUE_A = BLUE_A pattern)
        shadowing_pattern = r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*\1\s*$'
        for match in re.finditer(shadowing_pattern, code, re.MULTILINE):
            var_name = match.group(1)
            issues.append(f"Variable shadowing: '{var_name} = {var_name}' is a no-op assignment")

        # 2. Check for common Manim constant shadowing
        manim_constants = [
            "BLUE", "RED", "GREEN", "YELLOW", "WHITE", "BLACK", "GREY", "GRAY",
            "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E",
            "RED_A", "RED_B", "RED_C", "RED_D", "RED_E",
            "GREEN_A", "GREEN_B", "GREEN_C", "GREEN_D", "GREEN_E",
            "YELLOW_A", "YELLOW_B", "YELLOW_C", "YELLOW_D", "YELLOW_E",
            "PINK", "TEAL", "PURPLE", "ORANGE", "MAROON",
            "UP", "DOWN", "LEFT", "RIGHT", "ORIGIN",
            "UL", "UR", "DL", "DR",
            "PI", "TAU", "DEGREES",
        ]

        for const in manim_constants:
            # Check for both self-assignment and suspicious reassignment
            self_assign = rf'^\s*{const}\s*=\s*{const}\s*$'
            if re.search(self_assign, code, re.MULTILINE):
                issues.append(f"Constant shadowing: '{const}' is assigned to itself")

        # 3. Check for unused variables (simple heuristic)
        # Find variable assignments
        var_assignments = re.findall(r'^\s*([a-z_][a-z0-9_]*)\s*=\s*[^=]', code, re.MULTILINE)
        for var in set(var_assignments):
            # Count occurrences (excluding the assignment line itself)
            occurrences = len(re.findall(rf'\b{re.escape(var)}\b', code))
            if occurrences == 1:
                # Only appears in assignment, likely unused
                # Skip common patterns that are intentional
                if var not in ('_', 'self', 'cls') and not var.startswith('_'):
                    issues.append(f"Potentially unused variable: '{var}' is assigned but never used")

        # 4. Check for missing narration sync comments
        if narration:
            narration_comments = re.findall(r'#\s*\[NARRATION:', code)
            if len(narration_comments) < len(narration):
                issues.append(
                    f"Missing narration sync: {len(narration)} narration segments but only "
                    f"{len(narration_comments)} sync comments found"
                )

        # 5. Check for dead code patterns
        # Unreachable code after return
        lines = code.split('\n')
        in_function = False
        after_return = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def '):
                in_function = True
                after_return = False
            elif in_function and stripped.startswith('return'):
                after_return = True
            elif after_return and stripped and not stripped.startswith(('def ', 'class ', '#', '@')):
                # Non-empty line after return in same function
                if not stripped.startswith(('except', 'finally', 'elif', 'else')):
                    issues.append(f"Possible dead code after return on line {i+1}")
                after_return = False

        # 6. Check for VGroup opportunities
        # If more than 3 individual objects are animated together, suggest VGroup
        play_calls = re.findall(r'self\.play\([^)]+\)', code)
        for call in play_calls:
            # Count comma-separated animations
            animations = call.count(',') + 1
            if animations > 3:
                issues.append(
                    f"Consider using VGroup: play() call with {animations} animations "
                    "might be cleaner with VGroup"
                )

        # 7. Check for Add instead of FadeIn/Write (anti-3b1b pattern)
        if 'self.add(' in code and 'FadeIn' not in code and 'Write' not in code:
            issues.append(
                "Using self.add() without animations - consider FadeIn() or Write() "
                "for 3blue1brown style reveals"
            )

        return issues

    def _build_critique_prompt(
        self,
        code: str,
        rag_examples: list[dict] | None,
        narration: list[str] | None,
        static_issues: list[str],
    ) -> str:
        """Build the critique prompt for LLM.

        Args:
            code: Code to critique
            rag_examples: RAG examples for comparison
            narration: Narration text if available
            static_issues: Issues found by static analysis

        Returns:
            Formatted critique prompt
        """
        parts = [
            "Critique this Manim code for issues:",
            "",
            "```python",
            code,
            "```",
        ]

        if narration:
            parts.extend([
                "",
                "NARRATION (check for proper sync):",
                *[f"[{i+1}] {text}" for i, text in enumerate(narration)],
            ])

        if static_issues:
            parts.extend([
                "",
                "STATIC ANALYSIS ALREADY FOUND THESE ISSUES:",
                *[f"- {issue}" for issue in static_issues],
                "",
                "Review for additional issues beyond these.",
            ])

        if rag_examples:
            parts.extend([
                "",
                "REFERENCE EXAMPLES (good 3b1b style):",
            ])
            for i, example in enumerate(rag_examples[:2]):
                content = example.get("content", "")[:500]
                parts.append(f"\nExample {i+1}:\n```python\n{content}\n```")

        return "\n".join(parts)

    def _build_fix_prompt(self, code: str, critique: CritiqueResult) -> str:
        """Build the fix prompt for LLM.

        Args:
            code: Code with issues
            critique: Critique result

        Returns:
            Formatted fix prompt
        """
        parts = [
            "Fix the following issues in this Manim code:",
            "",
            "ISSUES TO FIX:",
            *[f"- {issue}" for issue in critique.issues],
        ]

        if critique.suggestions:
            parts.extend([
                "",
                "SUGGESTIONS:",
                *[f"- {sug}" for sug in critique.suggestions],
            ])

        parts.extend([
            "",
            "ORIGINAL CODE:",
            "```python",
            code,
            "```",
            "",
            "Provide the fixed code. Output ONLY the corrected Python code.",
        ])

        return "\n".join(parts)

    def _validate_code_structure(self, code: str) -> bool:
        """Validate that code has required structure.

        Checks for:
        - Manim import
        - Scene class
        - construct method

        Args:
            code: Code to validate

        Returns:
            True if structure is valid
        """
        has_import = any(x in code for x in [
            "from manimlib import",
            "import manimlib",
            "from manim_imports_ext import",
            "from big_ol_pile_of_manim_imports import",
        ])

        has_scene = "class " in code and any(
            x in code for x in ["(Scene)", "(ThreeDScene)", "(MovingCameraScene)"]
        )

        has_construct = "def construct(self)" in code

        return has_import and has_scene and has_construct

    async def _llm_call(self, prompt: str, system: str) -> str:
        """Make an LLM call with the given prompt and system instruction.

        Uses the provider-agnostic LLM client.
        """
        from manim_mcp.core.llm import GeminiClient, ClaudeClient

        if isinstance(self.llm, GeminiClient):
            from google import genai
            response = await self.llm.client.aio.models.generate_content(
                model=self.llm.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system,
                ),
            )
            return response.text.strip()
        elif isinstance(self.llm, ClaudeClient):
            response = await self.llm.client.messages.create(
                model=self.llm.model_name,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            raise ValueError(f"Unknown LLM client type: {type(self.llm)}")

    async def _llm_call_json(self, prompt: str, system: str) -> dict:
        """Make an LLM call expecting JSON output.

        Uses the provider-agnostic LLM client.
        """
        import json
        from manim_mcp.core.llm import GeminiClient, ClaudeClient

        if isinstance(self.llm, GeminiClient):
            from google import genai
            response = await self.llm.client.aio.models.generate_content(
                model=self.llm.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                ),
            )
            text = response.text.strip()
        elif isinstance(self.llm, ClaudeClient):
            json_system = f"{system}\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown fences."
            response = await self.llm.client.messages.create(
                model=self.llm.model_name,
                max_tokens=8192,
                system=json_system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        else:
            raise ValueError(f"Unknown LLM client type: {type(self.llm)}")

        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        import json

        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("[SELF-CRITIQUE] Failed to parse JSON: %s", e)
            return {}

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences from text."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:python)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
