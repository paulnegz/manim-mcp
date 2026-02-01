"""Template-First Generation (Fill-in-the-Middle style like Copilot).

This module implements a structured code generation approach where:
1. A code skeleton is built from a narration script
2. Each narration sentence becomes a commented section
3. Sections are filled independently using FIM-style generation
4. Structure compliance is enforced - comments are automatic, sections are isolated

This approach prevents hallucination of malformed structures and ensures
each visual step maps 1:1 with the narration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from manim_mcp.prompts import get_template_color_section, get_template_step_section

if TYPE_CHECKING:
    from manim_mcp.core.llm import BaseLLMClient
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


# ── Protocols ────────────────────────────────────────────────────────────────


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients to enable dependency injection."""

    async def generate_code(self, prompt: str) -> str:
        """Generate code from a prompt."""
        ...


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class Section:
    """A single section in the template to be filled.

    Attributes:
        index: Section index (0 for colors, 1+ for steps)
        narration: The narration text for this section (empty for color section)
        prefix: Code that comes before this section (fixed, not generated)
        suffix: Code that comes after this section (fixed, not generated)
        content: Generated code for this section (filled by LLM)
        is_color_section: Whether this is the color definitions section
    """

    index: int
    narration: str
    prefix: str
    suffix: str
    content: str = ""
    is_color_section: bool = False

    @property
    def placeholder(self) -> str:
        """Get the placeholder string for this section."""
        return f"{{FILL_SECTION_{self.index}}}"

    @property
    def header_comment(self) -> str:
        """Get the section header comment."""
        if self.is_color_section:
            return "# === COLOR DEFINITIONS ==="
        return f"# === STEP {self.index}: {self.narration} ==="

    def get_fim_prompt(self, rag_examples: list[dict] | None = None) -> str:
        """Generate the FIM-style prompt for filling this section.

        Args:
            rag_examples: Optional RAG examples for few-shot context

        Returns:
            Prompt string for the LLM
        """
        if self.is_color_section:
            return self._get_color_section_prompt()

        return self._get_step_section_prompt(rag_examples)

    def _get_color_section_prompt(self) -> str:
        """Generate prompt for color definitions section."""
        return get_template_color_section(
            prefix=self.prefix,
            header_comment=self.header_comment,
            suffix=self.suffix[:500],
        )

    def _get_step_section_prompt(self, rag_examples: list[dict] | None = None) -> str:
        """Generate prompt for a narration step section."""
        rag_context = ""
        if rag_examples:
            rag_context = "\n\nRAG EXAMPLES (similar code patterns):\n"
            for i, ex in enumerate(rag_examples[:2], 1):
                code_snippet = ex.get("content", "")[:400]
                rag_context += f"\nExample {i}:\n```python\n{code_snippet}\n```\n"

        return get_template_step_section(
            prefix=self.prefix[-800:],
            header_comment=self.header_comment,
            narration=self.narration,
            rag_context=rag_context,
            suffix=self.suffix[:300],
        )


@dataclass
class Template:
    """A code template with sections to be filled.

    Attributes:
        class_name: Name of the Scene class
        narration: List of narration sentences
        sections: List of Section objects to fill
        header: Fixed header code (imports, class definition, construct start)
        footer: Fixed footer code (end of construct method and class)
    """

    class_name: str
    narration: list[str]
    sections: list[Section] = field(default_factory=list)
    header: str = ""
    footer: str = ""

    @property
    def skeleton(self) -> str:
        """Get the template skeleton with placeholders."""
        parts = [self.header]

        for section in self.sections:
            parts.append(f"        {section.header_comment}")
            parts.append(f"        {section.placeholder}")
            if not section.is_color_section:
                parts.append("        self.wait(2)")
            parts.append("")

        parts.append(self.footer)
        return "\n".join(parts)

    def compile(self) -> str:
        """Assemble the final code from all filled sections.

        Returns:
            Complete Python code with all sections filled

        Raises:
            ValueError: If any section has not been filled
        """
        # Check all sections are filled
        unfilled = [s for s in self.sections if not s.content.strip()]
        if unfilled:
            indices = [s.index for s in unfilled]
            raise ValueError(f"Sections not filled: {indices}")

        # Start with header
        lines = [self.header]

        # Add each section
        for section in self.sections:
            lines.append(f"        {section.header_comment}")
            # Indent the content properly
            content_lines = section.content.strip().split("\n")
            for line in content_lines:
                # Preserve existing indentation but ensure minimum of 8 spaces
                stripped = line.lstrip()
                if stripped:
                    lines.append(f"        {stripped}")
            if not section.is_color_section:
                lines.append("        self.wait(2)")
            lines.append("")

        # Add footer
        lines.append(self.footer)

        return "\n".join(lines)

    def get_partial_code(self, up_to_section: int) -> str:
        """Get code with sections filled up to a certain point.

        Useful for providing context when filling later sections.

        Args:
            up_to_section: Fill sections with index < this value

        Returns:
            Partial code string
        """
        lines = [self.header]

        for section in self.sections:
            lines.append(f"        {section.header_comment}")
            if section.index < up_to_section and section.content:
                content_lines = section.content.strip().split("\n")
                for line in content_lines:
                    stripped = line.lstrip()
                    if stripped:
                        lines.append(f"        {stripped}")
            else:
                lines.append(f"        {section.placeholder}")
            if not section.is_color_section:
                lines.append("        self.wait(2)")
            lines.append("")

        lines.append(self.footer)
        return "\n".join(lines)


@dataclass
class FillResult:
    """Result of filling a section.

    Attributes:
        section_index: Index of the filled section
        content: Generated content
        success: Whether filling succeeded
        error: Error message if failed
    """

    section_index: int
    content: str = ""
    success: bool = True
    error: str | None = None


# ── Template Builder ─────────────────────────────────────────────────────────


def build_template(narration: list[str], class_name: str = "GeneratedScene") -> Template:
    """Create a code skeleton from a narration script.

    The template structure:
    1. Import and class definition header
    2. Color definitions section (auto-generated placeholder)
    3. One section per narration sentence with self.wait(2)
    4. Class footer

    Args:
        narration: List of narration sentences (each becomes a step)
        class_name: Name for the Scene class

    Returns:
        Template object with sections ready to fill
    """
    # Validate class name
    if not class_name.isidentifier():
        class_name = "GeneratedScene"

    # Build header
    header = f"""from manimlib import *


class {class_name}(Scene):
    def construct(self):
"""

    # Build footer
    footer = ""

    # Create sections
    sections = []

    # Section 0: Color definitions
    color_section = Section(
        index=0,
        narration="",
        prefix=header,
        suffix="",  # Will be updated after building all sections
        is_color_section=True,
    )
    sections.append(color_section)

    # Create step sections for each narration sentence
    for i, sentence in enumerate(narration, start=1):
        section = Section(
            index=i,
            narration=sentence.strip(),
            prefix="",  # Will be computed
            suffix="",  # Will be computed
        )
        sections.append(section)

    # Build the template to compute prefix/suffix for each section
    template = Template(
        class_name=class_name,
        narration=narration,
        sections=sections,
        header=header,
        footer=footer,
    )

    # Update prefix/suffix for each section
    _update_section_contexts(template)

    return template


def _update_section_contexts(template: Template) -> None:
    """Update prefix and suffix for each section in the template.

    This computes what code comes before and after each section,
    which is needed for FIM-style generation.
    """
    skeleton_lines = template.skeleton.split("\n")

    for section in template.sections:
        placeholder = section.placeholder
        placeholder_line_idx = None

        # Find the line with this section's placeholder
        for i, line in enumerate(skeleton_lines):
            if placeholder in line:
                placeholder_line_idx = i
                break

        if placeholder_line_idx is None:
            continue

        # Prefix: everything before the placeholder line
        prefix_lines = skeleton_lines[:placeholder_line_idx]
        section.prefix = "\n".join(prefix_lines)

        # Suffix: everything after the placeholder line
        suffix_lines = skeleton_lines[placeholder_line_idx + 1:]
        section.suffix = "\n".join(suffix_lines)


# ── Section Filler ───────────────────────────────────────────────────────────


async def fill_section(
    section: Section,
    llm_client: LLMClientProtocol,
    rag_examples: list[dict] | None = None,
    max_retries: int = 2,
) -> FillResult:
    """Fill a single section using FIM-style generation.

    Args:
        section: The section to fill
        llm_client: LLM client for code generation
        rag_examples: Optional RAG examples for context
        max_retries: Number of retries on failure

    Returns:
        FillResult with generated content or error
    """
    prompt = section.get_fim_prompt(rag_examples)

    for attempt in range(max_retries + 1):
        try:
            raw_content = await llm_client.generate_code(prompt)

            # Clean up the response
            content = _clean_generated_content(raw_content, section)

            if not content.strip():
                if attempt < max_retries:
                    logger.warning(
                        "Section %d: empty content, retrying (%d/%d)",
                        section.index, attempt + 1, max_retries
                    )
                    continue
                return FillResult(
                    section_index=section.index,
                    success=False,
                    error="Generated content is empty after cleanup",
                )

            section.content = content
            logger.debug("Section %d filled: %d chars", section.index, len(content))

            return FillResult(
                section_index=section.index,
                content=content,
                success=True,
            )

        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Section %d: generation failed, retrying (%d/%d): %s",
                    section.index, attempt + 1, max_retries, e
                )
                await asyncio.sleep(1)
                continue

            logger.error("Section %d: failed after %d attempts: %s",
                        section.index, max_retries + 1, e)
            return FillResult(
                section_index=section.index,
                success=False,
                error=str(e),
            )

    # Should not reach here
    return FillResult(
        section_index=section.index,
        success=False,
        error="Unexpected error in fill_section",
    )


def _clean_generated_content(content: str, section: Section) -> str:
    """Clean up generated content.

    Removes:
    - Markdown code fences
    - Section header comments (already in template)
    - self.wait() calls (already in template)
    - Leading/trailing whitespace

    Args:
        content: Raw generated content
        section: The section being filled (for context)

    Returns:
        Cleaned content string
    """
    # Remove markdown code fences
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:python)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

    # Remove section header if LLM repeated it
    header_patterns = [
        r"^#\s*===\s*STEP\s*\d+.*===\s*\n?",
        r"^#\s*===\s*COLOR\s*DEFINITIONS\s*===\s*\n?",
        r"^#\s*Step\s*\d+.*\n?",
    ]
    for pattern in header_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)

    # Remove self.wait() if present (for non-color sections)
    if not section.is_color_section:
        content = re.sub(r"\s*self\.wait\s*\([^)]*\)\s*$", "", content)
        content = re.sub(r"\n\s*self\.wait\s*\([^)]*\)", "", content)

    return content.strip()


# ── Parallel Section Filling ─────────────────────────────────────────────────


async def fill_all_sections_parallel(
    template: Template,
    llm_client: LLMClientProtocol,
    rag_service: ChromaDBService | None = None,
    max_concurrency: int = 3,
) -> list[FillResult]:
    """Fill all sections in parallel with controlled concurrency.

    Note: Color section is filled first, then step sections can be
    filled in parallel since they each have full context from the
    template structure.

    Args:
        template: Template with sections to fill
        llm_client: LLM client for generation
        rag_service: Optional RAG service for examples
        max_concurrency: Maximum concurrent LLM calls

    Returns:
        List of FillResult for each section
    """
    results: list[FillResult] = []

    # Fill color section first (section 0)
    color_sections = [s for s in template.sections if s.is_color_section]
    step_sections = [s for s in template.sections if not s.is_color_section]

    for section in color_sections:
        result = await fill_section(section, llm_client)
        results.append(result)
        if not result.success:
            logger.warning("Color section failed, using default colors")
            section.content = "PRIMARY_COLOR = BLUE\nSECONDARY_COLOR = GREEN\nACCENT_COLOR = YELLOW"

    # Update prefix for step sections to include filled color section
    _update_section_contexts(template)

    # Get RAG examples for step sections if available
    rag_examples_map: dict[int, list[dict]] = {}
    if rag_service and rag_service.available:
        for section in step_sections:
            try:
                examples = await rag_service.search_similar_scenes(
                    query=section.narration,
                    n_results=2,
                    prioritize_3b1b=True,
                )
                if examples:
                    rag_examples_map[section.index] = examples
            except Exception as e:
                logger.debug("RAG query failed for section %d: %s", section.index, e)

    # Fill step sections with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def fill_with_semaphore(section: Section) -> FillResult:
        async with semaphore:
            # Update prefix with previously filled sections
            # This provides context about what objects exist
            section.prefix = template.get_partial_code(section.index)
            rag_examples = rag_examples_map.get(section.index)
            return await fill_section(section, llm_client, rag_examples)

    # Run step sections in parallel
    step_tasks = [fill_with_semaphore(s) for s in step_sections]
    step_results = await asyncio.gather(*step_tasks, return_exceptions=True)

    for result in step_results:
        if isinstance(result, Exception):
            logger.error("Section fill raised exception: %s", result)
            results.append(FillResult(
                section_index=-1,
                success=False,
                error=str(result),
            ))
        else:
            results.append(result)

    return results


async def fill_all_sections_sequential(
    template: Template,
    llm_client: LLMClientProtocol,
    rag_service: ChromaDBService | None = None,
) -> list[FillResult]:
    """Fill all sections sequentially for maximum context.

    Each section gets the full context of previously filled sections
    in its prefix, allowing for better continuity.

    Args:
        template: Template with sections to fill
        llm_client: LLM client for generation
        rag_service: Optional RAG service for examples

    Returns:
        List of FillResult for each section
    """
    results: list[FillResult] = []

    for section in template.sections:
        # Update prefix with all previously filled sections
        section.prefix = template.get_partial_code(section.index)

        # Get RAG examples for this section
        rag_examples = None
        if rag_service and rag_service.available and not section.is_color_section:
            try:
                rag_examples = await rag_service.search_similar_scenes(
                    query=section.narration,
                    n_results=2,
                    prioritize_3b1b=True,
                )
            except Exception as e:
                logger.debug("RAG query failed for section %d: %s", section.index, e)

        result = await fill_section(section, llm_client, rag_examples)
        results.append(result)

        if not result.success:
            logger.warning(
                "Section %d failed: %s. Continuing with placeholder.",
                section.index, result.error
            )
            # Add a placeholder comment so compile doesn't fail
            if section.is_color_section:
                section.content = "PRIMARY_COLOR = BLUE"
            else:
                section.content = f"# TODO: Implement - {section.narration}"

    return results


# ── High-Level API ───────────────────────────────────────────────────────────


@dataclass
class TemplateGenerationResult:
    """Result of template-based code generation.

    Attributes:
        code: Generated Python code
        template: The template used
        fill_results: Results from filling each section
        success: Whether all sections were successfully filled
    """

    code: str
    template: Template
    fill_results: list[FillResult]
    success: bool


async def generate_from_narration(
    narration: list[str],
    llm_client: LLMClientProtocol,
    class_name: str = "GeneratedScene",
    rag_service: ChromaDBService | None = None,
    parallel: bool = True,
    max_concurrency: int = 3,
) -> TemplateGenerationResult:
    """Generate Manim code from a narration script using template-first approach.

    This is the main entry point for template-based generation.

    Args:
        narration: List of narration sentences (each becomes a visual step)
        llm_client: LLM client for code generation
        class_name: Name for the Scene class
        rag_service: Optional RAG service for examples
        parallel: Whether to fill sections in parallel (faster) or sequential (better context)
        max_concurrency: Maximum concurrent LLM calls if parallel

    Returns:
        TemplateGenerationResult with generated code and metadata
    """
    logger.info(
        "Template generation: %d narration steps, class=%s, parallel=%s",
        len(narration), class_name, parallel
    )

    # Build template skeleton
    template = build_template(narration, class_name)
    logger.debug("Template skeleton:\n%s", template.skeleton)

    # Fill all sections
    if parallel:
        fill_results = await fill_all_sections_parallel(
            template, llm_client, rag_service, max_concurrency
        )
    else:
        fill_results = await fill_all_sections_sequential(
            template, llm_client, rag_service
        )

    # Check success
    success = all(r.success for r in fill_results)
    failed_count = sum(1 for r in fill_results if not r.success)
    if failed_count > 0:
        logger.warning("%d/%d sections failed to fill", failed_count, len(fill_results))

    # Compile final code
    try:
        code = template.compile()
    except ValueError as e:
        logger.error("Template compilation failed: %s", e)
        # Return skeleton with placeholders as fallback
        code = template.skeleton
        success = False

    logger.info(
        "Template generation complete: %d sections, %d chars, success=%s",
        len(template.sections), len(code), success
    )

    return TemplateGenerationResult(
        code=code,
        template=template,
        fill_results=fill_results,
        success=success,
    )
