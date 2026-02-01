"""Prompt loader for Manim MCP.

All system prompts are stored as .md files in this directory and loaded at runtime.
This allows for easier editing and version control of prompts.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt from a .md file.

    Args:
        name: Name of the prompt file (without .md extension)

    Returns:
        The prompt content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_path = PROMPTS_DIR / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8").strip()
    logger.debug("Loaded prompt '%s' (%d chars)", name, len(content))
    return content


def load_prompt_with_vars(name: str, **variables) -> str:
    """Load a prompt and substitute variables.

    Args:
        name: Name of the prompt file (without .md extension)
        **variables: Variables to substitute in the prompt using {var_name} syntax

    Returns:
        The prompt content with variables substituted
    """
    content = load_prompt(name)
    if variables:
        content = content.format(**variables)
    return content


def clear_cache() -> None:
    """Clear the prompt cache. Useful for development/testing."""
    load_prompt.cache_clear()


# Convenience exports for commonly used prompts
def get_generate_system(latex_available: bool) -> str:
    """Get the generation system prompt with appropriate LaTeX instructions."""
    base = load_prompt("generate_base")

    if latex_available:
        latex_instructions = load_prompt("latex_instructions")
        latex_patterns = load_prompt("latex_patterns")
    else:
        latex_instructions = load_prompt("no_latex_instructions")
        latex_patterns = load_prompt("no_latex_patterns")

    return base.format(
        latex_instructions=latex_instructions,
        latex_patterns=latex_patterns,
    )


def get_edit_system() -> str:
    """Get the edit system prompt."""
    return load_prompt("edit_system")


def get_fix_system() -> str:
    """Get the fix system prompt."""
    return load_prompt("fix_system")


def get_concept_analyzer_system() -> str:
    """Get the concept analyzer agent system prompt."""
    return load_prompt("concept_analyzer")


def get_scene_planner_system() -> str:
    """Get the scene planner agent system prompt."""
    return load_prompt("scene_planner")


def get_code_reviewer_system() -> str:
    """Get the code reviewer agent system prompt."""
    return load_prompt("code_reviewer")


def get_code_generator_system(latex_available: bool = True) -> str:
    """Get the code generator agent system prompt."""
    return load_prompt("code_generator")


# ── TTS / Narration prompts ──────────────────────────────────────────────

def get_tts_narration_script(prompt: str) -> str:
    """Get the narration script generation prompt."""
    return load_prompt_with_vars("tts_narration_script", prompt=prompt)


def get_tts_narration_from_code(
    code: str,
    prompt: str,
    target_duration: float,
    target_word_count: int,
    sentence_count: int,
) -> str:
    """Get the narration-from-code generation prompt."""
    return load_prompt_with_vars(
        "tts_narration_from_code",
        code=code,
        prompt=prompt,
        target_duration=f"{target_duration:.1f}",
        target_word_count=str(target_word_count),
        sentence_count=str(sentence_count),
    )


def get_tts_narration_fallback(
    prompt: str,
    target_duration: float,
    target_word_count: int,
    sentence_count: int,
) -> str:
    """Get the fallback narration generation prompt."""
    return load_prompt_with_vars(
        "tts_narration_fallback",
        prompt=prompt,
        target_duration=f"{target_duration:.1f}",
        target_word_count=str(target_word_count),
        sentence_count=str(sentence_count),
    )


# ── Self-critique prompts ────────────────────────────────────────────────

def get_self_critique_system() -> str:
    """Get the self-critique system prompt."""
    return load_prompt("self_critique_system")


def get_self_critique_fix() -> str:
    """Get the self-critique fix system prompt."""
    return load_prompt("self_critique_fix")


def get_self_critique_verify() -> str:
    """Get the self-critique verify system prompt."""
    return load_prompt("self_critique_verify")


# ── Schema/Template generation prompts ───────────────────────────────────

def get_schema_generator_system() -> str:
    """Get the schema generator system prompt."""
    return load_prompt("schema_generator_system")


def get_schema_generator_narration(script_text: str, step_count: int) -> str:
    """Get the schema generator narration addition prompt."""
    return load_prompt_with_vars(
        "schema_generator_narration",
        script_text=script_text,
        step_count=str(step_count),
    )


def get_template_color_section(prefix: str, header_comment: str, suffix: str) -> str:
    """Get the template color section FIM prompt."""
    return load_prompt_with_vars(
        "template_color_section",
        prefix=prefix,
        header_comment=header_comment,
        suffix=suffix,
    )


def get_template_step_section(
    prefix: str,
    header_comment: str,
    narration: str,
    rag_context: str = "",
    suffix: str = "",
) -> str:
    """Get the template step section FIM prompt."""
    return load_prompt_with_vars(
        "template_step_section",
        prefix=prefix,
        header_comment=header_comment,
        narration=narration,
        rag_context=rag_context,
        suffix=suffix,
    )
