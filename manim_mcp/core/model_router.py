"""Model router for selecting appropriate LLM models based on task type.

This module centralizes model selection logic, making it easy to configure
which models are used for different tasks (code generation, narration, critique, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that require LLM inference."""

    # Code generation tasks
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_FIX = "code_fix"

    # Narration tasks
    NARRATION_SCRIPT = "narration_script"
    NARRATION_FROM_CODE = "narration_from_code"

    # Analysis tasks
    CONCEPT_ANALYSIS = "concept_analysis"
    SCENE_PLANNING = "scene_planning"

    # Critique tasks
    SELF_CRITIQUE = "self_critique"
    VERIFICATION = "verification"

    # Schema/template tasks
    SCHEMA_GENERATION = "schema_generation"
    TEMPLATE_FILL = "template_fill"

    # TTS (audio generation)
    TTS_AUDIO = "tts_audio"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider: str  # gemini, claude, deepseek
    temperature: float = 0.7
    max_tokens: int | None = None

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


class ModelRouter:
    """Routes tasks to appropriate models based on configuration.

    The router uses the application config to determine which models to use
    for different task types. This centralizes model selection and makes it
    easy to swap models for specific tasks.

    Usage:
        router = ModelRouter(config)
        model = router.get_model(TaskType.CODE_GENERATION)
        # model.name = "gemini-3-flash-preview"
    """

    def __init__(self, config: ManimMCPConfig) -> None:
        self.config = config
        self._task_overrides: dict[TaskType, ModelConfig] = {}

        # Build default model configs from application config
        self._default_models = self._build_default_models()

        logger.info(
            "ModelRouter initialized with provider=%s, model=%s",
            config.llm_provider,
            self._get_primary_model_name(),
        )

    def _get_primary_model_name(self) -> str:
        """Get the primary model name based on configured provider."""
        provider = self.config.llm_provider.lower()
        if provider == "claude":
            return self.config.claude_model
        elif provider == "deepseek":
            return self.config.deepseek_model
        else:  # gemini (default)
            return self.config.gemini_model

    def _build_default_models(self) -> dict[TaskType, ModelConfig]:
        """Build default model configurations for each task type."""
        provider = self.config.llm_provider.lower()
        primary_model = self._get_primary_model_name()

        # Most tasks use the primary model
        defaults = {
            TaskType.CODE_GENERATION: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.7,
            ),
            TaskType.CODE_REVIEW: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.3,  # Lower temp for more consistent reviews
            ),
            TaskType.CODE_FIX: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.5,
            ),
            TaskType.NARRATION_SCRIPT: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.8,  # Slightly higher for creative narration
            ),
            TaskType.NARRATION_FROM_CODE: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.7,
            ),
            TaskType.CONCEPT_ANALYSIS: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.5,
            ),
            TaskType.SCENE_PLANNING: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.6,
            ),
            TaskType.SELF_CRITIQUE: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.3,  # Low temp for consistent critique
            ),
            TaskType.VERIFICATION: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.2,  # Very low for verification
            ),
            TaskType.SCHEMA_GENERATION: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.5,
            ),
            TaskType.TEMPLATE_FILL: ModelConfig(
                name=primary_model,
                provider=provider,
                temperature=0.6,
            ),
            # TTS uses specific Gemini model regardless of primary provider
            TaskType.TTS_AUDIO: ModelConfig(
                name=self.config.tts_model,
                provider="gemini",  # TTS only works with Gemini
                temperature=1.0,  # TTS uses fixed temperature
            ),
        }

        return defaults

    def get_model(self, task_type: TaskType) -> ModelConfig:
        """Get the model configuration for a specific task type.

        Args:
            task_type: The type of task to get a model for

        Returns:
            ModelConfig with the model name and settings
        """
        # Check for task-specific override first
        if task_type in self._task_overrides:
            return self._task_overrides[task_type]

        # Fall back to defaults
        return self._default_models.get(
            task_type,
            self._default_models[TaskType.CODE_GENERATION],  # Ultimate fallback
        )

    def get_model_name(self, task_type: TaskType) -> str:
        """Get just the model name for a task type.

        Convenience method for when you only need the model name string.
        """
        return self.get_model(task_type).name

    def set_task_model(self, task_type: TaskType, model: ModelConfig) -> None:
        """Override the model for a specific task type.

        Args:
            task_type: The task type to override
            model: The model configuration to use
        """
        self._task_overrides[task_type] = model
        logger.info("Set model override for %s: %s", task_type.value, model)

    def clear_task_override(self, task_type: TaskType) -> None:
        """Clear the override for a specific task type."""
        if task_type in self._task_overrides:
            del self._task_overrides[task_type]
            logger.info("Cleared model override for %s", task_type.value)

    def get_all_models(self) -> dict[TaskType, ModelConfig]:
        """Get all model configurations (defaults merged with overrides)."""
        result = dict(self._default_models)
        result.update(self._task_overrides)
        return result

    @property
    def primary_provider(self) -> str:
        """Get the primary LLM provider."""
        return self.config.llm_provider.lower()

    @property
    def primary_model(self) -> str:
        """Get the primary model name."""
        return self._get_primary_model_name()


# Singleton instance (initialized lazily)
_router: ModelRouter | None = None


def get_model_router(config: ManimMCPConfig | None = None) -> ModelRouter:
    """Get the global model router instance.

    Args:
        config: Configuration to use. Required on first call.

    Returns:
        The global ModelRouter instance
    """
    global _router

    if _router is None:
        if config is None:
            raise RuntimeError("ModelRouter not initialized. Provide config on first call.")
        _router = ModelRouter(config)

    return _router


def reset_model_router() -> None:
    """Reset the global model router. Useful for testing."""
    global _router
    _router = None
