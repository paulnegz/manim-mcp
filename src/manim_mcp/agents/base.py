"""Base agent class for the multi-agent pipeline."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig
    from manim_mcp.core.llm import GeminiClient
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseAgent(ABC):
    """Abstract base class for all pipeline agents.

    Each agent has access to:
    - llm: The Gemini client for LLM calls
    - rag: The ChromaDB service (may be None or unavailable)
    - config: Application configuration
    """

    name: str = "base"

    def __init__(
        self,
        llm: GeminiClient,
        rag: ChromaDBService | None,
        config: ManimMCPConfig,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.config = config

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input and return agent-specific output.

        Subclasses must implement this method.
        """
        ...

    @property
    def rag_available(self) -> bool:
        """Check if RAG service is available."""
        return self.rag is not None and self.rag.available

    async def _llm_call(
        self,
        prompt: str,
        system: str,
    ) -> str:
        """Make an LLM call with the given prompt and system instruction.

        Uses the provider-agnostic LLM client abstraction.
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

    async def _llm_call_json(
        self,
        prompt: str,
        system: str,
    ) -> dict:
        """Make an LLM call expecting JSON output.

        Uses the provider-agnostic LLM client abstraction.
        """
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
            # Claude doesn't support response_mime_type, so we add JSON instruction
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

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM: %s", e)
            return {}

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences from text."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:python)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
