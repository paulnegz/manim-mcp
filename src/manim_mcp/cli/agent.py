"""Gemini function-calling agent loop for the CLI ``prompt`` command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from google import genai
from google.genai.types import Content, FunctionResponse, Part, Tool

from manim_mcp.cli.output import Printer, spinner
from manim_mcp.cli.tools_schema import TOOL_DECLARATIONS, execute_tool

if TYPE_CHECKING:
    from manim_mcp.bootstrap import AppContext

AGENT_SYSTEM = (
    "You are a Manim animation assistant running in a non-interactive CLI.\n"
    "You have tools to generate, edit, list, get, and delete Manim animations.\n"
    "Make reasonable assumptions rather than asking clarifying questions — "
    "the user cannot reply during execution.\n"
    "When the user asks for multiple steps (e.g. 'create then edit'), "
    "chain the appropriate tool calls.\n"
    "Always finish with a brief summary of what was done."
)


async def run_agent_loop(
    prompt: str,
    ctx: AppContext,
    printer: Printer,
    *,
    max_turns: int = 10,
) -> int:
    """Run a multi-turn Gemini function-calling loop.

    Returns an exit code: 0 on success, 1 on error.
    """
    client = genai.Client(api_key=ctx.config.gemini_api_key)
    tools = [Tool(function_declarations=TOOL_DECLARATIONS)]

    conversation: list[Content] = [
        Content(role="user", parts=[Part.from_text(text=prompt)]),
    ]

    for turn in range(max_turns):
        # 1. Send conversation to Gemini
        with spinner(f"Thinking (turn {turn + 1}/{max_turns})"):
            response = await client.aio.models.generate_content(
                model=ctx.config.gemini_model,
                contents=conversation,
                config={
                    "tools": tools,
                    "system_instruction": AGENT_SYSTEM,
                },
            )

        # 2. Separate text parts from function_call parts
        text_parts: list[str] = []
        function_calls: list[Part] = []

        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    function_calls.append(part)

        # Print any text the model produced
        for text in text_parts:
            printer.agent_text(text)

        # 3. If no function calls, the agent is done
        if not function_calls:
            break

        # Append the model's full response to the conversation
        model_parts = []
        for text in text_parts:
            model_parts.append(Part.from_text(text=text))
        model_parts.extend(function_calls)
        conversation.append(Content(role="model", parts=model_parts))

        # 4. Execute each function call
        response_parts: list[Part] = []
        for fc_part in function_calls:
            fc = fc_part.function_call
            args = dict(fc.args) if fc.args else {}
            printer.agent_tool_call(fc.name, args)

            with spinner(f"Running {fc.name}"):
                result = await execute_tool(fc.name, args, ctx)

            printer.agent_tool_result(fc.name, result)

            response_parts.append(
                Part.from_function_response(
                    name=fc.name,
                    response=_ensure_serialisable(result),
                )
            )

        # 5. Append function responses and continue
        conversation.append(Content(role="user", parts=response_parts))

    else:
        printer.warn(f"Agent reached maximum turns ({max_turns}).")

    return 0


def _ensure_serialisable(obj: dict) -> dict:
    """Round-trip through JSON to strip non-serialisable types (enums, etc.)."""
    return json.loads(json.dumps(obj, default=str))
