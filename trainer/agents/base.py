"""
Base agent class.

All agents share:
- Anthropic client with prompt caching on system prompts
- A simple run() method that handles multi-turn messages
- JSON extraction helpers for structured outputs
"""

from __future__ import annotations
import json
import re
from typing import Any

import anthropic

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096


def _strip_json(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


class BaseAgent:
    """
    Thin wrapper around the Anthropic Messages API.

    System prompts are cached via cache_control so repeated agent
    invocations within the same session avoid re-processing tokens.
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()

    def _system_block(self) -> list[dict]:
        """Return system prompt with prompt caching enabled."""
        return [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def run(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "system": self._system_block(),
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "text":
                return block.text

        return ""

    def run_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        tool_handler: dict[str, Any],
        max_iterations: int = 6,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=self._system_block(),
                messages=current_messages,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_uses:
                return text_blocks[0].text if text_blocks else ""

            current_messages.append({
                "role": "assistant",
                "content": [b.model_dump() for b in response.content],
            })

            tool_results = []
            for tu in tool_uses:
                fn = tool_handler.get(tu.name)
                if fn is None:
                    result = f"Error: unknown tool '{tu.name}'"
                else:
                    try:
                        result = fn(**tu.input)
                    except Exception as exc:
                        result = f"Error: {exc}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": str(result),
                })

            current_messages.append({"role": "user", "content": tool_results})

        return "Max tool iterations reached."

    def parse_json(self, text: str) -> dict:
        """Parse JSON from model output, stripping markdown fences."""
        clean = _strip_json(text)
        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[{self.name}] Failed to parse JSON: {exc}\nRaw:\n{text[:400]}"
            )

    def extract_tagged(self, text: str, tag: str) -> str:
        """Extract content between <tag>…</tag> in model output."""
        pattern = rf"<{tag}>([\s\S]+?)</{tag}>"
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"[{self.name}] Tag <{tag}> not found in output.")
        return match.group(1).strip()
