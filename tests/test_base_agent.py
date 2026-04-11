"""Tests for brainstorm/agents/base.py — BaseAgent core functionality."""

import json
import pytest
from unittest.mock import MagicMock, patch

from brainstorm.agents.base import BaseAgent, _strip_json
from tests.conftest import (
    make_api_response,
    make_text_block,
    make_tool_use_block,
    make_mock_client,
)


# ---------------------------------------------------------------------------
# _strip_json helper
# ---------------------------------------------------------------------------

class TestStripJson:
    def test_plain_json_unchanged(self):
        raw = '{"key": "value"}'
        assert _strip_json(raw) == raw

    def test_strips_json_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _strip_json(raw) == '{"key": "value"}'

    def test_strips_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        assert _strip_json(raw) == '{"key": "value"}'

    def test_strips_leading_trailing_whitespace(self):
        raw = '  \n{"key": "value"}\n  '
        assert _strip_json(raw).strip() == '{"key": "value"}'

    def test_nested_json_preserved(self):
        raw = '```json\n{"a": {"b": [1, 2, 3]}}\n```'
        result = _strip_json(raw)
        assert json.loads(result) == {"a": {"b": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# BaseAgent.parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def setup_method(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            self.agent = BaseAgent("test", "system prompt")

    def test_parses_clean_json(self):
        result = self.agent.parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_with_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = self.agent.parse_json(raw)
        assert result == {"key": "value"}

    def test_parses_nested_json(self):
        raw = '{"a": [1, 2], "b": {"c": true}}'
        result = self.agent.parse_json(raw)
        assert result["a"] == [1, 2]
        assert result["b"]["c"] is True

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            self.agent.parse_json("this is not json")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            self.agent.parse_json("")

    def test_error_includes_agent_name(self):
        with pytest.raises(ValueError, match="test"):
            self.agent.parse_json("bad json")


# ---------------------------------------------------------------------------
# BaseAgent.extract_tagged
# ---------------------------------------------------------------------------

class TestExtractTagged:
    def setup_method(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            self.agent = BaseAgent("test", "system")

    def test_extracts_content(self):
        text = "Before <idea_document>hello world</idea_document> after"
        assert self.agent.extract_tagged(text, "idea_document") == "hello world"

    def test_extracts_multiline_content(self):
        text = "<result>\nline1\nline2\n</result>"
        assert self.agent.extract_tagged(text, "result") == "line1\nline2"

    def test_raises_when_tag_missing(self):
        with pytest.raises(ValueError, match="not found"):
            self.agent.extract_tagged("no tags here", "missing_tag")

    def test_error_includes_agent_name(self):
        with pytest.raises(ValueError, match="test"):
            self.agent.extract_tagged("no tags", "tag")

    def test_extracts_json_content(self):
        text = '<idea_document>{"key": "val"}</idea_document>'
        content = self.agent.extract_tagged(text, "idea_document")
        assert json.loads(content) == {"key": "val"}


# ---------------------------------------------------------------------------
# BaseAgent.run — single-shot API call
# ---------------------------------------------------------------------------

class TestBaseAgentRun:
    def test_returns_text_from_response(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("Hello from model"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "You are a test agent.")
            result = agent.run([{"role": "user", "content": "hi"}])
        assert result == "Hello from model"

    def test_system_prompt_sent_with_cache_control(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("ok"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "My system prompt")
            agent.run([{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "My system prompt"
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_model_set_correctly(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("ok"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            agent.run([{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_messages_forwarded(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("ok"))
        )
        messages = [{"role": "user", "content": "test message"}]
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            agent.run(messages)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == messages

    def test_tools_forwarded_when_provided(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("ok"))
        )
        tools = [{"name": "my_tool", "description": "does stuff", "input_schema": {}}]
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            agent.run([{"role": "user", "content": "hi"}], tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools

    def test_empty_response_returns_empty_string(self):
        mock_client = make_mock_client(make_api_response())  # no blocks
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_custom_max_tokens(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("ok"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            agent.run([{"role": "user", "content": "hi"}], max_tokens=1024)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# BaseAgent.run_with_tools — agentic tool loop
# ---------------------------------------------------------------------------

class TestRunWithTools:
    TOOL_SCHEMA = [{
        "name": "calculator",
        "description": "Add two numbers",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }]

    def test_single_tool_call_then_text(self):
        """Model uses tool once, then produces final text."""
        tool_block = make_tool_use_block("call_1", "calculator", {"a": 2, "b": 3})
        text_block = make_text_block("The result is 5")

        responses = [
            make_api_response(tool_block),        # first: tool request
            make_api_response(text_block),         # second: final answer
        ]
        mock_client = make_mock_client(responses)

        called_with = {}

        def fake_calculator(a, b):
            called_with["a"] = a
            called_with["b"] = b
            return a + b

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "What is 2+3?"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={"calculator": fake_calculator},
            )

        assert result == "The result is 5"
        assert called_with == {"a": 2, "b": 3}

    def test_multiple_tool_calls(self):
        """Model requests tool twice before answering."""
        tool1 = make_tool_use_block("c1", "calculator", {"a": 1, "b": 2})
        tool2 = make_tool_use_block("c2", "calculator", {"a": 3, "b": 4})
        final = make_text_block("Done: 3 and 7")

        responses = [
            make_api_response(tool1),
            make_api_response(tool2),
            make_api_response(final),
        ]
        mock_client = make_mock_client(responses)
        call_count = {"n": 0}

        def fake_calculator(a, b):
            call_count["n"] += 1
            return a + b

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "calc"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={"calculator": fake_calculator},
            )

        assert result == "Done: 3 and 7"
        assert call_count["n"] == 2

    def test_unknown_tool_returns_error_result(self):
        """If model requests an unregistered tool, error is fed back."""
        tool_block = make_tool_use_block("c1", "unknown_tool", {})
        final = make_text_block("Handled the error")

        responses = [make_api_response(tool_block), make_api_response(final)]
        mock_client = make_mock_client(responses)

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={},  # no handlers registered
            )

        assert result == "Handled the error"
        # Verify a tool_result with error was sent back
        second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
        tool_result_turn = second_call_messages[-1]
        assert tool_result_turn["role"] == "user"
        assert "Error" in tool_result_turn["content"][0]["content"]

    def test_tool_exception_is_caught(self):
        """Tool that raises an exception should not crash the loop."""
        tool_block = make_tool_use_block("c1", "calculator", {"a": 1, "b": 2})
        final = make_text_block("Got the error")

        responses = [make_api_response(tool_block), make_api_response(final)]
        mock_client = make_mock_client(responses)

        def broken_calc(**kwargs):
            raise RuntimeError("calculation exploded")

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "calc"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={"calculator": broken_calc},
            )
        assert result == "Got the error"

    def test_immediate_text_no_tools(self):
        """If model never requests tools, return text directly."""
        final = make_text_block("Answer without tools")
        mock_client = make_mock_client(make_api_response(final))

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={},
            )
        assert result == "Answer without tools"
        assert mock_client.messages.create.call_count == 1

    def test_max_iterations_respected(self):
        """Loop should stop after max_iterations even if tools keep being requested."""
        tool_block = make_tool_use_block("c1", "calculator", {"a": 1, "b": 2})
        # Always return a tool request — never a final text
        mock_client = make_mock_client([make_api_response(tool_block)] * 10)

        def fake_calc(a, b):
            return a + b

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = BaseAgent("test", "sys")
            result = agent.run_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=self.TOOL_SCHEMA,
                tool_handler={"calculator": fake_calc},
                max_iterations=3,
            )
        assert result == "Max tool iterations reached."
        assert mock_client.messages.create.call_count == 3
