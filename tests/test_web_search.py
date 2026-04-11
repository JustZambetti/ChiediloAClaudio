"""Tests for brainstorm/tools/web_search.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from brainstorm.tools.web_search import (
    TOOL_HANDLERS,
    WEB_SEARCH_TOOL,
    web_search,
)


class TestWebSearchToolSchema:
    def test_tool_has_name(self):
        assert WEB_SEARCH_TOOL["name"] == "web_search"

    def test_tool_has_description(self):
        assert len(WEB_SEARCH_TOOL["description"]) > 10

    def test_tool_input_schema_has_query_required(self):
        schema = WEB_SEARCH_TOOL["input_schema"]
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_tool_has_max_results_optional(self):
        schema = WEB_SEARCH_TOOL["input_schema"]
        assert "max_results" in schema["properties"]
        # max_results is not in required
        assert "max_results" not in schema.get("required", [])

    def test_tool_handlers_contains_web_search(self):
        assert "web_search" in TOOL_HANDLERS
        assert callable(TOOL_HANDLERS["web_search"])


class TestWebSearchFunction:
    def test_returns_json_string(self):
        mock_results = [{"title": "Result", "href": "http://example.com", "body": "Content"}]
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", True):
            with patch("brainstorm.tools.web_search.DDGS") as mock_ddgs:
                mock_ctx = MagicMock()
                mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ctx)
                mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
                mock_ctx.text.return_value = iter(mock_results)

                result = web_search("test query")

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Result"

    def test_default_max_results_is_5(self):
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", True):
            with patch("brainstorm.tools.web_search.DDGS") as mock_ddgs:
                mock_ctx = MagicMock()
                mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ctx)
                mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
                mock_ctx.text.return_value = iter([])

                web_search("test")
                mock_ctx.text.assert_called_once_with("test", max_results=5)

    def test_max_results_capped_at_10(self):
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", True):
            with patch("brainstorm.tools.web_search.DDGS") as mock_ddgs:
                mock_ctx = MagicMock()
                mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ctx)
                mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
                mock_ctx.text.return_value = iter([])

                web_search("test", max_results=999)
                mock_ctx.text.assert_called_once_with("test", max_results=10)

    def test_fallback_when_ddgs_not_available(self):
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", False):
            result = web_search("test query")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "not installed" in parsed[0]["title"].lower()

    def test_handles_search_exception_gracefully(self):
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", True):
            with patch("brainstorm.tools.web_search.DDGS") as mock_ddgs:
                mock_ddgs.return_value.__enter__ = MagicMock(
                    side_effect=Exception("network error")
                )
                mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)

                result = web_search("test")

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert "error" in parsed[0]["title"].lower()

    def test_custom_max_results_forwarded(self):
        with patch("brainstorm.tools.web_search._DDGS_AVAILABLE", True):
            with patch("brainstorm.tools.web_search.DDGS") as mock_ddgs:
                mock_ctx = MagicMock()
                mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_ctx)
                mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
                mock_ctx.text.return_value = iter([])

                web_search("query", max_results=3)
                mock_ctx.text.assert_called_once_with("query", max_results=3)
