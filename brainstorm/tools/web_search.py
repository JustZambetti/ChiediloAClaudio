"""
Web search tool using DuckDuckGo (no API key required).

Exposed as both:
  - A plain Python function  → used by tests or direct calls
  - An Anthropic tool schema → passed to the Researcher agent
"""

from __future__ import annotations
import json

try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy fallback
        _DDGS_AVAILABLE = True
    except ImportError:
        _DDGS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tool schema (Anthropic format)
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL: dict = {
    "name": "web_search",
    "description": (
        "Search the web for information. Use targeted queries. "
        "Call multiple times with different queries to cover different angles."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo and return results as a JSON string.

    Returns a JSON array of objects with keys: title, href, body.
    Falls back to a mock response if duckduckgo-search is not installed.
    """
    max_results = min(max_results, 10)

    if not _DDGS_AVAILABLE:
        return json.dumps([{
            "title": "duckduckgo-search not installed",
            "href": "",
            "body": "Install with: pip install duckduckgo-search",
        }])

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps([{"title": "Search error", "href": "", "body": str(exc)}])


# ---------------------------------------------------------------------------
# Handler map (passed to BaseAgent.run_with_tools)
# ---------------------------------------------------------------------------

TOOL_HANDLERS: dict = {
    "web_search": web_search,
}
