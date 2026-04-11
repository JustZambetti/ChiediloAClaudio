"""
Agent 3: Researcher

Uses web search (via Anthropic tool use) to find:
  - Existing solutions and competitors
  - Market/domain context
  - Adjacent ideas from other domains
  - Trend signals

The agentic tool loop in BaseAgent handles the search ↔ model cycle.
"""

from __future__ import annotations

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import IdeaDocument, ResearchDocument
from brainstorm.tools.web_search import WEB_SEARCH_TOOL, TOOL_HANDLERS

SYSTEM_PROMPT = """\
You are a fast, focused research analyst. Given an idea, your job is to \
search the web and build a factual research brief that will help creative \
agents expand on or differentiate the idea.

## What to research (use web_search for each)

1. **Existing solutions** — What direct competitors or partial solutions \
already exist? Search for the core concept + "startup", "product", "app", \
"company". Get specific names, not vague statements.

2. **Market context** — How big is this space? How mature? Who are the \
key players? Search for market size data and industry reports.

3. **Adjacent ideas** — What solved a similar problem in a completely \
different domain? (e.g., if the idea is about education, look at how \
healthcare or logistics solved similar coordination problems)

4. **Trend signals** — Is this space heating up or cooling? Search for \
recent news, funding rounds, or research papers from the last 2 years.

5. **Key data points** — Find 3-5 concrete numbers or facts (stats, \
studies, benchmarks) that are directly relevant.

## Rules
- Use web_search multiple times with different targeted queries.
- Be specific — cite names, numbers, dates when available.
- If a search returns nothing useful, try a different query angle.
- Do NOT hallucinate — only report what you actually found.

## Output format
After all searches, return ONLY valid JSON:

{
  "round": <int>,
  "existing_solutions": [
    {
      "title": "...",
      "summary": "...",
      "source_query": "the query that found this",
      "relevance": "why this matters for the idea"
    }
  ],
  "market_context": "...",
  "adjacent_ideas": [
    {
      "title": "...",
      "summary": "...",
      "source_query": "...",
      "relevance": "..."
    }
  ],
  "trend_signals": ["...", "..."],
  "key_data_points": ["...", "..."]
}
"""


class ResearcherAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Researcher", system_prompt=SYSTEM_PROMPT)

    def research(
        self,
        idea_doc: IdeaDocument,
        round_num: int,
    ) -> ResearchDocument:
        prompt = (
            f"## Round: {round_num}\n\n"
            f"## Idea to research\n"
            f"{idea_doc.model_dump_json(indent=2)}\n\n"
            "Search the web thoroughly and return your ResearchDocument JSON."
        )

        messages = [{"role": "user", "content": prompt}]

        raw = self.run_with_tools(
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_handler=TOOL_HANDLERS,
            max_iterations=8,
        )

        data = self.parse_json(raw)
        data["round"] = round_num
        return ResearchDocument(**data)
