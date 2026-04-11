"""Tests for brainstorm/agents/researcher.py — agentic tool loop and ResearchDocument."""

import json
from unittest.mock import patch

import pytest

from brainstorm.agents.researcher import ResearcherAgent
from brainstorm.schemas import ResearchDocument
from tests.conftest import (
    make_api_response,
    make_text_block,
    make_tool_use_block,
    make_mock_client,
)

# ---------------------------------------------------------------------------
# Sample research JSON output
# ---------------------------------------------------------------------------

def make_research_json(round_num: int = 1) -> str:
    return json.dumps({
        "round": round_num,
        "existing_solutions": [
            {
                "title": "Khan Academy",
                "summary": "Free adaptive learning platform",
                "source_query": "adaptive learning app students",
                "relevance": "Direct competitor",
            }
        ],
        "market_context": "Global EdTech market $340B, 16% YoY growth",
        "adjacent_ideas": [
            {
                "title": "Duolingo streak mechanic",
                "summary": "Gamified daily habit loop",
                "source_query": "gamification education retention",
                "relevance": "Habit loop increases retention",
            }
        ],
        "trend_signals": ["Post-COVID EdTech boom", "AI tutoring VC interest up 3x"],
        "key_data_points": ["1 in 3 students lacks tutoring access", "Average tutor costs $50/hr"],
    })


class TestResearcherInit:
    def test_name_and_system_prompt(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = ResearcherAgent()
        assert agent.name == "Researcher"
        assert len(agent.system_prompt) > 200


class TestResearcherResearch:
    def test_returns_research_document_no_tool_calls(self, sample_idea_doc):
        """Model returns final JSON without using any tools."""
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_research_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        result = agent.research(sample_idea_doc, round_num=1)
        assert isinstance(result, ResearchDocument)

    def test_round_number_set(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_research_json(2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        result = agent.research(sample_idea_doc, round_num=2)
        assert result.round == 2

    def test_research_with_one_tool_call(self, sample_idea_doc):
        """Model calls web_search once, then returns final JSON."""
        search_result = json.dumps([
            {"title": "EdTech startup raises $50M", "href": "http://...", "body": "Detail"}
        ])
        tool_block = make_tool_use_block("t1", "web_search", {"query": "EdTech market size"})
        final_block = make_text_block(make_research_json())

        responses = [
            make_api_response(tool_block),   # first: tool request
            make_api_response(final_block),  # second: final answer
        ]
        mock_client = make_mock_client(responses)

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        # Patch the actual search to avoid network calls
        with patch("brainstorm.tools.web_search.web_search", return_value=search_result):
            result = agent.research(sample_idea_doc, round_num=1)

        assert isinstance(result, ResearchDocument)
        assert mock_client.messages.create.call_count == 2

    def test_research_with_multiple_tool_calls(self, sample_idea_doc):
        """Model calls web_search three times, then produces final JSON."""
        search_result = json.dumps([{"title": "r", "href": "", "body": "b"}])
        tool1 = make_tool_use_block("t1", "web_search", {"query": "q1"})
        tool2 = make_tool_use_block("t2", "web_search", {"query": "q2"})
        tool3 = make_tool_use_block("t3", "web_search", {"query": "q3"})
        final = make_text_block(make_research_json())

        responses = [
            make_api_response(tool1),
            make_api_response(tool2),
            make_api_response(tool3),
            make_api_response(final),
        ]
        mock_client = make_mock_client(responses)

        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        with patch("brainstorm.tools.web_search.web_search", return_value=search_result):
            result = agent.research(sample_idea_doc, round_num=1)

        assert isinstance(result, ResearchDocument)
        assert mock_client.messages.create.call_count == 4

    def test_existing_solutions_populated(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_research_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        result = agent.research(sample_idea_doc, round_num=1)
        assert len(result.existing_solutions) > 0
        assert result.existing_solutions[0].title == "Khan Academy"

    def test_trend_signals_populated(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_research_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        result = agent.research(sample_idea_doc, round_num=1)
        assert len(result.trend_signals) > 0

    def test_idea_doc_in_prompt(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_research_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        agent.research(sample_idea_doc, round_num=1)
        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert sample_idea_doc.core_concept in prompt

    def test_invalid_json_raises_value_error(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block("This is not valid JSON"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        with pytest.raises(ValueError):
            agent.research(sample_idea_doc, round_num=1)

    def test_round_overrides_json_field(self, sample_idea_doc):
        """round is always set from the call parameter, not trusted from JSON."""
        # JSON says round=99 but we call with round_num=2
        research_data = json.loads(make_research_json(99))
        mock_client = make_mock_client(
            make_api_response(make_text_block(json.dumps(research_data)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ResearcherAgent()
        agent.client = mock_client

        result = agent.research(sample_idea_doc, round_num=2)
        assert result.round == 2
