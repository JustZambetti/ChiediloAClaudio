"""Tests for brainstorm/agents/expander.py — variant generation and anti-convergence."""

import json
from unittest.mock import patch

import pytest

from brainstorm.agents.expander import ExpanderAgent
from brainstorm.schemas import IdeaSet
from tests.conftest import make_api_response, make_text_block, make_mock_client


# ---------------------------------------------------------------------------
# Sample JSON the mock LLM returns
# ---------------------------------------------------------------------------

def make_expander_json(round_num: int = 1, num_variants: int = 3) -> str:
    variants = [
        {
            "id": f"v{round_num}-variant-{i}",
            "title": f"Variant {i} Title",
            "description": f"Description of variant {i} in round {round_num}.",
            "technique_used": ["SCAMPER-Eliminate", "Constraint-Inversion", "Domain-Analogy:healthcare"][i % 3],
            "diverges_from_original_by": f"Changes element {i} of the original idea",
            "addresses_critique": "No revenue model" if i == 0 else None,
            "inspired_by_research": "Khan Academy competitor analysis" if i == 1 else None,
        }
        for i in range(num_variants)
    ]
    return json.dumps({
        "round": round_num,
        "directive_received": "Explore broadly using all SCAMPER verbs",
        "variants": variants,
    })


class TestExpanderInit:
    def test_name_and_system_prompt(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = ExpanderAgent()
        assert agent.name == "Expander"
        assert "SCAMPER" in agent.system_prompt


class TestExpanderExpand:
    def _make_agent(self, response_text: str) -> ExpanderAgent:
        mock_client = make_mock_client(
            make_api_response(make_text_block(response_text))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client
        return agent

    def test_returns_idea_set(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(1))
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="Explore broadly",
        )
        assert isinstance(result, IdeaSet)

    def test_round_number_set(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(2))
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=2,
            directive="Go deeper",
        )
        assert result.round == 2

    def test_variants_populated(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(1, num_variants=5))
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="Explore broadly",
        )
        assert len(result.variants) == 5

    def test_directive_in_prompt(self, sample_idea_doc, sample_critique, sample_research):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_expander_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client

        directive = "Focus specifically on constraint inversion for financial model"
        agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive=directive,
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert directive in prompt

    def test_critique_included_in_prompt(self, sample_idea_doc, sample_critique, sample_research):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_expander_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client

        agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="test",
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert sample_critique.weaknesses[0] in prompt

    def test_research_included_in_prompt(self, sample_idea_doc, sample_critique, sample_research):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_expander_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client

        agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="test",
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert sample_research.existing_solutions[0].title in prompt

    def test_prior_idea_sets_included_for_anti_convergence(
        self, sample_idea_doc, sample_critique, sample_research, sample_idea_set
    ):
        """Prior variants should appear in prompt so Expander avoids repeating them."""
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_expander_json(2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client

        agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=2,
            directive="Go deeper",
            prior_idea_sets=[sample_idea_set],
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        # Prior variant titles should appear in the anti-convergence section
        assert sample_idea_set.variants[0].title in prompt

    def test_no_prior_sets_round_1(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(1))
        # Should not raise
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="Explore broadly",
            prior_idea_sets=None,
        )
        assert isinstance(result, IdeaSet)

    def test_variant_ids_are_unique(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(1, num_variants=5))
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="test",
        )
        ids = [v.id for v in result.variants]
        assert len(ids) == len(set(ids))

    def test_directive_received_preserved(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent(make_expander_json(1))
        result = agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="Explore broadly using all SCAMPER verbs",
        )
        assert result.directive_received == "Explore broadly using all SCAMPER verbs"

    def test_invalid_json_raises(self, sample_idea_doc, sample_critique, sample_research):
        agent = self._make_agent("not json at all")
        with pytest.raises(ValueError):
            agent.expand(
                idea_doc=sample_idea_doc,
                critique=sample_critique,
                research=sample_research,
                round_num=1,
                directive="test",
            )

    def test_uses_large_max_tokens(self, sample_idea_doc, sample_critique, sample_research):
        """Expander needs a large token budget for 5-7 variants."""
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_expander_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = ExpanderAgent()
        agent.client = mock_client

        agent.expand(
            idea_doc=sample_idea_doc,
            critique=sample_critique,
            research=sample_research,
            round_num=1,
            directive="test",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] >= 4096
