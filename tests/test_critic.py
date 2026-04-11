"""Tests for brainstorm/agents/critic.py — critique output and round-awareness."""

import json
from unittest.mock import patch

import pytest

from brainstorm.agents.critic import CriticAgent
from brainstorm.schemas import CritiqueDocument
from tests.conftest import make_api_response, make_text_block, make_mock_client

# ---------------------------------------------------------------------------
# Sample JSON output the mock LLM will return
# ---------------------------------------------------------------------------

def make_critique_json(round_num: int = 1) -> str:
    return json.dumps({
        "round": round_num,
        "strengths": ["Clear problem statement", "Large market"],
        "weaknesses": ["No revenue model", "Relies on unproven AI accuracy"],
        "assumptions_challenged": [
            "Assumes students will self-select: most disengage from homework entirely"
        ],
        "risks": ["Competition from Khan Academy", "Low switching costs"],
        "open_questions": ["How do you acquire the first 1000 students?"],
        "feasibility_scores": {
            "technical": 0.65,
            "financial": 0.50,
            "social": 0.70,
            "timing": 0.80,
        },
    })


class TestCriticInit:
    def test_name_and_system_prompt(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = CriticAgent()
        assert agent.name == "Critic"
        assert len(agent.system_prompt) > 200


class TestCriticCritique:
    def _make_agent(self, response_text: str) -> CriticAgent:
        mock_client = make_mock_client(
            make_api_response(make_text_block(response_text))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = CriticAgent()
        agent.client = mock_client
        return agent

    def test_returns_critique_document(self, sample_idea_doc):
        agent = self._make_agent(make_critique_json(1))
        result = agent.critique(sample_idea_doc, round_num=1)
        assert isinstance(result, CritiqueDocument)

    def test_round_number_preserved(self, sample_idea_doc):
        agent = self._make_agent(make_critique_json(2))
        result = agent.critique(sample_idea_doc, round_num=2)
        assert result.round == 2

    def test_strengths_populated(self, sample_idea_doc):
        agent = self._make_agent(make_critique_json())
        result = agent.critique(sample_idea_doc, round_num=1)
        assert len(result.strengths) > 0

    def test_weaknesses_populated(self, sample_idea_doc):
        agent = self._make_agent(make_critique_json())
        result = agent.critique(sample_idea_doc, round_num=1)
        assert len(result.weaknesses) > 0

    def test_feasibility_scores_all_axes(self, sample_idea_doc):
        agent = self._make_agent(make_critique_json())
        result = agent.critique(sample_idea_doc, round_num=1)
        for axis in ("technical", "financial", "social", "timing"):
            assert axis in result.feasibility_scores
            assert 0.0 <= result.feasibility_scores[axis] <= 1.0

    def test_idea_doc_json_in_prompt(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_critique_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = CriticAgent()
        agent.client = mock_client
        agent.critique(sample_idea_doc, round_num=1)

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt_text = call_messages[0]["content"]
        # IdeaDocument should appear in the prompt
        assert sample_idea_doc.core_concept in prompt_text

    def test_prior_idea_sets_included_in_prompt(self, sample_idea_doc, sample_idea_set):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_critique_json(2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = CriticAgent()
        agent.client = mock_client
        agent.critique(sample_idea_doc, round_num=2, prior_idea_sets=[sample_idea_set])

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt_text = call_messages[0]["content"]
        # Prior variants should appear so the Critic knows not to repeat them
        assert sample_idea_set.variants[0].title in prompt_text

    def test_no_prior_idea_sets_round_1(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_critique_json(1)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = CriticAgent()
        agent.client = mock_client
        # Should not raise
        result = agent.critique(sample_idea_doc, round_num=1, prior_idea_sets=None)
        assert isinstance(result, CritiqueDocument)

    def test_invalid_json_raises_value_error(self, sample_idea_doc):
        agent = self._make_agent("This is not JSON")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            agent.critique(sample_idea_doc, round_num=1)

    def test_round_num_in_prompt(self, sample_idea_doc):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_critique_json(3)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = CriticAgent()
        agent.client = mock_client
        agent.critique(sample_idea_doc, round_num=3)

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt_text = call_messages[0]["content"]
        assert "3" in prompt_text  # Round number appears
