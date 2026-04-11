"""Tests for brainstorm/agents/harvester.py — interview loop and IdeaDocument extraction."""

import json
from unittest.mock import patch, MagicMock

import pytest

from brainstorm.agents.harvester import HarvesterAgent
from brainstorm.schemas import IdeaDocument
from tests.conftest import make_api_response, make_text_block, make_mock_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_IDEA_JSON = {
    "version": 1,
    "core_concept": "AI tutoring app that adapts to learning pace",
    "problem_solved": "One-size-fits-all classroom pacing",
    "target_audience": "High school students",
    "inspiration_source": "Watching a sibling struggle",
    "constraints": ["Must work offline", "Under $10/month"],
    "unknowns": ["B2B vs B2C"],
    "excitement_level": 8,
    "domain_tags": ["education", "ai"],
    "raw_quotes": ["every kid deserves a private tutor"],
}

def make_idea_doc_response(data: dict | None = None) -> str:
    """Build a model response string that contains a valid <idea_document> tag."""
    doc = data or VALID_IDEA_JSON
    return (
        "I've captured your idea. Here's the IdeaDocument:\n"
        f"<idea_document>\n{json.dumps(doc)}\n</idea_document>\n"
        "Does this look right to you?"
    )


def make_harvester_with_responses(responses: list):
    """Return a HarvesterAgent backed by a mock client with the given responses."""
    mock_client = make_mock_client(responses)
    with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
        agent = HarvesterAgent()
    # Re-patch the already-created client so subsequent calls use our mock
    agent.client = mock_client
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHarvesterInit:
    def test_creates_with_empty_history(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = HarvesterAgent()
        assert agent.conversation_history == []
        assert agent.name == "Harvester"

    def test_system_prompt_non_empty(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = HarvesterAgent()
        assert len(agent.system_prompt) > 100


class TestHarvesterChat:
    def test_adds_messages_to_history(self):
        mock_client = make_mock_client(
            make_api_response(make_text_block("Nice idea!"))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = HarvesterAgent()
        agent.client = mock_client

        response = agent._chat("Hello")
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"
        assert response == "Nice idea!"

    def test_history_grows_with_multiple_turns(self):
        mock_client = make_mock_client([
            make_api_response(make_text_block("Turn 1")),
            make_api_response(make_text_block("Turn 2")),
        ])
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = HarvesterAgent()
        agent.client = mock_client

        agent._chat("First")
        agent._chat("Second")
        assert len(agent.conversation_history) == 4


class TestHarvesterHarvest:
    def _run_harvest(self, responses, user_inputs):
        """Run harvest() with controlled LLM responses and user inputs."""
        mock_client = make_mock_client(responses)
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = HarvesterAgent()
        agent.client = mock_client

        prints = []
        input_iter = iter(user_inputs)

        return agent.harvest(
            ui_print=lambda t: prints.append(t),
            ui_input=lambda _: next(input_iter),
        ), prints

    def test_returns_idea_document(self):
        """Happy path: agent drafts IdeaDocument, user confirms."""
        responses = [
            make_api_response(make_text_block("Tell me about your idea?")),    # opening
            make_api_response(make_text_block("Interesting! What problem does it solve?")),
            make_api_response(make_text_block(make_idea_doc_response())),       # draft
        ]
        user_inputs = ["My tutoring idea", "Slow classrooms", "yes"]

        result, _ = self._run_harvest(responses, user_inputs)

        assert isinstance(result, IdeaDocument)
        assert result.core_concept == VALID_IDEA_JSON["core_concept"]
        assert result.excitement_level == 8

    def test_user_corrects_before_confirming(self):
        """User requests changes; agent re-drafts; user then confirms."""
        corrected_doc = {**VALID_IDEA_JSON, "excitement_level": 9}
        responses = [
            make_api_response(make_text_block("Tell me about your idea?")),
            make_api_response(make_text_block("Great! Can you tell me more?")),
            make_api_response(make_text_block(make_idea_doc_response())),             # first draft
            make_api_response(make_text_block(make_idea_doc_response(corrected_doc))), # corrected draft
        ]
        user_inputs = [
            "My tutoring idea",
            "More details",
            "The excitement level is wrong, it should be 9",  # correction
            "yes",  # confirm
        ]

        result, _ = self._run_harvest(responses, user_inputs)

        assert result.excitement_level == 9

    def test_prints_opening_question(self):
        responses = [
            make_api_response(make_text_block("What's your idea?")),
            make_api_response(make_text_block(make_idea_doc_response())),
        ]
        user_inputs = ["my idea", "yes"]

        _, prints = self._run_harvest(responses, user_inputs)

        # First non-empty print should be the opening question
        non_empty = [p for p in prints if p.strip()]
        assert "What's your idea?" in non_empty[1]  # index 0 is the "Starting..." header

    def test_skips_empty_user_input(self):
        """Empty user inputs should be ignored and the loop continues."""
        # After 2 empty skips, "My idea" triggers the LLM which returns the
        # draft IdeaDocument immediately; "yes" then confirms it.
        responses = [
            make_api_response(make_text_block("Tell me your idea?")),       # opening
            make_api_response(make_text_block(make_idea_doc_response())),   # draft on first real input
        ]
        user_inputs = ["", "", "My idea", "yes"]

        result, _ = self._run_harvest(responses, user_inputs)
        assert isinstance(result, IdeaDocument)

    def test_idea_document_fields_preserved(self):
        """All fields from the JSON should appear in the returned IdeaDocument."""
        responses = [
            make_api_response(make_text_block("Tell me your idea?")),
            make_api_response(make_text_block(make_idea_doc_response())),
        ]
        user_inputs = ["My tutoring idea", "yes"]

        result, _ = self._run_harvest(responses, user_inputs)

        assert result.constraints == VALID_IDEA_JSON["constraints"]
        assert result.unknowns == VALID_IDEA_JSON["unknowns"]
        assert result.domain_tags == VALID_IDEA_JSON["domain_tags"]
        assert result.raw_quotes == VALID_IDEA_JSON["raw_quotes"]

    def test_confirmation_keywords(self):
        """Different confirmation phrases all end the loop."""
        for keyword in ["yes", "correct", "good", "perfect", "that's right", "looks good", "confirmed"]:
            responses = [
                make_api_response(make_text_block("Your idea?")),
                make_api_response(make_text_block(make_idea_doc_response())),
            ]
            user_inputs = ["My idea", keyword]

            result, _ = self._run_harvest(responses, user_inputs)
            assert isinstance(result, IdeaDocument)
