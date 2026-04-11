"""Tests for brainstorm/agents/synthesizer.py — cross-round synthesis and report format."""

import json
from unittest.mock import patch

import pytest

from brainstorm.agents.synthesizer import SynthesizerAgent
from brainstorm.schemas import BrainstormReport
from tests.conftest import make_api_response, make_text_block, make_mock_client


# ---------------------------------------------------------------------------
# Sample JSON output
# ---------------------------------------------------------------------------

def make_report_json(total_rounds: int = 2) -> str:
    return json.dumps({
        "idea_title": "Adaptive AI Tutor for Underserved Students",
        "total_rounds": total_rounds,
        "top_directions": [
            {
                "title": "Diagnostic-First Learning Platform",
                "description": "Run a 5-minute diagnostic before each session.",
                "emerged_in_round": 1,
                "variant_ids": ["v1-domain-analogy-healthcare"],
                "open_questions": ["Can the diagnostic be built accurately?"],
                "suggested_experiment": "Build diagnostic quiz, test on 50 students",
                "risks": ["AI diagnostic accuracy", "Teacher adoption resistance"],
                "excitement_score": 0.88,
            },
            {
                "title": "SMS-Based No-App Delivery",
                "description": "Deliver lessons over SMS. No app needed.",
                "emerged_in_round": 1,
                "variant_ids": ["v1-scamper-eliminate"],
                "open_questions": ["SMS delivery costs at scale?"],
                "suggested_experiment": "Run a 2-week SMS pilot with 20 students",
                "risks": ["SMS cost at scale", "Limited interactivity"],
                "excitement_score": 0.72,
            },
        ],
        "evolution_summary": "Started as general adaptive tutor, evolved into diagnosis-first approach.",
        "what_surprised_us": ["SMS could reach 3x more students than app-first"],
        "convergence_point": "Diagnostic-First is the strongest direction",
        "next_steps": ["Build diagnostic prototype", "Partner with 3 pilot schools"],
    })


class TestSynthesizerInit:
    def test_name_and_system_prompt(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = SynthesizerAgent()
        assert agent.name == "Synthesizer"
        assert "cross" in agent.system_prompt.lower() or "round" in agent.system_prompt.lower()


class TestSynthesizerSynthesize:
    def _make_agent(self, response_text: str) -> SynthesizerAgent:
        mock_client = make_mock_client(
            make_api_response(make_text_block(response_text))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = SynthesizerAgent()
        agent.client = mock_client
        return agent

    def test_returns_brainstorm_report(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json())
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )
        assert isinstance(result, BrainstormReport)

    def test_total_rounds_correct(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json(total_rounds=3))
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set] * 3,
            all_verdicts=[sample_verdict] * 3,
        )
        assert result.total_rounds == 3

    def test_top_directions_populated(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json())
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )
        assert len(result.top_directions) == 2

    def test_idea_title_set(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json())
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )
        assert result.idea_title == "Adaptive AI Tutor for Underserved Students"

    def test_evolution_summary_set(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json())
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )
        assert len(result.evolution_summary) > 10

    def test_next_steps_populated(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent(make_report_json())
        result = agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )
        assert len(result.next_steps) > 0

    def test_all_rounds_included_in_prompt(self, sample_idea_doc, sample_idea_set, sample_verdict):
        """The synthesizer should see variants from ALL rounds, not just the last."""
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_report_json(2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = SynthesizerAgent()
        agent.client = mock_client

        # Two rounds of ideas
        from brainstorm.schemas import IdeaSet, IdeaVariant
        round2_set = IdeaSet(
            round=2,
            directive_received="Deeper constraint inversion",
            variants=[
                IdeaVariant(
                    id="v2-unique",
                    title="Round 2 Unique Title",
                    description="A round 2 idea",
                    technique_used="SCAMPER-Combine",
                    diverges_from_original_by="Combines two domains",
                )
            ],
        )

        agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set, round2_set],
            all_verdicts=[sample_verdict, sample_verdict],
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        # Both round 1 and round 2 variants should appear
        assert sample_idea_set.variants[0].title in prompt
        assert "Round 2 Unique Title" in prompt

    def test_judge_scores_included_in_prompt(self, sample_idea_doc, sample_idea_set, sample_verdict):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_report_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = SynthesizerAgent()
        agent.client = mock_client

        agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        # Judge scores should be in the prompt
        assert str(sample_verdict.novelty_score) in prompt or "novelty" in prompt

    def test_original_idea_doc_in_prompt(self, sample_idea_doc, sample_idea_set, sample_verdict):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_report_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = SynthesizerAgent()
        agent.client = mock_client

        agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[sample_verdict],
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert sample_idea_doc.core_concept in prompt

    def test_invalid_json_raises(self, sample_idea_doc, sample_idea_set, sample_verdict):
        agent = self._make_agent("not valid json")
        with pytest.raises(ValueError):
            agent.synthesize(
                idea_doc=sample_idea_doc,
                all_idea_sets=[sample_idea_set],
                all_verdicts=[sample_verdict],
            )

    def test_breakout_idea_highlighted_in_prompt(self, sample_idea_doc, sample_idea_set):
        from brainstorm.schemas import JudgeVerdict
        verdict_with_breakout = JudgeVerdict(
            round=1,
            novelty_score=0.9,
            diversity_score=0.85,
            quality_score=0.92,
            similarity_to_previous=0.1,
            should_continue=False,
            stop_reason="Breakout detected",
            breakout_idea_id="v1-variant-0",
            directive="N/A",
            top_variant_ids=["v1-variant-0"],
        )
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_report_json()))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = SynthesizerAgent()
        agent.client = mock_client

        agent.synthesize(
            idea_doc=sample_idea_doc,
            all_idea_sets=[sample_idea_set],
            all_verdicts=[verdict_with_breakout],
        )

        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert "v1-variant-0" in prompt
