"""Tests for brainstorm/agents/judge.py — scoring, stopping conditions, quality trend."""

import json
from unittest.mock import patch

import pytest

from brainstorm.agents.judge import JudgeAgent
from brainstorm.schemas import JudgeVerdict
from tests.conftest import make_api_response, make_text_block, make_mock_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_verdict_json(
    round_num: int = 1,
    should_continue: bool = True,
    novelty: float = 0.8,
    diversity: float = 0.75,
    quality: float = 0.70,
    similarity: float = 0.15,
    stop_reason: str | None = None,
    breakout: str | None = None,
    directive: str = "Focus on constraint inversion for round 2",
    top_ids: list | None = None,
) -> str:
    return json.dumps({
        "round": round_num,
        "novelty_score": novelty,
        "diversity_score": diversity,
        "quality_score": quality,
        "similarity_to_previous": similarity,
        "should_continue": should_continue,
        "stop_reason": stop_reason,
        "breakout_idea_id": breakout,
        "directive": directive,
        "top_variant_ids": top_ids or ["v1-variant-0"],
    })


class TestJudgeInit:
    def test_name_and_system_prompt(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            agent = JudgeAgent()
        assert agent.name == "Judge"
        assert "novelty" in agent.system_prompt.lower()
        assert agent.prior_quality_scores == []


class TestJudgeEvaluate:
    def _make_agent(self, response_text: str) -> JudgeAgent:
        mock_client = make_mock_client(
            make_api_response(make_text_block(response_text))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = JudgeAgent()
        agent.client = mock_client
        return agent

    def test_returns_judge_verdict(self, sample_idea_set):
        agent = self._make_agent(make_verdict_json())
        result = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert isinstance(result, JudgeVerdict)

    def test_continues_when_quality_good(self, sample_idea_set):
        agent = self._make_agent(make_verdict_json(should_continue=True, quality=0.75))
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert verdict.should_continue is True

    def test_stops_when_model_says_stop(self, sample_idea_set):
        agent = self._make_agent(
            make_verdict_json(
                should_continue=False,
                stop_reason="Similarity too high",
                similarity=0.85,
            )
        )
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert verdict.should_continue is False
        assert verdict.stop_reason is not None

    def test_quality_tracked_in_history(self, sample_idea_set):
        agent = self._make_agent(make_verdict_json(quality=0.72))
        agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert len(agent.prior_quality_scores) == 1
        assert agent.prior_quality_scores[0] == pytest.approx(0.72)

    def test_quality_trend_stops_on_decline(self, sample_idea_set):
        """If quality drops by >0.1 in round 2+, override should_continue to False."""
        from brainstorm.schemas import IdeaSet
        round2_set = IdeaSet(
            round=2,
            directive_received="go deeper",
            variants=sample_idea_set.variants,
        )

        mock_client = make_mock_client([
            make_api_response(make_text_block(make_verdict_json(round_num=1, quality=0.75, should_continue=True))),
            make_api_response(make_text_block(make_verdict_json(round_num=2, quality=0.55, should_continue=True))),
        ])
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = JudgeAgent()
        agent.client = mock_client

        # Round 1: good quality (current_idea_set.round=1 → override doesn't fire)
        agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        # Round 2: quality drops 0.20 (> threshold), current_idea_set.round=2 → override fires
        verdict2 = agent.evaluate(round2_set, [sample_idea_set, round2_set], max_rounds=3)

        assert verdict2.should_continue is False
        assert "declining" in verdict2.stop_reason.lower()

    def test_quality_trend_continues_if_small_drop(self, sample_idea_set):
        """A small quality drop (<0.1) should NOT trigger the override."""
        mock_client = make_mock_client([
            make_api_response(make_text_block(make_verdict_json(round_num=1, quality=0.75, should_continue=True))),
            make_api_response(make_text_block(make_verdict_json(round_num=2, quality=0.70, should_continue=True))),
        ])
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = JudgeAgent()
        agent.client = mock_client

        agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        verdict2 = agent.evaluate(sample_idea_set, [sample_idea_set, sample_idea_set], max_rounds=3)

        assert verdict2.should_continue is True

    def test_breakout_idea_preserved(self, sample_idea_set):
        agent = self._make_agent(
            make_verdict_json(breakout="v1-variant-0", quality=0.88)
        )
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert verdict.breakout_idea_id == "v1-variant-0"

    def test_directive_non_empty(self, sample_idea_set):
        agent = self._make_agent(make_verdict_json())
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert len(verdict.directive) > 5

    def test_top_variant_ids_preserved(self, sample_idea_set):
        agent = self._make_agent(
            make_verdict_json(top_ids=["v1-variant-0", "v1-variant-1"])
        )
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        assert "v1-variant-0" in verdict.top_variant_ids
        assert "v1-variant-1" in verdict.top_variant_ids

    def test_current_round_in_prompt(self, sample_idea_set):
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_verdict_json(round_num=2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = JudgeAgent()
        agent.client = mock_client

        agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert "max_rounds" in prompt.lower() or "3" in prompt

    def test_prior_sets_comparison_in_prompt(self, sample_idea_set):
        """Previous round variants should appear for similarity comparison."""
        mock_client = make_mock_client(
            make_api_response(make_text_block(make_verdict_json(round_num=2)))
        )
        with patch("brainstorm.agents.base.anthropic.Anthropic", return_value=mock_client):
            agent = JudgeAgent()
        agent.client = mock_client

        # Two rounds in history
        agent.evaluate(
            sample_idea_set,
            [sample_idea_set, sample_idea_set],
            max_rounds=3
        )
        call_messages = mock_client.messages.create.call_args[1]["messages"]
        prompt = call_messages[0]["content"]
        assert sample_idea_set.variants[0].title in prompt

    def test_quality_trend_only_triggers_from_round_2(self, sample_idea_set):
        """The quality decline override should never fire on round 1."""
        # Even a steep initial drop should not stop round 1
        agent = self._make_agent(make_verdict_json(round_num=1, quality=0.2, should_continue=True))
        verdict = agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
        # No prior score to compare against, so should_continue stays True
        assert verdict.should_continue is True

    def test_invalid_json_raises(self, sample_idea_set):
        agent = self._make_agent("not valid json")
        with pytest.raises(ValueError):
            agent.evaluate(sample_idea_set, [sample_idea_set], max_rounds=3)
