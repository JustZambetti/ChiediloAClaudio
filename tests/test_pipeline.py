"""
Integration tests for brainstorm/pipeline/runner.py.

All agents are mocked at the method level — we test that the pipeline
orchestrates them correctly: right call order, parallel execution,
user checkpoints, loop control, and final synthesis.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch, call

import pytest

from brainstorm.pipeline.runner import BrainstormPipeline, INITIAL_DIRECTIVE
from brainstorm.schemas import (
    BrainstormReport,
    CritiqueDocument,
    Direction,
    IdeaDocument,
    IdeaSet,
    IdeaVariant,
    JudgeVerdict,
    ResearchDocument,
    ResearchFinding,
)


# ---------------------------------------------------------------------------
# Fixtures — pipeline-level objects
# ---------------------------------------------------------------------------

@pytest.fixture
def idea_doc():
    return IdeaDocument(
        core_concept="AI tutoring app",
        problem_solved="Fixed classroom pace",
        target_audience="High school students",
        excitement_level=8,
        domain_tags=["education", "ai"],
    )


@pytest.fixture
def critique():
    return CritiqueDocument(
        round=1,
        strengths=["Clear problem"],
        weaknesses=["No revenue model"],
        assumptions_challenged=["Students will self-select"],
        risks=["Competition"],
        open_questions=["How to acquire users?"],
        feasibility_scores={"technical": 0.7, "financial": 0.5, "social": 0.6, "timing": 0.8},
    )


@pytest.fixture
def research():
    return ResearchDocument(
        round=1,
        existing_solutions=[
            ResearchFinding(title="Khan Academy", summary="Free platform", source_query="q", relevance="competitor")
        ],
        market_context="$340B market",
        adjacent_ideas=[],
        trend_signals=["Growing"],
        key_data_points=["$50/hr tutor"],
    )


@pytest.fixture
def idea_set():
    return IdeaSet(
        round=1,
        directive_received=INITIAL_DIRECTIVE,
        variants=[
            IdeaVariant(
                id="v1-scamper",
                title="SMS Tutor",
                description="Deliver via SMS",
                technique_used="SCAMPER-Eliminate",
                diverges_from_original_by="No app needed",
            )
        ],
    )


@pytest.fixture
def verdict_continue():
    return JudgeVerdict(
        round=1,
        novelty_score=0.8,
        diversity_score=0.75,
        quality_score=0.70,
        similarity_to_previous=0.15,
        should_continue=True,
        directive="Go deeper on constraint inversion",
        top_variant_ids=["v1-scamper"],
    )


@pytest.fixture
def verdict_stop():
    return JudgeVerdict(
        round=2,
        novelty_score=0.5,
        diversity_score=0.4,
        quality_score=0.60,
        similarity_to_previous=0.75,
        should_continue=False,
        stop_reason="Similarity too high",
        directive="N/A",
        top_variant_ids=["v1-scamper"],
    )


@pytest.fixture
def final_report():
    return BrainstormReport(
        idea_title="AI Tutor",
        total_rounds=2,
        top_directions=[
            Direction(
                title="SMS-Based Delivery",
                description="Deliver via SMS",
                emerged_in_round=1,
                variant_ids=["v1-scamper"],
                open_questions=["Scale cost?"],
                suggested_experiment="Pilot with 20 students",
                risks=["SMS cost"],
                excitement_score=0.80,
            )
        ],
        evolution_summary="Evolved from general app to SMS delivery",
        what_surprised_us=["SMS reaches 3x more students"],
        convergence_point="SMS-Based Delivery wins",
        next_steps=["Build SMS pilot"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline(
    max_rounds: int = 2,
    ui_inputs: list[str] | None = None,
) -> tuple[BrainstormPipeline, list[str]]:
    """Create a BrainstormPipeline with mocked UI."""
    prints = []
    input_iter = iter(ui_inputs or [])
    pipeline = BrainstormPipeline(
        max_rounds=max_rounds,
        ui_print=lambda t: prints.append(t),
        ui_input=lambda _: next(input_iter, "continue"),
    )
    return pipeline, prints


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_agents_instantiated(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline()
        assert pipeline.harvester is not None
        assert pipeline.critic is not None
        assert pipeline.researcher is not None
        assert pipeline.expander is not None
        assert pipeline.judge is not None
        assert pipeline.synthesizer is not None

    def test_max_rounds_stored(self):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=4)
        assert pipeline.max_rounds == 4


# ---------------------------------------------------------------------------
# Full pipeline run (1 round, stops immediately)
# ---------------------------------------------------------------------------

class TestPipelineRunSingleRound:
    def test_returns_brainstorm_report(
        self, idea_doc, critique, research, idea_set, verdict_stop, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=1, ui_inputs=["yes"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_stop)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        result = pipeline.run()
        assert isinstance(result, BrainstormReport)

    def test_synthesizer_called_with_all_data(
        self, idea_doc, critique, research, idea_set, verdict_stop, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=1, ui_inputs=["yes"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_stop)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        pipeline.synthesizer.synthesize.assert_called_once_with(
            idea_doc=idea_doc,
            all_idea_sets=[idea_set],
            all_verdicts=[verdict_stop],
        )

    def test_harvester_called_once(
        self, idea_doc, critique, research, idea_set, verdict_stop, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=1, ui_inputs=["yes"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_stop)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()
        pipeline.harvester.harvest.assert_called_once()


# ---------------------------------------------------------------------------
# Multi-round loop control
# ---------------------------------------------------------------------------

class TestDivergenceLoop:
    def test_loop_runs_max_rounds_if_judge_always_continues(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """If Judge always says continue, loop runs exactly max_rounds times."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=3, ui_inputs=["yes", "continue", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        # Always continue
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        assert pipeline.expander.expand.call_count == 3
        assert pipeline.judge.evaluate.call_count == 3

    def test_loop_stops_early_when_judge_says_stop(
        self, idea_doc, critique, research, idea_set, verdict_continue, verdict_stop, final_report
    ):
        """Judge says stop after round 2 — expander called only twice."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=3, ui_inputs=["yes", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        # Continue round 1, stop at round 2
        pipeline.judge.evaluate = MagicMock(side_effect=[verdict_continue, verdict_stop])
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        assert pipeline.expander.expand.call_count == 2

    def test_critic_and_researcher_called_each_round(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=2, ui_inputs=["yes", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        assert pipeline.critic.critique.call_count == 2
        assert pipeline.researcher.research.call_count == 2

    def test_expander_receives_judge_directive(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """After round 1, Expander must receive the Judge's directive."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=2, ui_inputs=["yes", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        # Second expander call should use the judge's directive
        second_call_kwargs = pipeline.expander.expand.call_args_list[1][1]
        assert verdict_continue.directive in second_call_kwargs["directive"]

    def test_prior_idea_sets_accumulated(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """Each Expander call should receive all prior idea sets."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=3, ui_inputs=["yes", "continue", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        calls = pipeline.expander.expand.call_args_list
        # Round 1: no prior sets
        assert calls[0][1]["prior_idea_sets"] is None
        # Round 2: 1 prior set
        assert len(calls[1][1]["prior_idea_sets"]) == 1
        # Round 3: 2 prior sets
        assert len(calls[2][1]["prior_idea_sets"]) == 2


# ---------------------------------------------------------------------------
# User checkpoints
# ---------------------------------------------------------------------------

class TestUserCheckpoints:
    def test_pulse_check_after_round_1_continue(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """User says 'yes' at pulse check — pipeline continues unchanged."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=2, ui_inputs=["yes", "continue"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        result = pipeline.run()
        assert isinstance(result, BrainstormReport)

    def test_pulse_check_with_steering_modifies_directive(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """User provides steering text at pulse check — it's prepended to directive."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(
                max_rounds=2,
                ui_inputs=["focus more on hardware solutions", "continue"],
            )

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        round2_kwargs = pipeline.expander.expand.call_args_list[1][1]
        assert "hardware" in round2_kwargs["directive"].lower()

    def test_steering_checkpoint_after_round_2(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """Full steering checkpoint fires after round 2."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, prints = make_pipeline(
                max_rounds=3,
                ui_inputs=["yes", "focus on SMS delivery", "continue"],
            )

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        checkpoint_prints = [p for p in prints if "STEERING" in p]
        assert len(checkpoint_prints) == 1  # fired once after round 2

    def test_no_checkpoint_for_round_3_and_beyond(
        self, idea_doc, critique, research, idea_set, verdict_continue, final_report
    ):
        """Round 3 has no user checkpoint — runs fully automatically."""
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            # Only 2 user inputs (round 1 pulse + round 2 steering)
            # If round 3 tries to get input, StopIteration would be raised
            pipeline, _ = make_pipeline(
                max_rounds=3,
                ui_inputs=["yes", "continue"],
            )

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_continue)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        # Should not raise StopIteration
        result = pipeline.run()
        assert isinstance(result, BrainstormReport)


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_critic_and_researcher_called_per_round(
        self, idea_doc, critique, research, idea_set, verdict_stop, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=1, ui_inputs=["yes"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_stop)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        # Round 1: no prior idea sets yet — runner passes None (not [])
        pipeline.critic.critique.assert_called_once_with(idea_doc, 1, None)
        pipeline.researcher.research.assert_called_once_with(idea_doc, 1)

    def test_parallel_results_passed_to_expander(
        self, idea_doc, critique, research, idea_set, verdict_stop, final_report
    ):
        with patch("brainstorm.agents.base.anthropic.Anthropic"):
            pipeline, _ = make_pipeline(max_rounds=1, ui_inputs=["yes"])

        pipeline.harvester.harvest = MagicMock(return_value=idea_doc)
        pipeline.critic.critique = MagicMock(return_value=critique)
        pipeline.researcher.research = MagicMock(return_value=research)
        pipeline.expander.expand = MagicMock(return_value=idea_set)
        pipeline.judge.evaluate = MagicMock(return_value=verdict_stop)
        pipeline.synthesizer.synthesize = MagicMock(return_value=final_report)

        pipeline.run()

        expand_kwargs = pipeline.expander.expand.call_args[1]
        assert expand_kwargs["critique"] is critique
        assert expand_kwargs["research"] is research
