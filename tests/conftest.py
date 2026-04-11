"""
Shared fixtures and mock utilities for the test suite.

All Anthropic API calls are mocked — tests never hit the real API.
"""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest

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
# Mock builder helpers
# ---------------------------------------------------------------------------

def make_text_block(text: str) -> MagicMock:
    """Simulate an Anthropic text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    block.model_dump.return_value = {"type": "text", "text": text}
    return block


def make_tool_use_block(
    tool_id: str, name: str, input_data: dict
) -> MagicMock:
    """Simulate an Anthropic tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    block.model_dump.return_value = {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": input_data,
    }
    return block


def make_api_response(*blocks) -> MagicMock:
    """Simulate an Anthropic Messages API response."""
    response = MagicMock()
    response.content = list(blocks)
    return response


def make_mock_client(responses) -> MagicMock:
    """
    Build a mock Anthropic client whose messages.create() returns
    each response in `responses` in sequence (side_effect list).
    """
    client = MagicMock()
    if isinstance(responses, list):
        client.messages.create.side_effect = responses
    else:
        client.messages.create.return_value = responses
    return client


# ---------------------------------------------------------------------------
# Sample document fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_idea_doc() -> IdeaDocument:
    return IdeaDocument(
        version=1,
        core_concept="An AI-powered tutoring app that adapts to each student's learning pace",
        problem_solved="Students learn at different speeds but classrooms move at one fixed pace",
        target_audience="High school students aged 14-18 struggling with math and science",
        inspiration_source="Watching my younger sibling fall behind in school",
        constraints=["Must work offline", "No subscription fees over $10/month"],
        unknowns=["Whether schools will adopt it or it needs to go direct-to-consumer"],
        excitement_level=8,
        domain_tags=["education", "ai", "mobile"],
        raw_quotes=[
            "I want every kid to feel like they have a private tutor",
            "The app should know when you're stuck before you do",
        ],
    )


@pytest.fixture
def sample_critique(sample_idea_doc) -> CritiqueDocument:
    return CritiqueDocument(
        round=1,
        strengths=["Clear problem", "Passionate founder", "Large addressable market"],
        weaknesses=["No clear revenue model", "Relies on AI accuracy that may not exist yet"],
        assumptions_challenged=[
            "Assumes students will engage voluntarily: many disengage from homework entirely",
            "Assumes AI can diagnose learning gaps better than teachers: unproven",
        ],
        risks=["Regulatory barriers in EdTech", "Competition from Khan Academy, Duolingo"],
        open_questions=[
            "How do you acquire the first 1000 students?",
            "What does the teacher think of this tool?",
        ],
        feasibility_scores={
            "technical": 0.65,
            "financial": 0.50,
            "social": 0.70,
            "timing": 0.80,
        },
    )


@pytest.fixture
def sample_research() -> ResearchDocument:
    return ResearchDocument(
        round=1,
        existing_solutions=[
            ResearchFinding(
                title="Khan Academy",
                summary="Free platform with adaptive exercises",
                source_query="adaptive learning app students",
                relevance="Direct competitor but lacks true AI personalization",
            )
        ],
        market_context="Global EdTech market valued at $340B, growing 16% YoY",
        adjacent_ideas=[
            ResearchFinding(
                title="Duolingo's streak mechanic",
                summary="Gamified daily habit loop that doubled retention",
                source_query="gamification education retention",
                relevance="Habit loop design could dramatically improve engagement",
            )
        ],
        trend_signals=["Post-COVID EdTech investment boom", "AI tutoring VC interest up 3x"],
        key_data_points=["1 in 3 students lacks access to private tutoring", "Average tutor costs $50/hr"],
    )


@pytest.fixture
def sample_idea_set() -> IdeaSet:
    return IdeaSet(
        round=1,
        directive_received="Explore broadly using all SCAMPER verbs",
        variants=[
            IdeaVariant(
                id="v1-scamper-eliminate",
                title="No-App Tutor: SMS-Based Learning",
                description="Deliver adaptive lessons over SMS. No app download, works on any phone.",
                technique_used="SCAMPER-Eliminate",
                diverges_from_original_by="Eliminates the app entirely — reaches students without smartphones",
                addresses_critique="Relies on students having modern smartphones",
                inspired_by_research=None,
            ),
            IdeaVariant(
                id="v1-constraint-inversion",
                title="Offline-First Peer Mesh Network",
                description="Students teach each other, AI orchestrates matching and curriculum.",
                technique_used="Constraint-Inversion",
                diverges_from_original_by="Turns the offline constraint into a P2P feature",
                addresses_critique=None,
                inspired_by_research="Duolingo's streak mechanic",
            ),
            IdeaVariant(
                id="v1-domain-analogy-healthcare",
                title="Diagnostic-First Learning",
                description="Like a doctor running tests before prescribing, run a 5-minute diagnostic before any lesson.",
                technique_used="Domain-Analogy:healthcare",
                diverges_from_original_by="Reframes teaching as diagnosis + treatment",
                addresses_critique="AI can diagnose learning gaps better than teachers: unproven",
                inspired_by_research=None,
            ),
        ],
    )


@pytest.fixture
def sample_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        round=1,
        novelty_score=0.82,
        diversity_score=0.75,
        quality_score=0.70,
        similarity_to_previous=0.15,
        should_continue=True,
        stop_reason=None,
        breakout_idea_id=None,
        directive="Focus on constraint inversion — push the offline/no-subscription limits further",
        top_variant_ids=["v1-domain-analogy-healthcare", "v1-constraint-inversion"],
    )


@pytest.fixture
def sample_report() -> BrainstormReport:
    return BrainstormReport(
        idea_title="Adaptive AI Tutor for Underserved Students",
        total_rounds=2,
        top_directions=[
            Direction(
                title="Diagnostic-First Learning Platform",
                description="Run a 5-minute diagnostic before each session. Prescribe personalized paths.",
                emerged_in_round=1,
                variant_ids=["v1-domain-analogy-healthcare"],
                open_questions=["Can the diagnostic be built accurately with available data?"],
                suggested_experiment="Build the diagnostic quiz and test on 50 students",
                risks=["AI diagnostic accuracy", "Teacher adoption resistance"],
                excitement_score=0.88,
            )
        ],
        evolution_summary="The idea started as a general adaptive tutor and evolved into a diagnosis-first approach.",
        what_surprised_us=["SMS delivery could reach 3x more students than app-first approach"],
        convergence_point="Diagnostic-First is the strongest direction by quality score",
        next_steps=["Build diagnostic prototype", "Partner with 3 pilot schools"],
    )


# ---------------------------------------------------------------------------
# Anthropic client patcher context manager
# ---------------------------------------------------------------------------

class MockAnthropicPatcher:
    """
    Context manager / callable that patches brainstorm.agents.base.anthropic.Anthropic
    and returns a pre-configured mock client.
    """
    def __init__(self, responses):
        self.mock_client = make_mock_client(responses)
        self._patcher = patch(
            "brainstorm.agents.base.anthropic.Anthropic",
            return_value=self.mock_client,
        )

    def __enter__(self):
        self._patcher.start()
        return self.mock_client

    def __exit__(self, *args):
        self._patcher.stop()


@pytest.fixture
def patch_anthropic():
    """
    Fixture factory: returns a helper that patches the Anthropic client.

    Usage in tests:
        def test_something(patch_anthropic):
            with patch_anthropic([response1, response2]):
                agent = SomeAgent()
                result = agent.some_method(...)
    """
    def _factory(responses):
        return MockAnthropicPatcher(responses)
    return _factory
