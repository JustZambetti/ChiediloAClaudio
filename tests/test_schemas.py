"""Tests for brainstorm/schemas.py — Pydantic model validation and serialization."""

import json
import pytest
from pydantic import ValidationError

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
# IdeaDocument
# ---------------------------------------------------------------------------

class TestIdeaDocument:
    def test_valid_creation(self, sample_idea_doc):
        assert sample_idea_doc.core_concept != ""
        assert 1 <= sample_idea_doc.excitement_level <= 10

    def test_default_version_is_1(self):
        doc = IdeaDocument(
            core_concept="Test",
            problem_solved="A problem",
            target_audience="Someone",
            excitement_level=5,
        )
        assert doc.version == 1

    def test_excitement_level_bounds(self):
        with pytest.raises(ValidationError):
            IdeaDocument(
                core_concept="x", problem_solved="y",
                target_audience="z", excitement_level=0,
            )
        with pytest.raises(ValidationError):
            IdeaDocument(
                core_concept="x", problem_solved="y",
                target_audience="z", excitement_level=11,
            )

    def test_excitement_level_valid_extremes(self):
        for val in [1, 10]:
            doc = IdeaDocument(
                core_concept="x", problem_solved="y",
                target_audience="z", excitement_level=val,
            )
            assert doc.excitement_level == val

    def test_optional_fields_default_to_none_or_empty(self):
        doc = IdeaDocument(
            core_concept="x", problem_solved="y",
            target_audience="z", excitement_level=5,
        )
        assert doc.inspiration_source is None
        assert doc.constraints == []
        assert doc.unknowns == []
        assert doc.domain_tags == []
        assert doc.raw_quotes == []

    def test_serialization_roundtrip(self, sample_idea_doc):
        dumped = json.loads(sample_idea_doc.model_dump_json())
        restored = IdeaDocument(**dumped)
        assert restored == sample_idea_doc

    def test_version_increment(self, sample_idea_doc):
        updated = sample_idea_doc.model_copy(update={"version": 2})
        assert updated.version == 2
        assert sample_idea_doc.version == 1  # original unchanged

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            IdeaDocument()  # missing required fields

    def test_lists_are_mutable_per_instance(self):
        doc1 = IdeaDocument(core_concept="a", problem_solved="b", target_audience="c", excitement_level=5)
        doc2 = IdeaDocument(core_concept="a", problem_solved="b", target_audience="c", excitement_level=5)
        doc1.constraints.append("x")
        assert doc2.constraints == []  # no shared state


# ---------------------------------------------------------------------------
# CritiqueDocument
# ---------------------------------------------------------------------------

class TestCritiqueDocument:
    def test_valid_creation(self, sample_critique):
        assert sample_critique.round == 1
        assert len(sample_critique.strengths) > 0
        assert len(sample_critique.weaknesses) > 0

    def test_feasibility_scores_are_floats(self, sample_critique):
        for axis, score in sample_critique.feasibility_scores.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_feasibility_scores_default_empty(self):
        doc = CritiqueDocument(
            round=1, strengths=[], weaknesses=[],
            assumptions_challenged=[], risks=[], open_questions=[],
        )
        assert doc.feasibility_scores == {}

    def test_serialization_roundtrip(self, sample_critique):
        data = json.loads(sample_critique.model_dump_json())
        restored = CritiqueDocument(**data)
        assert restored == sample_critique


# ---------------------------------------------------------------------------
# ResearchDocument
# ---------------------------------------------------------------------------

class TestResearchDocument:
    def test_valid_creation(self, sample_research):
        assert sample_research.round == 1
        assert len(sample_research.existing_solutions) == 1
        assert sample_research.market_context != ""

    def test_research_finding_fields(self, sample_research):
        finding = sample_research.existing_solutions[0]
        assert finding.title == "Khan Academy"
        assert finding.source_query != ""
        assert finding.relevance != ""

    def test_empty_research_doc(self):
        doc = ResearchDocument(
            round=2,
            existing_solutions=[],
            market_context="Unknown",
            adjacent_ideas=[],
            trend_signals=[],
            key_data_points=[],
        )
        assert doc.round == 2

    def test_serialization_roundtrip(self, sample_research):
        data = json.loads(sample_research.model_dump_json())
        restored = ResearchDocument(**data)
        assert restored == sample_research


# ---------------------------------------------------------------------------
# IdeaSet and IdeaVariant
# ---------------------------------------------------------------------------

class TestIdeaSet:
    def test_valid_creation(self, sample_idea_set):
        assert sample_idea_set.round == 1
        assert len(sample_idea_set.variants) == 3

    def test_variant_ids_are_unique(self, sample_idea_set):
        ids = [v.id for v in sample_idea_set.variants]
        assert len(ids) == len(set(ids))

    def test_optional_variant_fields(self):
        v = IdeaVariant(
            id="v1-test",
            title="Test",
            description="A test variant",
            technique_used="SCAMPER-Reverse",
            diverges_from_original_by="Completely flipped",
        )
        assert v.addresses_critique is None
        assert v.inspired_by_research is None

    def test_serialization_roundtrip(self, sample_idea_set):
        data = json.loads(sample_idea_set.model_dump_json())
        restored = IdeaSet(**data)
        assert restored == sample_idea_set


# ---------------------------------------------------------------------------
# JudgeVerdict
# ---------------------------------------------------------------------------

class TestJudgeVerdict:
    def test_valid_verdict(self, sample_verdict):
        assert 0.0 <= sample_verdict.novelty_score <= 1.0
        assert 0.0 <= sample_verdict.diversity_score <= 1.0
        assert 0.0 <= sample_verdict.quality_score <= 1.0
        assert 0.0 <= sample_verdict.similarity_to_previous <= 1.0

    def test_score_bounds_enforced(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(
                round=1, novelty_score=1.5, diversity_score=0.5,
                quality_score=0.5, similarity_to_previous=0.1,
                should_continue=True, directive="test", top_variant_ids=[],
            )

    def test_stop_reason_optional(self, sample_verdict):
        assert sample_verdict.stop_reason is None  # continuing

    def test_stop_reason_set_when_stopping(self):
        v = JudgeVerdict(
            round=3, novelty_score=0.2, diversity_score=0.3,
            quality_score=0.4, similarity_to_previous=0.8,
            should_continue=False,
            stop_reason="Spinning in place: similarity > 0.7",
            directive="N/A",
            top_variant_ids=["v3-x"],
        )
        assert v.stop_reason is not None
        assert not v.should_continue

    def test_breakout_idea_optional(self, sample_verdict):
        assert sample_verdict.breakout_idea_id is None

    def test_serialization_roundtrip(self, sample_verdict):
        data = json.loads(sample_verdict.model_dump_json())
        restored = JudgeVerdict(**data)
        assert restored == sample_verdict


# ---------------------------------------------------------------------------
# BrainstormReport
# ---------------------------------------------------------------------------

class TestBrainstormReport:
    def test_valid_creation(self, sample_report):
        assert sample_report.total_rounds == 2
        assert len(sample_report.top_directions) == 1

    def test_direction_excitement_score_bounds(self):
        with pytest.raises(ValidationError):
            Direction(
                title="x", description="y", emerged_in_round=1,
                variant_ids=[], open_questions=[], suggested_experiment="z",
                risks=[], excitement_score=1.5,
            )

    def test_to_markdown_contains_title(self, sample_report):
        md = sample_report.to_markdown()
        assert sample_report.idea_title in md

    def test_to_markdown_contains_directions(self, sample_report):
        md = sample_report.to_markdown()
        assert sample_report.top_directions[0].title in md

    def test_to_markdown_contains_next_steps(self, sample_report):
        md = sample_report.to_markdown()
        for step in sample_report.next_steps:
            assert step in md

    def test_to_markdown_contains_evolution_summary(self, sample_report):
        md = sample_report.to_markdown()
        assert sample_report.evolution_summary in md

    def test_to_markdown_convergence_point_shown(self, sample_report):
        md = sample_report.to_markdown()
        assert sample_report.convergence_point in md

    def test_to_markdown_no_convergence_point(self, sample_report):
        report = sample_report.model_copy(update={"convergence_point": None})
        md = report.to_markdown()
        assert "Convergence" not in md

    def test_to_markdown_what_surprised_us(self, sample_report):
        md = sample_report.to_markdown()
        for surprise in sample_report.what_surprised_us:
            assert surprise in md

    def test_serialization_roundtrip(self, sample_report):
        data = json.loads(sample_report.model_dump_json())
        restored = BrainstormReport(**data)
        assert restored == sample_report

    def test_multiple_directions_all_rendered(self):
        report = BrainstormReport(
            idea_title="Test",
            total_rounds=2,
            top_directions=[
                Direction(
                    title=f"Direction {i}",
                    description="Desc",
                    emerged_in_round=1,
                    variant_ids=[],
                    open_questions=["Q?"],
                    suggested_experiment="Exp",
                    risks=["Risk"],
                    excitement_score=0.7,
                )
                for i in range(3)
            ],
            evolution_summary="It evolved.",
            what_surprised_us=[],
            next_steps=["Step 1"],
        )
        md = report.to_markdown()
        for i in range(3):
            assert f"Direction {i}" in md
