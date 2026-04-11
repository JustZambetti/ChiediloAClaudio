"""
Pipeline runner — the divergence loop + user checkpoints.

Flow:
  1. Harvester interviews user → IdeaDocument
  2. Divergence loop (up to max_rounds):
       a. Critic + Researcher run in parallel (ThreadPoolExecutor)
       b. Expander generates IdeaSet
       c. Judge evaluates → continue/stop + directive
       d. After round 1: light pulse-check with user
       e. After round 2: full steering checkpoint
  3. Synthesizer produces BrainstormReport
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from brainstorm.agents.critic import CriticAgent
from brainstorm.agents.expander import ExpanderAgent
from brainstorm.agents.harvester import HarvesterAgent
from brainstorm.agents.judge import JudgeAgent
from brainstorm.agents.researcher import ResearcherAgent
from brainstorm.agents.synthesizer import SynthesizerAgent
from brainstorm.schemas import (
    BrainstormReport,
    IdeaDocument,
    IdeaSet,
    JudgeVerdict,
)

# Default directive for round 1 (no Judge has run yet)
INITIAL_DIRECTIVE = (
    "Explore the idea space broadly. Apply every SCAMPER verb once. "
    "Try at least one domain analogy and one constraint inversion. "
    "Prioritise variety over polish."
)


class BrainstormPipeline:
    def __init__(
        self,
        max_rounds: int = 3,
        ui_print: Callable[[str], None] | None = None,
        ui_input: Callable[[str], str] | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.ui_print = ui_print or print
        self.ui_input = ui_input or input

        # Instantiate agents (system prompts cached after first call)
        self.harvester = HarvesterAgent()
        self.critic = CriticAgent()
        self.researcher = ResearcherAgent()
        self.expander = ExpanderAgent()
        self.judge = JudgeAgent()
        self.synthesizer = SynthesizerAgent()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> BrainstormReport:
        # ── Phase 1: Harvest ─────────────────────────────────────────────
        idea_doc = self.harvester.harvest(
            ui_print=self.ui_print,
            ui_input=self.ui_input,
        )

        # ── Phase 2: Divergence loop ─────────────────────────────────────
        all_idea_sets: list[IdeaSet] = []
        all_verdicts: list[JudgeVerdict] = []
        directive = INITIAL_DIRECTIVE

        for round_num in range(1, self.max_rounds + 1):
            self.ui_print(
                f"\n{'─'*60}\n"
                f"  DIVERGENCE ROUND {round_num} / {self.max_rounds}\n"
                f"{'─'*60}"
            )

            # ── a. Critic + Researcher in parallel ────────────────────
            self.ui_print(
                "  Running Critic and Researcher in parallel..."
            )
            t0 = time.time()

            critique, research = self._parallel_critique_and_research(
                idea_doc, round_num, all_idea_sets
            )

            self.ui_print(
                f"  Done in {time.time() - t0:.1f}s"
            )

            # ── b. Expander ───────────────────────────────────────────
            self.ui_print("  Expander generating idea variants...")
            idea_set = self.expander.expand(
                idea_doc=idea_doc,
                critique=critique,
                research=research,
                round_num=round_num,
                directive=directive,
                prior_idea_sets=all_idea_sets if all_idea_sets else None,
            )
            all_idea_sets.append(idea_set)

            self.ui_print(
                f"  Generated {len(idea_set.variants)} variants."
            )
            self._print_variants(idea_set)

            # ── c. Judge ─────────────────────────────────────────────
            self.ui_print("  Judge evaluating...")
            verdict = self.judge.evaluate(
                current_idea_set=idea_set,
                all_idea_sets=all_idea_sets,
                max_rounds=self.max_rounds,
            )
            all_verdicts.append(verdict)

            self.ui_print(
                f"  Scores — novelty: {verdict.novelty_score:.2f}  "
                f"diversity: {verdict.diversity_score:.2f}  "
                f"quality: {verdict.quality_score:.2f}  "
                f"similarity: {verdict.similarity_to_previous:.2f}"
            )

            if verdict.breakout_idea_id:
                self.ui_print(
                    f"  ★ Breakout idea detected: {verdict.breakout_idea_id}"
                )

            # ── d. User checkpoints ───────────────────────────────────
            if round_num == 1:
                steered = self._pulse_check(idea_set)
                if steered:
                    directive = (
                        f"User steering: {steered}. "
                        f"Also: {verdict.directive}"
                    )
                else:
                    directive = verdict.directive

            elif round_num == 2:
                steered = self._steering_checkpoint(idea_set, verdict)
                if steered:
                    directive = (
                        f"User steering: {steered}. "
                        f"Also: {verdict.directive}"
                    )
                else:
                    directive = verdict.directive
            else:
                directive = verdict.directive

            # ── e. Stop? ──────────────────────────────────────────────
            if not verdict.should_continue:
                self.ui_print(
                    f"\n  Loop converging: {verdict.stop_reason or 'quality threshold reached'}"
                )
                break

        # ── Phase 3: Synthesize ──────────────────────────────────────────
        self.ui_print(
            f"\n{'─'*60}\n  SYNTHESIZING final report...\n{'─'*60}"
        )
        report = self.synthesizer.synthesize(
            idea_doc=idea_doc,
            all_idea_sets=all_idea_sets,
            all_verdicts=all_verdicts,
        )

        return report

    # ------------------------------------------------------------------
    # Parallel execution helpers
    # ------------------------------------------------------------------

    def _parallel_critique_and_research(
        self,
        idea_doc: IdeaDocument,
        round_num: int,
        prior_idea_sets: list[IdeaSet],
    ):
        """Run Critic and Researcher concurrently via threads."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_critique = executor.submit(
                self.critic.critique,
                idea_doc,
                round_num,
                prior_idea_sets if prior_idea_sets else None,
            )
            future_research = executor.submit(
                self.researcher.research,
                idea_doc,
                round_num,
            )

            critique = future_critique.result()
            research = future_research.result()

        return critique, research

    # ------------------------------------------------------------------
    # User checkpoint helpers
    # ------------------------------------------------------------------

    def _pulse_check(self, idea_set: IdeaSet) -> str | None:
        """
        Round 1 lightweight check: "Are we in the right space?"
        Just a yes/no — no idea selection.
        """
        self.ui_print(
            "\n┌─ PULSE CHECK (Round 1) ──────────────────────────────────┐"
        )
        self.ui_print(
            "│ Are we exploring the right territory?                    │\n"
            "│ Say 'yes' to continue, or describe what's missing.       │"
        )
        self.ui_print(
            "└───────────────────────────────────────────────────────────┘"
        )

        response = self.ui_input("You: ").strip()
        if not response:
            return None

        if any(kw in response.lower() for kw in ["yes", "good", "right", "correct", "fine"]):
            return None

        return response  # treat as steering directive

    def _steering_checkpoint(
        self,
        idea_set: IdeaSet,
        verdict: JudgeVerdict,
    ) -> str | None:
        """
        Round 2 full checkpoint: show top variants, let user pick direction.
        """
        self.ui_print(
            "\n┌─ STEERING CHECKPOINT (Round 2) ──────────────────────────┐"
        )
        self.ui_print("│ Top ideas so far — which direction excites you most?     │")
        self.ui_print("└───────────────────────────────────────────────────────────┘\n")

        # Show top variants from Judge
        top_ids = set(verdict.top_variant_ids)
        shown = 0
        for v in idea_set.variants:
            if v.id in top_ids or shown < 3:
                self.ui_print(f"  [{v.id}] {v.title}")
                self.ui_print(f"        {v.description}\n")
                shown += 1

        self.ui_print(
            "Options:\n"
            "  - Type an idea ID to push that direction harder\n"
            "  - Describe a direction you'd like to explore\n"
            "  - Say 'continue' to let the Judge steer automatically\n"
        )

        response = self.ui_input("You: ").strip()
        if not response or response.lower() in ["continue", "auto", "skip"]:
            return None

        return response

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_variants(self, idea_set: IdeaSet) -> None:
        self.ui_print("")
        for v in idea_set.variants:
            self.ui_print(f"  [{v.id}] {v.title}")
            self.ui_print(f"         Technique: {v.technique_used}")
        self.ui_print("")
