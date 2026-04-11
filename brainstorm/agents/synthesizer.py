"""
Agent 6: Synthesizer

Receives the full history of all idea sets across all rounds,
the original IdeaDocument, and produces the final BrainstormReport.

Key insight: the best ideas may have appeared in round 1, not round 3.
The Synthesizer looks across ALL rounds, not just the last one.
"""

from __future__ import annotations

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import (
    IdeaDocument,
    IdeaSet,
    JudgeVerdict,
    BrainstormReport,
)

SYSTEM_PROMPT = """\
You are a synthesis expert. You have witnessed a full creative brainstorming \
session and must now distill it into a clear, actionable report.

## Your job

1. **Identify the 3-5 strongest directions** — look across ALL rounds, not \
   just the last one. The best idea might have emerged in round 1 and evolved, \
   or appeared as a surprise in round 2.

2. **Write the evolution narrative** — how did the idea space change across \
   rounds? What did the critique and research reveal that redirected things?

3. **Surface what surprised us** — unexpected research findings, counterintuitive \
   critiques, analogies that opened new doors.

4. **Name the convergence point** — if one direction clearly stands above the \
   others, say so directly. Don't hedge if the answer is clear.

5. **Write next steps** — concrete, prioritized, actionable. Not "explore X" \
   but "build a 2-page landing page for direction A and run 50 user interviews \
   in the next 2 weeks."

## Selecting top directions
For each direction, group related variants across rounds that converged on the \
same core insight. A direction may evolve significantly — that's fine, follow \
the thread.

For each direction report:
  - excitement_score: your assessment of potential (0.0–1.0)
  - emerged_in_round: the round where the seed idea first appeared
  - open_questions: specific questions that MUST be answered before pursuing
  - suggested_experiment: the simplest possible test to validate this direction
  - risks: the 2-3 most plausible ways this fails

## Output format
Return ONLY valid JSON:

{
  "idea_title": "...",
  "total_rounds": <int>,
  "top_directions": [
    {
      "title": "...",
      "description": "2-3 sentence pitch",
      "emerged_in_round": <int>,
      "variant_ids": ["v1-tag", "v2-tag"],
      "open_questions": ["...", "..."],
      "suggested_experiment": "...",
      "risks": ["...", "..."],
      "excitement_score": 0.0
    }
  ],
  "evolution_summary": "...",
  "what_surprised_us": ["...", "..."],
  "convergence_point": "... or null",
  "next_steps": ["...", "..."]
}
"""


class SynthesizerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Synthesizer", system_prompt=SYSTEM_PROMPT)

    def synthesize(
        self,
        idea_doc: IdeaDocument,
        all_idea_sets: list[IdeaSet],
        all_verdicts: list[JudgeVerdict],
    ) -> BrainstormReport:
        context_parts = [
            "## Original Idea (the seed)",
            idea_doc.model_dump_json(indent=2),
            f"\n## Total rounds: {len(all_idea_sets)}",
            "\n## All idea variants across all rounds",
        ]

        for idea_set in all_idea_sets:
            context_parts.append(f"\n### Round {idea_set.round} — directive: \"{idea_set.directive_received}\"")
            for v in idea_set.variants:
                lines = [
                    f"\n**[{v.id}] {v.title}**",
                    v.description,
                    f"Technique: {v.technique_used}",
                    f"Diverges by: {v.diverges_from_original_by}",
                ]
                if v.addresses_critique:
                    lines.append(f"Addresses: {v.addresses_critique}")
                if v.inspired_by_research:
                    lines.append(f"Inspired by: {v.inspired_by_research}")
                context_parts.append("\n".join(lines))

        context_parts.append("\n## Judge scores per round")
        for verdict in all_verdicts:
            context_parts.append(
                f"Round {verdict.round}: novelty={verdict.novelty_score:.2f}, "
                f"diversity={verdict.diversity_score:.2f}, "
                f"quality={verdict.quality_score:.2f}"
            )
            if verdict.breakout_idea_id:
                context_parts.append(
                    f"  → Breakout idea: {verdict.breakout_idea_id}"
                )

        context_parts.append(
            "\nSynthesize the full session into a BrainstormReport JSON."
        )

        messages = [{"role": "user", "content": "\n".join(context_parts)}]
        raw = self.run(messages, max_tokens=6000)

        data = self.parse_json(raw)
        return BrainstormReport(**data)
