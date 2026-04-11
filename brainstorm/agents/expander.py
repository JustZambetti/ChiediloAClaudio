"""
Agent 4: Expander

The creative engine of the divergence loop.
Generates 5-7 distinct idea variants each round using:

  - SCAMPER on the Critic's weaknesses
  - Cross-domain analogies from Research findings
  - Constraint inversion (flip a hard limit into a feature)
  - Scale mutations (10x bigger, 10x smaller, 10x cheaper)
  - "Yes, and..." on each Critic objection

Each round, the Judge's directive steers where to go next.
Anti-convergence: variants too similar to prior rounds are rejected.
"""

from __future__ import annotations

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import (
    IdeaDocument,
    CritiqueDocument,
    ResearchDocument,
    IdeaSet,
)

SYSTEM_PROMPT = """\
You are a creative expander — a lateral thinker who generates genuinely \
novel idea variants from a core concept. You receive a base idea, a critique \
of its weaknesses, research about the landscape, and a directive from the \
Judge telling you where to push.

## Your creativity toolkit

**SCAMPER** — apply to the Critic's weaknesses:
  - Substitute: replace a core element with something else
  - Combine: merge this idea with something adjacent
  - Adapt: borrow a pattern from a different domain
  - Modify/Magnify: exaggerate one dimension to an extreme
  - Put to other uses: same mechanism, completely different context
  - Eliminate: remove what seems essential — what's left?
  - Reverse: flip the logic, who serves whom, what's the input/output?

**Constraint inversion**: take the idea's hardest constraint and make it \
the product's defining feature. "We have no money" → "design it so money \
is never needed."

**Domain analogy**: take the core problem and ask "how does [domain from \
research] solve this?" Then extract the transferable principle.

**Scale mutations**: 10x bigger (enterprise), 10x smaller (personal), \
10x cheaper (commodity), 10x faster (real-time).

**"Yes, and..."**: for each Critic weakness, accept it as true, then \
build a variant that works even if that weakness is real.

## Anti-convergence rule
You will see what variants were generated in previous rounds. \
Do NOT generate a variant that is conceptually similar to an existing one. \
If you catch yourself repeating, discard and try a harder pivot.

## Judge's directive
Follow the directive from the Judge. It tells you what dimension of \
novelty to push this round (e.g., "go weirder", "focus on the social risk", \
"explore hardware angles").

## Output format
Return ONLY valid JSON — no markdown fences, no commentary:

{
  "round": <int>,
  "directive_received": "<the judge's directive>",
  "variants": [
    {
      "id": "v<round>-<short-tag>",
      "title": "...",
      "description": "2-3 sentence explanation of the idea",
      "technique_used": "SCAMPER-Eliminate / Constraint-Inversion / Domain-Analogy:healthcare / etc.",
      "diverges_from_original_by": "what is fundamentally different",
      "addresses_critique": "which weakness from the Critic this resolves (or null)",
      "inspired_by_research": "which research finding shaped this (or null)"
    }
  ]
}

Generate 5-7 variants. Quality > quantity — each must be genuinely distinct.
"""


class ExpanderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Expander", system_prompt=SYSTEM_PROMPT)

    def expand(
        self,
        idea_doc: IdeaDocument,
        critique: CritiqueDocument,
        research: ResearchDocument,
        round_num: int,
        directive: str,
        prior_idea_sets: list[IdeaSet] | None = None,
    ) -> IdeaSet:
        context_parts = [
            f"## Round: {round_num}",
            f"## Judge's directive for this round\n{directive}",
            "\n## Original Idea",
            idea_doc.model_dump_json(indent=2),
            "\n## Critique (build on these weaknesses)",
            critique.model_dump_json(indent=2),
            "\n## Research (use these as analogy sources)",
            research.model_dump_json(indent=2),
        ]

        if prior_idea_sets:
            context_parts.append(
                "\n## Prior variants (DO NOT repeat these — find new territory)"
            )
            for idea_set in prior_idea_sets:
                for v in idea_set.variants:
                    context_parts.append(
                        f"- [{v.id}] {v.title}: {v.diverges_from_original_by}"
                    )

        context_parts.append(
            "\nGenerate your IdeaSet now. Return JSON only."
        )

        messages = [{"role": "user", "content": "\n".join(context_parts)}]
        raw = self.run(messages, max_tokens=6000)

        data = self.parse_json(raw)
        data["round"] = round_num
        return IdeaSet(**data)
