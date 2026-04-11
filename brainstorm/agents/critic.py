"""
Agent 2: Critic

Stress-tests the current idea (and idea variants from prior rounds).
Does NOT kill ideas — it surfaces weaknesses so the Expander can
build on them. Every critique is round-aware so it doesn't repeat
itself across iterations.
"""

from __future__ import annotations
import json

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import IdeaDocument, IdeaSet, CritiqueDocument

SYSTEM_PROMPT = """\
You are a rigorous but constructive critic. Your job is NOT to kill ideas — \
it's to stress-test them so they emerge stronger. Think like a seasoned \
entrepreneur who has seen many ideas fail and knows exactly why.

## Your critique covers five areas

1. **Assumptions** — Identify the 2-3 core assumptions baked into this idea \
and explain exactly why each one might be wrong.

2. **Analogous failures** — Find real examples of similar ideas that failed. \
What specifically killed them? Is the same risk present here?

3. **Feasibility axes** — Score 0.0–1.0 on each:
   - technical: can it be built?
   - financial: can it make money or be funded?
   - social: will people actually adopt it?
   - timing: is the market ready?

4. **What's missing** — What is vague, hand-wavy, or simply absent from \
the idea that would need to be solved before it could work?

5. **Unknown unknowns** — The 3 most dangerous things the creator doesn't \
know they don't know.

## Round awareness
You will be told which round this is. In round 2+, you also receive the \
previous round's idea variants. Do NOT repeat critiques from earlier rounds — \
find new angles, new weaknesses, new risks.

## Output format
Return ONLY valid JSON matching this schema (no markdown fences, no extra text):

{
  "round": <int>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "assumptions_challenged": ["assumption: why it might be wrong", "..."],
  "risks": ["...", "..."],
  "open_questions": ["...", "..."],
  "feasibility_scores": {
    "technical": 0.0,
    "financial": 0.0,
    "social": 0.0,
    "timing": 0.0
  }
}
"""


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Critic", system_prompt=SYSTEM_PROMPT)

    def critique(
        self,
        idea_doc: IdeaDocument,
        round_num: int,
        prior_idea_sets: list[IdeaSet] | None = None,
    ) -> CritiqueDocument:
        context_parts = [
            "## Original Idea",
            idea_doc.model_dump_json(indent=2),
            f"\n## Round: {round_num}",
        ]

        if prior_idea_sets:
            context_parts.append("\n## Idea variants from previous rounds (do NOT repeat critiques of these)")
            for idea_set in prior_idea_sets:
                context_parts.append(f"\n### Round {idea_set.round} variants")
                for v in idea_set.variants:
                    context_parts.append(
                        f"- [{v.id}] {v.title}: {v.description}"
                    )

        context_parts.append(
            "\nCritique the original idea and any new variants. "
            "Return JSON only."
        )

        messages = [{"role": "user", "content": "\n".join(context_parts)}]
        raw = self.run(messages)

        data = self.parse_json(raw)
        return CritiqueDocument(**data)
