"""
Agent 5: Judge (loop controller)

The Judge is the most architecturally critical agent — it decides
whether the divergence loop continues and tells the Expander exactly
where to go next.

Stopping conditions (any one is sufficient):
  1. similarity_to_previous > 0.7  → spinning in place
  2. quality not improving AND round >= 2
  3. round >= max_rounds (hard cap)
  4. breakout idea detected (one variant dramatically outscores rest)

Directives are specific, not generic: "explore the timing constraint",
"go weirder with domain analogies", "focus on eliminating the [X] assumption".
"""

from __future__ import annotations

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import IdeaSet, JudgeVerdict

SYSTEM_PROMPT = """\
You are the Judge — the loop controller of a creative brainstorming pipeline. \
Your job is to evaluate each round's idea variants and make two decisions:

1. **Continue or stop?** Should the pipeline run another divergence round?
2. **What directive to give next?** If continuing, exactly where should the \
   Expander push?

## Scoring criteria (all 0.0–1.0)

**novelty_score**: How genuinely different are these ideas from what came \
before? 0 = basically the same, 1 = completely new territory.

**diversity_score**: How different are the variants from each other *within* \
this round? 0 = all variations of the same idea, 1 = totally distinct directions.

**quality_score**: Composite of:
  - originality (is this surprising and non-obvious?)
  - feasibility (could this actually be built/done?)
  - usefulness (does it solve a real problem in a meaningful way?)

**similarity_to_previous**: What fraction of ideas are conceptually recycled \
from prior rounds? 0 = all new, 1 = all repeats.

## Stopping rules
- STOP if similarity_to_previous > 0.7 (spinning in place)
- STOP if quality_score < 0.3 for round 2+ (ideas getting worse)
- STOP if round >= max_rounds (passed in context)
- STOP if a breakout idea exists AND quality_score > 0.75
- CONTINUE otherwise

## Directive rules
The directive is the most important output. Be specific:
- BAD: "be more creative"
- GOOD: "apply Constraint-Inversion to the financial feasibility problem — \
  what if the business model was inverted so users pay nothing?"
- GOOD: "explore domain analogies from logistics or supply chain — \
  how does Amazon last-mile delivery thinking apply here?"
- GOOD: "push the Eliminate technique harder — strip out the core \
  assumption about needing [X] entirely"

If similarity is high, the directive must inject a wildcard:
use constraint inversion, a forced random domain (pick one: military \
logistics, surgery, kindergartens, jazz, Formula 1 pit stops), or \
reverse the entire user/provider relationship.

## Output format
Return ONLY valid JSON:

{
  "round": <int>,
  "novelty_score": 0.0,
  "diversity_score": 0.0,
  "quality_score": 0.0,
  "similarity_to_previous": 0.0,
  "should_continue": true,
  "stop_reason": null,
  "breakout_idea_id": null,
  "directive": "...",
  "top_variant_ids": ["v1-tag", "v1-tag2"]
}
"""


class JudgeAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Judge", system_prompt=SYSTEM_PROMPT)
        self.prior_quality_scores: list[float] = []

    def evaluate(
        self,
        current_idea_set: IdeaSet,
        all_idea_sets: list[IdeaSet],
        max_rounds: int,
    ) -> JudgeVerdict:
        context_parts = [
            f"## Current round: {current_idea_set.round}",
            f"## Max rounds allowed: {max_rounds}",
            "\n## Current round's variants (evaluate these)",
            current_idea_set.model_dump_json(indent=2),
        ]

        if len(all_idea_sets) > 1:
            context_parts.append("\n## All prior rounds' variants (for similarity comparison)")
            for idea_set in all_idea_sets[:-1]:  # exclude current
                context_parts.append(f"\n### Round {idea_set.round}")
                for v in idea_set.variants:
                    context_parts.append(
                        f"- [{v.id}] {v.title}: {v.diverges_from_original_by}"
                    )
        else:
            context_parts.append("\n(This is the first round — no prior variants to compare.)")

        context_parts.append("\nEvaluate and return your JudgeVerdict JSON.")

        messages = [{"role": "user", "content": "\n".join(context_parts)}]
        raw = self.run(messages)

        data = self.parse_json(raw)
        verdict = JudgeVerdict(**data)

        # Track quality trend for stopping condition
        self.prior_quality_scores.append(verdict.quality_score)

        # Override should_continue if quality is declining
        if (
            len(self.prior_quality_scores) >= 2
            and verdict.quality_score < self.prior_quality_scores[-2] - 0.1
            and current_idea_set.round >= 2
        ):
            verdict.should_continue = False
            verdict.stop_reason = (
                f"Quality declining: {self.prior_quality_scores[-2]:.2f} → "
                f"{verdict.quality_score:.2f}"
            )

        return verdict
