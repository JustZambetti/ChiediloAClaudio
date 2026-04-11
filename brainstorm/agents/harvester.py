"""
Agent 1: Idea Harvester

Conducts a structured 5-round interview with the user to extract
a rich, accurate IdeaDocument. Uses conversational prompt chaining:
each agent turn is informed by the full conversation history so far.

Interview protocol:
  Round 1 — LISTEN    : open question, let them talk freely
  Round 2 — PROBE 5W  : who, what, where, when, why
  Round 3 — EMOTIONAL : excitement, fears, non-negotiables
  Round 4 — CONSTRAINTS: hard limits and unknowns
  Round 5 — CONFIRM   : draft IdeaDocument, user corrects until satisfied
"""

from __future__ import annotations
import json

from brainstorm.agents.base import BaseAgent
from brainstorm.schemas import IdeaDocument

SYSTEM_PROMPT = """\
You are a creative idea harvester — warm, curious, and deeply attentive.
Your sole job is to extract the full richness of an idea from the person \
in front of you, then distill it into a precise IdeaDocument.

## Your interview protocol has 5 rounds

**ROUND 1 — LISTEN**
Ask one broad, welcoming open question. Let the person speak freely.
Do NOT interrupt or ask sub-questions yet. Reflect back a one-sentence \
summary of what you heard before moving on.

**ROUND 2 — PROBE (5W)**
Ask targeted follow-up questions to fill in Who, What, Where, When, Why.
Ask ONE question at a time. After each answer, reflect it back briefly.

**ROUND 3 — EMOTIONAL**
Probe the emotional dimension:
- "What excites you most about this?"
- "What worries you?"
- "If this succeeded completely, what would that look like?"
Ask one at a time.

**ROUND 4 — CONSTRAINTS**
Surface hard limits and unknowns:
- "What absolutely cannot change about this idea?"
- "What do you not know yet that you'll need to figure out?"

**ROUND 5 — CONFIRM**
Draft the IdeaDocument JSON and show it to the user inside <idea_document> tags.
Ask: "Does this capture your idea accurately? What's missing or wrong?"
Keep refining until the user confirms it's right.

## Rules
- Ask ONE question at a time — never a list of questions.
- Reflect back what you heard before each new question.
- Use "Yes, and..." framing to show you're building on their words.
- Be warm and encouraging — ideas are fragile at this stage.
- When drafting the IdeaDocument, be specific, not generic.
- The excitement_level should come from what they actually said.

## IdeaDocument JSON schema
Output the draft inside <idea_document> tags like this:

<idea_document>
{
  "version": 1,
  "core_concept": "...",
  "problem_solved": "...",
  "target_audience": "...",
  "inspiration_source": "...",
  "constraints": ["...", "..."],
  "unknowns": ["...", "..."],
  "excitement_level": 8,
  "domain_tags": ["...", "..."],
  "raw_quotes": ["exact phrase they used", "..."]
}
</idea_document>
"""


class HarvesterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Harvester", system_prompt=SYSTEM_PROMPT)
        self.conversation_history: list[dict] = []

    def _chat(self, user_message: str) -> str:
        """Send a user message, get assistant response, update history."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        response_text = self.run(self.conversation_history)
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })
        return response_text

    def harvest(self, ui_print: callable, ui_input: callable) -> IdeaDocument:
        """
        Run the full interview loop.

        ui_print(text) — display agent output to user
        ui_input(prompt) — get user input
        """
        ui_print("\n[Harvester] Starting the idea interview...\n")

        # ── Round 1: open question ────────────────────────────────────────
        opening = self._chat(
            "Please start the interview. Ask me the opening question."
        )
        ui_print(opening)

        # Main conversation loop — runs until we have a confirmed IdeaDocument
        idea_doc: IdeaDocument | None = None
        confirm_mode = False

        while idea_doc is None:
            user_input = ui_input("You: ")
            if not user_input.strip():
                continue

            response = self._chat(user_input)
            ui_print(response)

            # Check if agent has drafted an IdeaDocument
            if "<idea_document>" in response:
                try:
                    raw_json = self.extract_tagged(response, "idea_document")
                    data = json.loads(raw_json)
                    draft = IdeaDocument(**data)
                    confirm_mode = True

                    # Let user confirm or correct
                    while confirm_mode:
                        correction = ui_input("You: ").strip()
                        if not correction:
                            continue

                        # If user signals acceptance
                        if any(
                            kw in correction.lower()
                            for kw in ["yes", "correct", "good", "perfect",
                                       "that's right", "looks good", "confirmed"]
                        ):
                            idea_doc = draft
                            confirm_mode = False
                        else:
                            # User is correcting — feed back and re-draft
                            response = self._chat(correction)
                            ui_print(response)
                            if "<idea_document>" in response:
                                raw_json = self.extract_tagged(
                                    response, "idea_document"
                                )
                                data = json.loads(raw_json)
                                draft = IdeaDocument(**data)

                except (ValueError, json.JSONDecodeError) as exc:
                    ui_print(f"\n[Harvester] Could not parse IdeaDocument: {exc}\n")

        ui_print("\n[Harvester] IdeaDocument confirmed. Passing to the pipeline.\n")
        return idea_doc
