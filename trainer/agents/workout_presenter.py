"""WorkoutPresenterAgent: generates a motivating pre-session briefing."""
from __future__ import annotations

from trainer.agents.base import BaseAgent
from trainer.schemas import AthleteProfile, WorkoutDay, WorkoutLog

SYSTEM_PROMPT = """\
Sei un coach di climbing e calisthenics che presenta il briefing pre-sessione.
Dato il workout pianificato e i dati recenti dell'atleta, produci un piano di
sessione chiaro e motivante in italiano.

Includi:
- Scopo della sessione di oggi (perche' e' importante per l'obiettivo)
- Indicazioni specifiche per il riscaldamento in base al tipo di sessione
- Per ogni esercizio: cue tecnici chiave e aggiustamenti del carico
  se l'RPE recente suggerisce fatica o recupero
- Un focus cue finale da portarsi in sessione

Sii conciso (max 300 parole). Solo testo plain — nessun JSON.
Rispondi in italiano.
"""


class WorkoutPresenterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="WorkoutPresenter", system_prompt=SYSTEM_PROMPT)

    def present(
        self,
        day: WorkoutDay,
        profile: AthleteProfile,
        recent_logs: list[WorkoutLog],
    ) -> str:
        recent_summary = "\n".join(
            f"- {log.log_date}: {log.day_label} | RPE {log.session_rpe} | {log.notes}"
            for log in recent_logs[-3:]
        ) or "Nessuna sessione recente."

        prompt = (
            f"## Sessione pianificata\n{day.model_dump_json(indent=2)}\n\n"
            f"## Profilo atleta\n{profile.model_dump_json(indent=2)}\n\n"
            f"## Ultime sessioni\n{recent_summary}\n\n"
            "Scrivi il briefing della sessione in italiano."
        )
        return self.run([{"role": "user", "content": prompt}])
