"""LogProcessorAgent: parses free-form session descriptions into WorkoutLog."""
from __future__ import annotations

from trainer.agents.base import BaseAgent
from trainer.schemas import WorkoutDay, WorkoutLog

SYSTEM_PROMPT = """\
Sei un parser di log di allenamento. L'atleta descrivera' la propria sessione
in linguaggio naturale (italiano o inglese). Il tuo compito e' mappare la
descrizione sul workout pianificato fornito.

Per ogni esercizio pianificato, estrai:
- serie completate (array di set log)
- reps eseguite per serie
- carico utilizzato (kg, BW, percentuale, ecc.)
- RIR (reps in riserva) se menzionato
- note eventuali

Se un esercizio e' stato saltato: skipped: true con la ragione.
Se reps/carico non specificati per una serie: usa i valori pianificati
e aggiungi notes: "come da programma".
Non fallire mai — produci sempre un log valido.

Restituisci SOLO JSON valido. Nessun markdown, nessuna spiegazione.

Schema:
{
  "log_date": "YYYY-MM-DD",
  "week_number": 1,
  "day_label": "...",
  "session_type": "strength",
  "planned_day": {},
  "exercises": [
    {
      "name": "...",
      "sets": [
        {
          "set_num": 1,
          "reps_done": "5",
          "load_used": "+10kg",
          "rir": 2,
          "notes": ""
        }
      ],
      "skipped": false,
      "skip_reason": ""
    }
  ],
  "duration_minutes": 90,
  "session_rpe": 7.5,
  "notes": "...",
  "completed": true
}
"""


class LogProcessorAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="LogProcessor", system_prompt=SYSTEM_PROMPT)

    def process(
        self,
        planned_day: WorkoutDay,
        user_text: str,
        week_number: int,
        log_date: str,
    ) -> WorkoutLog:
        prompt = (
            f"## Sessione pianificata\n{planned_day.model_dump_json(indent=2)}\n\n"
            f"## Descrizione dell'atleta\n{user_text}\n\n"
            f"Data: {log_date} | Settimana: {week_number}\n\n"
            "Converti in WorkoutLog JSON."
        )
        raw = self.run([{"role": "user", "content": prompt}])
        data = self.parse_json(raw)
        data["log_date"] = log_date
        data["week_number"] = week_number
        data["planned_day"] = planned_day.model_dump()
        return WorkoutLog(**data)
