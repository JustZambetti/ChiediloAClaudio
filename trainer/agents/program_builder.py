"""ProgramBuilderAgent: generates the complete N-week training program."""
from __future__ import annotations

from trainer.agents.base import BaseAgent
from trainer.guidelines import GUIDELINES
from trainer.schemas import AthleteProfile, TrainingProgram

SYSTEM_PROMPT = f"""\
Sei un coach esperto di climbing e calisthenics.
Genera un programma di allenamento completo e personalizzato basandoti
sul profilo dell'atleta e sulle linee guida qui sotto.

## Linee guida di allenamento (seguile esattamente)

{GUIDELINES}

## Regole di generazione

1. Rispetta training_days_per_week e session_duration_minutes
2. Distribuisci i tipi di sessione: 2-3 climbing + 2 forza a settimana
3. Inserisci una settimana di deload ogni 4a settimana
   (is_deload: true, riduci volume 30-40%, mantieni intensita')
4. Per programmi >= 6 settimane: segna test_scheduled: true
   alla settimana 4-5 (o come piu' appropriato)
5. Ogni settimana deve avere un tema che descrive la progressione
6. Non aumentare mai volume + intensita' + difficolta' simultaneamente
7. Adatta gli esercizi all'attrezzatura disponibile
8. Se ci sono infortuni, evita esercizi che li aggravano
9. Usa RPE (es. "RPE 7-8") come linea guida carico quando i valori
   assoluti non sono noti; usa pesi specifici solo se il profilo
   indica le metriche di performance

## IMPORTANTE: Restituisci SOLO JSON valido. Nessun testo fuori dal JSON.

Schema:
{{
  "version": 1,
  "sport": "climbing_calisthenics",
  "total_weeks": 8,
  "current_week": 1,
  "created_date": "YYYY-MM-DD",
  "last_adjusted_date": null,
  "weeks": [
    {{
      "week_number": 1,
      "is_deload": false,
      "theme": "Accumulo base - forza massimale + tecnica",
      "days": [
        {{
          "day_label": "Sessione A - Forza Massimale",
          "session_type": "strength",
          "focus": "trazioni zavorrate, dip, front lever progressioni",
          "duration_minutes": 90,
          "exercises": [
            {{
              "name": "Trazioni zavorrate",
              "sets": 4,
              "reps": "3-5",
              "load": "+10kg o RPE 7-8",
              "rest_seconds": 180,
              "notes": "RIR 1-2, presa prona, ROM completo"
            }}
          ],
          "warmup": "...",
          "cooldown": "..."
        }}
      ],
      "test_scheduled": false,
      "coaching_notes": "..."
    }}
  ]
}}
"""


class ProgramBuilderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ProgramBuilder", system_prompt=SYSTEM_PROMPT)

    def build(self, profile: AthleteProfile, today: str) -> TrainingProgram:
        prompt = (
            f"## Profilo atleta\n{profile.model_dump_json(indent=2)}\n\n"
            f"Data di oggi: {today}\n\n"
            "Genera il programma di allenamento completo in JSON."
        )
        raw = self.run([{"role": "user", "content": prompt}], max_tokens=8192)
        data = self.parse_json(raw)
        data["created_date"] = today
        return TrainingProgram(**data)
