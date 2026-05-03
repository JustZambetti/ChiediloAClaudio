"""AdjustmentAgent: analyzes progress and adapts the training program."""
from __future__ import annotations

from trainer.agents.base import BaseAgent
from trainer.guidelines import GUIDELINES
from trainer.schemas import AthleteProfile, TrainingProgram, ProgressSnapshot, AdjustmentDecision

SYSTEM_PROMPT = f"""\
Sei un coach di climbing e calisthenics che conduce una revisione
mid-programma delle performance dell'atleta.

## Linee guida e regole di adattamento

{GUIDELINES}

## Opzioni di decisione
- "continue": nessuna modifica, l'atleta e' in carreggiata
- "increase_load": progressione ottimale, aumenta difficolta'
- "deload_now": segnali di fatica (RPE crescente, sessioni saltate,
  performance in calo) — inserisci deload prima del previsto
- "reduce_volume": troppo volume, riduci per migliorare recupero
- "add_endurance": endurance in calo, aggiungi sessioni vie
- "increase_skill_complexity": skill migliorata, passa al livello successivo
- "full_replan": cambio maggiore necessario (infortunio, cambio orari,
  plateau fondamentale)

## Procedura di analisi
1. Controlla compliance rate (< 70% -> considera deload o rimodulazione)
2. Confronta ultimo test vs precedente (se disponibili):
   - FSI +5-10%: aumenta difficolta' boulder, riduci volume hangboard
   - Stagnazione pull-up: aumenta recupero, riduci accessori, aggiungi zavorra
   - Endurance scende: piu' sessioni vie, riduci intensita' massimale
   - Skill migliorata: aumenta complessita' (carico, tempo, leva)
3. Controlla trend RPE nelle ultime sessioni (RPE crescente su stessi pesi = fatica)
4. Verifica se e' dovuto il deload (ogni 4 settimane)

Restituisci SOLO JSON valido:
{{
  "action": "...",
  "reason": "...",
  "changes": ["modifica specifica 1", "modifica specifica 2"],
  "updated_weeks": null,
  "coaching_message": "messaggio per l'atleta in italiano, 2-3 frasi"
}}
"""


class AdjustmentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Adjustment", system_prompt=SYSTEM_PROMPT)

    def analyze(
        self,
        snapshot: ProgressSnapshot,
        program: TrainingProgram,
        profile: AthleteProfile,
    ) -> AdjustmentDecision:
        prompt = (
            f"## Profilo atleta\n{profile.model_dump_json(indent=2)}\n\n"
            f"## Stato programma\n"
            f"Settimana corrente: {program.current_week}/{program.total_weeks}\n\n"
            f"## Snapshot progresso\n{snapshot.model_dump_json(indent=2)}\n\n"
            "Analizza e restituisci AdjustmentDecision JSON."
        )
        raw = self.run([{"role": "user", "content": prompt}])
        data = self.parse_json(raw)
        return AdjustmentDecision(**data)
