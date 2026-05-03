"""InterviewAgent: gathers athlete profile through conversational intake."""
from __future__ import annotations
import json

from trainer.agents.base import BaseAgent
from trainer.schemas import AthleteProfile

SYSTEM_PROMPT = """\
Sei un coach professionista di climbing e calisthenics che conduce un colloquio
di onboarding con un nuovo atleta. Il tuo obiettivo e' raccogliere tutte le
informazioni necessarie per costruire un programma personalizzato.

Fai UNA domanda alla volta. Accetta la risposta e vai avanti.
Sii caldo, professionale e incoraggiante.

## Fasi del colloquio (in ordine)

1. Dati personali:
   - Nome completo
   - Peso corporeo (kg) e altezza (cm)

2. Background climbing:
   - Anni di esperienza nell'arrampicata
   - Grado boulder attuale (scala francese: 6A, 6B, 6C, 7A, 7B...)
   - Grado vie sportive attuale (scala francese: 6a, 6b, 7a, 7b...)
   - Se non arrampica su vie, accetta "solo boulder"

3. Logistica allenamento:
   - Giorni a settimana disponibili per allenarsi (2-7)
   - Durata media della sessione (minuti)
   - Attrezzatura disponibile (hangboard e tipo, barra trazioni, anelli,
     campus board, pesi/zavorra, accesso palestra boulder, falesia)

4. Salute e limitazioni:
   - Infortuni o limitazioni fisiche attuali (dita, spalle, gomiti, altro)
   - Se nessuno, confermalo esplicitamente

5. Obiettivi:
   - Obiettivo principale (sii specifico: un grado, una skill, una metrica)
   - Durata desiderata del programma in settimane (tipicamente 6-12)

## Regole
- UNA domanda per turno
- Non elencare mai piu' domande insieme
- Se l'atleta non sa il grado esatto, accetta una descrizione approssimativa
- Dopo aver raccolto tutto, produci il profilo dentro i tag <athlete_profile>
  e chiedi conferma

## Formato output per la conferma

Quando hai tutte le informazioni:

<athlete_profile>
{
  "version": 1,
  "name": "...",
  "body_weight_kg": 70.0,
  "height_cm": 175.0,
  "years_climbing": 3,
  "boulder_grade": "6C+",
  "sport_grade": "7a",
  "training_days_per_week": 4,
  "session_duration_minutes": 90,
  "equipment": ["hangboard", "barra trazioni", "palestra boulder"],
  "injuries": [],
  "primary_goal": "raggiungere 7A boulder in 10 settimane",
  "program_weeks": 10,
  "setup_date": "YYYY-MM-DD"
}
</athlete_profile>

Poi chiedi: "E' corretto? Cosa devo cambiare?"
"""


class InterviewAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Interview", system_prompt=SYSTEM_PROMPT)
        self._history: list[dict] = []

    def _chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})
        response = self.run(self._history)
        self._history.append({"role": "assistant", "content": response})
        return response

    def interview(self, ui_print: callable, ui_input: callable, today: str) -> AthleteProfile:
        ui_print("\n[bold cyan]Iniziamo il colloquio di onboarding...[/bold cyan]\n")

        opening = self._chat(
            f"Inizia il colloquio. La data di oggi e' {today}. Fai la prima domanda."
        )
        ui_print(opening)

        profile: AthleteProfile | None = None
        draft: AthleteProfile | None = None

        while profile is None:
            user_input = ui_input("Tu: ")
            if not user_input.strip():
                continue

            response = self._chat(user_input)
            ui_print(response)

            if "<athlete_profile>" in response:
                try:
                    raw = self.extract_tagged(response, "athlete_profile")
                    data = json.loads(raw)
                    data["setup_date"] = today
                    draft = AthleteProfile(**data)
                except (ValueError, json.JSONDecodeError) as exc:
                    ui_print(f"[dim red]Errore parsing profilo: {exc}[/dim red]")
                    continue

                while True:
                    correction = ui_input("Tu: ").strip()
                    if not correction:
                        continue
                    if any(
                        kw in correction.lower()
                        for kw in ["si", "si'", "yes", "ok", "corretto", "bene",
                                   "perfetto", "giusto", "confermo", "va bene",
                                   "correct", "good", "esatto"]
                    ):
                        profile = draft
                        break
                    else:
                        response = self._chat(correction)
                        ui_print(response)
                        if "<athlete_profile>" in response:
                            try:
                                raw = self.extract_tagged(response, "athlete_profile")
                                data = json.loads(raw)
                                data["setup_date"] = today
                                draft = AthleteProfile(**data)
                            except (ValueError, json.JSONDecodeError) as exc:
                                ui_print(f"[dim red]Errore parsing: {exc}[/dim red]")

        return profile
