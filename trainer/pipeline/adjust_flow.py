"""Adjust flow: analyze progress with AI and optionally adapt the program."""
from __future__ import annotations
from datetime import date
from rich.console import Console
from rich.panel import Panel

from trainer import storage
from trainer.agents.adjustment import AdjustmentAgent
from trainer.schemas import ProgressSnapshot


ACTION_COLORS = {
    "continue": "green",
    "increase_load": "cyan",
    "deload_now": "yellow",
    "reduce_volume": "yellow",
    "add_endurance": "blue",
    "increase_skill_complexity": "magenta",
    "full_replan": "red",
}


def _build_snapshot(program, all_logs, all_tests) -> ProgressSnapshot:
    planned = sum(len(w.days) for w in program.weeks[:max(1, program.current_week - 1)])
    completed = sum(1 for log in all_logs if log.completed)
    compliance = min(1.0, completed / planned) if planned > 0 else 1.0

    trend_notes = []
    recent = all_logs[-6:]
    rpes = [log.session_rpe for log in recent[-4:] if log.session_rpe is not None]
    if len(rpes) >= 3 and all(rpes[i] < rpes[i + 1] for i in range(len(rpes) - 1)):
        trend_notes.append("RPE in aumento costante — possibile affaticamento")
    skipped = sum(1 for log in recent if not log.completed)
    if skipped >= 2:
        trend_notes.append(f"{skipped} sessioni saltate nelle ultime 6")

    return ProgressSnapshot(
        weeks_completed=max(0, program.current_week - 1),
        total_sessions=len(all_logs),
        compliance_rate=compliance,
        latest_test=all_tests[-1] if all_tests else None,
        previous_test=all_tests[-2] if len(all_tests) >= 2 else None,
        recent_logs=all_logs[-12:],
        trend_notes=trend_notes,
    )


def run(ui_input: callable, console: Console) -> None:
    profile = storage.load_profile()
    program = storage.load_program()
    all_logs = storage.load_all_logs()
    all_tests = storage.load_all_tests()

    snapshot = _build_snapshot(program, all_logs, all_tests)

    console.print("\n[dim]Analizzo i progressi...[/dim]")
    agent = AdjustmentAgent()
    decision = agent.analyze(snapshot, program, profile)

    color = ACTION_COLORS.get(decision.action, "white")
    changes_text = "\n".join(f"  • {c}" for c in decision.changes)
    console.print(Panel(
        f"[{color}][bold]{decision.action.upper().replace('_', ' ')}[/bold][/{color}]\n\n"
        f"[bold]Motivo:[/bold] {decision.reason}\n\n"
        f"{changes_text}\n\n"
        f"[italic]{decision.coaching_message}[/italic]",
        title="Analisi Adattamento",
        border_style=color,
    ))

    if decision.action == "continue":
        console.print("[dim]Nessuna modifica necessaria. Continua cosi'![/dim]")
        return

    confirm = ui_input("Applicare le modifiche? [S/n]: ").strip().lower()
    if confirm in ("n", "no"):
        console.print("[dim]Nessuna modifica apportata.[/dim]")
        return

    if decision.updated_weeks:
        for updated_week in decision.updated_weeks:
            idx = updated_week.week_number - 1
            if 0 <= idx < len(program.weeks):
                program.weeks[idx] = updated_week

    program.last_adjusted_date = date.today().isoformat()
    storage.save_program(program)
    console.print("[green]Programma aggiornato.[/green]")
    console.print("[dim]Usa [bold]python main.py workout[/bold] per la prossima sessione.[/dim]")
