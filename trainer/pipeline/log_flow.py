"""Log flow: record today's session from free-form text."""
from __future__ import annotations
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trainer import storage
from trainer.agents.log_processor import LogProcessorAgent
from trainer.schemas import WorkoutDay, WorkoutLog, ExerciseLog


def _next_unlogged_day(program, all_logs: list) -> WorkoutDay | None:
    current_week = program.current()
    logged_labels = {log.day_label for log in all_logs if log.week_number == program.current_week}
    for day in current_week.days:
        if day.day_label not in logged_labels:
            return day
    return None


def _all_logged_this_week(program, all_logs: list) -> bool:
    current_week = program.current()
    logged_labels = {log.day_label for log in all_logs if log.week_number == program.current_week}
    return all(day.day_label in logged_labels for day in current_week.days)


def run(ui_print: callable, ui_input: callable, console: Console) -> None:
    today = date.today().isoformat()
    program = storage.load_program()
    all_logs = storage.load_all_logs()

    if storage.load_log(today):
        console.print(Panel("[yellow]Hai gia' registrato una sessione oggi.[/yellow]", border_style="yellow"))
        return

    day = _next_unlogged_day(program, all_logs)
    if day is None:
        console.print(Panel(
            "[green]Tutte le sessioni di questa settimana sono registrate.[/green]\n"
            "Usa [bold]python main.py next-week[/bold] per avanzare.",
            border_style="green",
        ))
        return

    table = Table(title=f"Piano — {day.day_label}", show_header=True)
    table.add_column("Esercizio")
    table.add_column("Serie x Reps")
    table.add_column("Carico")
    for ex in day.exercises:
        table.add_row(ex.name, f"{ex.sets}x{ex.reps}", ex.load)
    console.print(table)

    user_text = ui_input("\nDescrivi la tua sessione (o 'skip' per saltare): ").strip()

    if user_text.lower() in ("skip", "s"):
        skip_log = WorkoutLog(
            log_date=today,
            week_number=program.current_week,
            day_label=day.day_label,
            session_type=day.session_type,
            planned_day=day,
            exercises=[
                ExerciseLog(name=ex.name, sets=[], skipped=True, skip_reason="Sessione saltata")
                for ex in day.exercises
            ],
            duration_minutes=0,
            completed=False,
            notes="Sessione saltata",
        )
        storage.save_log(skip_log)
        console.print("[dim]Sessione saltata registrata.[/dim]")
        return

    console.print("\n[dim]Elaboro il log...[/dim]")
    agent = LogProcessorAgent()
    log = agent.process(day, user_text, program.current_week, today)

    parsed_table = Table(title="Log elaborato — conferma", show_header=True)
    parsed_table.add_column("Esercizio")
    parsed_table.add_column("Serie")
    parsed_table.add_column("Dettaglio")
    for ex_log in log.exercises:
        detail = ", ".join(f"{s.reps_done}x{s.load_used}" for s in ex_log.sets) if ex_log.sets else "saltato"
        parsed_table.add_row(ex_log.name, str(len(ex_log.sets)), detail)
    console.print(parsed_table)
    console.print(f"RPE: {log.session_rpe} | Durata: {log.duration_minutes}min | Note: {log.notes}")

    confirm = ui_input("Confermi? [S/n]: ").strip().lower()
    if confirm in ("n", "no"):
        console.print("[dim]Log annullato.[/dim]")
        return

    storage.save_log(log)
    console.print("[green]Sessione registrata.[/green]")

    all_logs_updated = storage.load_all_logs()
    if _all_logged_this_week(program, all_logs_updated):
        if program.current_week < program.total_weeks:
            program.current_week += 1
            storage.save_program(program)
            console.print(
                f"\n[bold green]Settimana {program.current_week - 1} completata! "
                f"Avanzato alla settimana {program.current_week}.[/bold green]"
            )
        else:
            console.print("\n[bold green]Programma completato! Usa [bold]python main.py progress[/bold].[/bold green]")
    else:
        console.print("\n[dim]Prossima sessione: [bold]python main.py workout[/bold][/dim]")
