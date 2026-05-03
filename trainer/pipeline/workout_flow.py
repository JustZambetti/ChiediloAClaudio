"""Workout flow: show today's planned session with AI briefing."""
from __future__ import annotations
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trainer import storage
from trainer.agents.workout_presenter import WorkoutPresenterAgent
from trainer.schemas import WorkoutDay


def _next_unlogged_day(program, all_logs: list) -> WorkoutDay | None:
    current_week = program.current()
    logged_labels = {log.day_label for log in all_logs if log.week_number == program.current_week}
    for day in current_week.days:
        if day.day_label not in logged_labels:
            return day
    return None


def run(console: Console) -> None:
    profile = storage.load_profile()
    program = storage.load_program()
    all_logs = storage.load_all_logs()
    recent_logs = storage.load_recent_logs(n=6)

    if program.current_week > program.total_weeks:
        console.print(Panel(
            "[bold green]Programma completato![/bold green]\n"
            "Usa [bold]python main.py progress[/bold] per vedere i risultati.",
            border_style="green",
        ))
        return

    today = date.today().isoformat()
    today_log = storage.load_log(today)
    if today_log:
        console.print(Panel(
            f"[yellow]Sessione di oggi gia' registrata: {today_log.day_label}[/yellow]\n"
            f"RPE: {today_log.session_rpe} | Note: {today_log.notes}",
            border_style="yellow",
        ))
        return

    day = _next_unlogged_day(program, all_logs)
    if day is None:
        console.print(Panel(
            f"[green]Settimana {program.current_week} completata![/green]\n"
            "Usa [bold]python main.py next-week[/bold] per avanzare.",
            border_style="green",
        ))
        return

    week = program.current()
    console.print(f"\n[bold]Settimana {program.current_week}/{program.total_weeks}[/bold] — {week.theme}")
    if week.is_deload:
        console.print("[yellow]SETTIMANA DI DELOAD[/yellow]")
    if week.test_scheduled:
        console.print(
            "[cyan]TEST DI PERFORMANCE programmato questa settimana — "
            "usa [bold]python main.py test[/bold][/cyan]"
        )

    console.print("\n[dim]Preparando briefing sessione...[/dim]")
    presenter = WorkoutPresenterAgent()
    briefing = presenter.present(day, profile, recent_logs)

    console.print(Panel(briefing, title=f"[bold]{day.day_label}[/bold]", border_style="blue"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Esercizio")
    table.add_column("Serie")
    table.add_column("Reps")
    table.add_column("Carico")
    table.add_column("Recupero")
    table.add_column("Note")
    for ex in day.exercises:
        table.add_row(ex.name, str(ex.sets), ex.reps, ex.load, f"{ex.rest_seconds}s", ex.notes)
    console.print(table)

    console.print("\n[dim]Dopo la sessione: [bold]python main.py log[/bold][/dim]")
