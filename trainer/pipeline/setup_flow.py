"""Setup flow: interview -> build program."""
from __future__ import annotations
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trainer import storage
from trainer.agents.interview import InterviewAgent
from trainer.agents.program_builder import ProgramBuilderAgent


def run(ui_print: callable, ui_input: callable, console: Console) -> None:
    today = date.today().isoformat()

    if storage.is_setup_complete():
        answer = ui_input("Profilo esistente trovato. Ricominciare da capo? [s/N]: ").strip().lower()
        if answer not in ("s", "si", "si'", "y", "yes"):
            console.print("[dim]Setup annullato.[/dim]")
            return

    interview_agent = InterviewAgent()
    profile = interview_agent.interview(ui_print, ui_input, today)
    storage.save_profile(profile)
    console.print(Panel(f"[green]Profilo salvato per {profile.name}[/green]", border_style="green"))

    console.print("\n[dim]Costruisco il programma di allenamento...[/dim]")
    builder = ProgramBuilderAgent()
    program = builder.build(profile, today)
    storage.save_program(program)

    week1 = program.weeks[0]
    table = Table(title=f"Settimana 1: {week1.theme}", show_header=True)
    table.add_column("Sessione", style="bold")
    table.add_column("Tipo")
    table.add_column("Focus")
    table.add_column("Durata")
    for day in week1.days:
        table.add_row(day.day_label, day.session_type, day.focus, f"{day.duration_minutes}min")

    console.print(f"\n[bold green]Programma pronto![/bold green] {program.total_weeks} settimane.")
    console.print(table)
    console.print(
        "\n[dim]Hai gia' dei risultati di test? Esegui [bold]python main.py test[/bold] per registrarli.\n"
        "Quando sei pronto: [bold]python main.py workout[/bold][/dim]"
    )
