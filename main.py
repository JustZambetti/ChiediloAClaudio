#!/usr/bin/env python3
"""AI Personal Trainer — Climbing (Boulder + Vie) e Calisthenics"""
from __future__ import annotations
import argparse
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def ui_print(text: str) -> None:
    console.print(text)


def ui_input(prompt: str) -> str:
    return console.input(f"[bold green]{prompt}[/bold green]")


def print_header() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold]AI Personal Trainer[/bold]\n"
        "[dim]Climbing (Boulder + Vie) e Calisthenics[/dim]",
        border_style="blue",
    ))
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trainer",
        description="AI Personal Trainer per Climbing e Calisthenics",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("setup",     help="Setup iniziale: colloquio + costruzione programma")
    sub.add_parser("workout",   help="Mostra la sessione di oggi")
    sub.add_parser("log",       help="Registra la sessione di oggi")
    sub.add_parser("test",      help="Registra i risultati dei test di performance")
    sub.add_parser("progress",  help="Visualizza i progressi")
    sub.add_parser("adjust",    help="Analizza e adatta il programma con l'AI")
    sub.add_parser("next-week", help="Avanza manualmente alla prossima settimana")

    args = parser.parse_args()

    if not args.command:
        print_header()
        parser.print_help()
        sys.exit(0)

    try:
        from trainer import storage
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        console.print("[dim]Esegui: pip install -r requirements.txt[/dim]")
        sys.exit(1)

    if args.command != "setup" and not storage.is_setup_complete():
        console.print("[red]Nessun profilo trovato. Esegui prima: python main.py setup[/red]")
        sys.exit(1)

    print_header()

    try:
        if args.command == "setup":
            from trainer.pipeline import setup_flow
            setup_flow.run(ui_print, ui_input, console)

        elif args.command == "workout":
            from trainer.pipeline import workout_flow
            workout_flow.run(console)

        elif args.command == "log":
            from trainer.pipeline import log_flow
            log_flow.run(ui_print, ui_input, console)

        elif args.command == "test":
            from trainer.pipeline import test_flow
            test_flow.run(ui_input, console)

        elif args.command == "progress":
            from trainer.pipeline import progress_flow
            progress_flow.run(console)

        elif args.command == "adjust":
            from trainer.pipeline import adjust_flow
            adjust_flow.run(ui_input, console)

        elif args.command == "next-week":
            program = storage.load_program()
            if program.current_week >= program.total_weeks:
                console.print("[yellow]Sei gia' all'ultima settimana del programma.[/yellow]")
            else:
                program.current_week += 1
                storage.save_program(program)
                console.print(f"[green]Avanzato alla settimana {program.current_week}.[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrotto.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
