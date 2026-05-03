"""Progress flow: display training summary (no AI call)."""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trainer import storage


def _compliance(all_logs, program) -> float:
    planned = sum(len(w.days) for w in program.weeks[:max(1, program.current_week - 1)])
    completed = sum(1 for log in all_logs if log.completed)
    return min(1.0, completed / planned) if planned > 0 else 1.0


def _trend_notes(all_logs) -> list[str]:
    if len(all_logs) < 4:
        return ["Dati insufficienti per rilevare trend (servono almeno 4 sessioni)."]
    notes = []
    rpes = [log.session_rpe for log in all_logs[-4:] if log.session_rpe is not None]
    if len(rpes) >= 3:
        if all(rpes[i] < rpes[i + 1] for i in range(len(rpes) - 1)):
            notes.append("RPE in aumento costante — possibile affaticamento accumulato.")
        elif all(rpes[i] > rpes[i + 1] for i in range(len(rpes) - 1)):
            notes.append("RPE in calo — buon recupero e adattamento.")
    skipped = sum(1 for log in all_logs[-6:] if not log.completed)
    if skipped >= 2:
        notes.append(f"{skipped} sessioni saltate nelle ultime 6 — valuta il carico totale.")
    return notes or ["Nessun trend anomalo rilevato."]


def run(console: Console) -> None:
    profile = storage.load_profile()
    program = storage.load_program()
    all_logs = storage.load_all_logs()
    all_tests = storage.load_all_tests()

    compliance = _compliance(all_logs, program)
    trends = _trend_notes(all_logs)

    console.print(Panel(
        f"[bold]{profile.name}[/bold] — {profile.primary_goal}\n"
        f"Settimana [bold]{program.current_week}/{program.total_weeks}[/bold] | "
        f"Sessioni: [bold]{len(all_logs)}[/bold] | "
        f"Compliance: [bold]{'%.0f' % (compliance * 100)}%[/bold]",
        title="Progressi",
        border_style="blue",
    ))

    if all_logs:
        t = Table(title="Ultime sessioni", show_header=True)
        t.add_column("Data")
        t.add_column("Sessione")
        t.add_column("Tipo")
        t.add_column("RPE")
        t.add_column("Durata")
        for log in all_logs[-8:]:
            icon = "[green]✓[/green]" if log.completed else "[red]✗[/red]"
            t.add_row(
                log.log_date,
                f"{icon} {log.day_label[:35]}",
                log.session_type,
                str(log.session_rpe) if log.session_rpe else "—",
                f"{log.duration_minutes}min",
            )
        console.print(t)

    if all_tests:
        t2 = Table(title="Storico Test di Performance", show_header=True)
        t2.add_column("Data")
        t2.add_column("FSI (kg)")
        t2.add_column("Pull-up")
        t2.add_column("Endurance (s)")
        t2.add_column("Muscle-up")
        t2.add_column("Pistol")
        for test in all_tests:
            t2.add_row(
                test.test_date,
                str(test.fsi_kg) if test.fsi_kg is not None else "—",
                str(test.pullup_max_reps) if test.pullup_max_reps is not None else "—",
                str(test.endurance_seconds) if test.endurance_seconds is not None else "—",
                str(test.muscle_up_level) if test.muscle_up_level is not None else "—",
                str(test.pistol_level) if test.pistol_level is not None else "—",
            )
        console.print(t2)

    console.print("\n[bold]Trend:[/bold]")
    for note in trends:
        console.print(f"  {note}")

    console.print("\n[dim]Usa [bold]python main.py adjust[/bold] per adattare il programma.[/dim]")
