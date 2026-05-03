"""Test flow: record performance test results (FSI, pull-ups, endurance, skills)."""
from __future__ import annotations
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trainer import storage
from trainer.schemas import PerformanceTest


def _parse_float(s: str) -> float | None:
    s = s.strip()
    if not s or s.lower() in ("skip", "s", "n", "-", "?", "non lo so", "unknown"):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _parse_int(s: str) -> int | None:
    f = _parse_float(s)
    return int(f) if f is not None else None


def run(ui_input: callable, console: Console) -> None:
    today = date.today().isoformat()
    all_tests = storage.load_all_tests()
    prev = all_tests[-1] if all_tests else None

    console.print(Panel(
        "[bold]Test di Performance[/bold]\n\n"
        "Esegui i test seguendo i protocolli e inserisci i risultati.\n"
        "Premi Invio o digita 'skip' per saltare un test.",
        border_style="cyan",
    ))
    if prev:
        console.print(f"[dim]Ultimo test: {prev.test_date}[/dim]")

    # Test 1: FSI
    console.print("\n[bold cyan]Test 1: Finger Strength Index (FSI)[/bold cyan]")
    console.print(
        "[dim]Protocollo: dead hang 10s su presa 20mm half crimp, "
        "aggiungi peso progressivamente (3-5 tentativi).\n"
        "Inserisci i kg EXTRA oltre al peso corporeo.[/dim]"
    )
    if prev and prev.fsi_kg is not None:
        console.print(f"[dim]Precedente: +{prev.fsi_kg}kg[/dim]")
    fsi = _parse_float(ui_input("FSI — kg extra (Invio=skip): "))

    # Test 2: Pull-up
    console.print("\n[bold cyan]Test 2: Trazioni Massimali[/bold cyan]")
    console.print("[dim]Protocollo: strict pull-up, presa prona, ROM completo.[/dim]")
    if prev and prev.pullup_max_reps is not None:
        console.print(f"[dim]Precedente: {prev.pullup_max_reps} reps BW[/dim]")
    pullup_reps = _parse_int(ui_input("Reps a corpo libero (Invio=skip): "))
    if prev and prev.pullup_weighted_kg is not None:
        console.print(f"[dim]Precedente (zavorra): +{prev.pullup_weighted_kg}kg x3[/dim]")
    pullup_weighted = _parse_float(ui_input("Zavorra per 3 reps (kg extra, Invio=skip): "))

    # Test 3: Endurance
    console.print("\n[bold cyan]Test 3: Forza-Resistenza — Pump Test[/bold cyan]")
    console.print(
        "[dim]Protocollo: circuito su pannello/falesia a media-alta intensita', "
        "misura secondi fino a cedimento tecnico.[/dim]"
    )
    if prev and prev.endurance_seconds is not None:
        console.print(f"[dim]Precedente: {prev.endurance_seconds}s[/dim]")
    endurance = _parse_int(ui_input("Tempo (secondi, Invio=skip): "))

    # Test 4: Muscle-up
    console.print("\n[bold cyan]Test 4: Muscle-Up Level (0-4)[/bold cyan]")
    console.print("[dim]0=nessuno | 1=jump assisted | 2=band | 3=strict singolo | 4=multipli[/dim]")
    if prev and prev.muscle_up_level is not None:
        console.print(f"[dim]Precedente: livello {prev.muscle_up_level}[/dim]")
    mu_level = _parse_int(ui_input("Livello muscle-up (0-4, Invio=skip): "))

    # Test 5: Pistol
    console.print("\n[bold cyan]Test 5: Pistol Squat Level (0-4)[/bold cyan]")
    console.print("[dim]0=nessuno | 1=assistito | 2=libero instabile | 3=controllato | 4=zavorrato[/dim]")
    if prev and prev.pistol_level is not None:
        console.print(f"[dim]Precedente: livello {prev.pistol_level}[/dim]")
    pistol_level = _parse_int(ui_input("Livello pistol (0-4, Invio=skip): "))

    notes = ui_input("Note aggiuntive (Invio=nessuna): ").strip()

    test = PerformanceTest(
        test_date=today,
        fsi_kg=fsi,
        pullup_max_reps=pullup_reps,
        pullup_weighted_kg=pullup_weighted,
        endurance_seconds=endurance,
        muscle_up_level=mu_level,
        pistol_level=pistol_level,
        notes=notes,
    )
    storage.save_test(test)

    if prev:
        def _delta(new, old) -> str:
            if new is None or old is None:
                return "—"
            diff = new - old
            sign = "+" if diff >= 0 else ""
            color = "green" if diff >= 0 else "red"
            return f"[{color}]{sign}{diff}[/{color}]"

        table = Table(title="Risultati vs test precedente", show_header=True)
        table.add_column("Test")
        table.add_column(f"Oggi ({today})")
        table.add_column(f"Precedente ({prev.test_date})")
        table.add_column("Delta")
        table.add_row("FSI (kg extra)", str(fsi) if fsi is not None else "—", str(prev.fsi_kg) if prev.fsi_kg is not None else "—", _delta(fsi, prev.fsi_kg))
        table.add_row("Pull-up reps (BW)", str(pullup_reps) if pullup_reps is not None else "—", str(prev.pullup_max_reps) if prev.pullup_max_reps is not None else "—", _delta(pullup_reps, prev.pullup_max_reps))
        table.add_row("Pull-up zavorra", str(pullup_weighted) if pullup_weighted is not None else "—", str(prev.pullup_weighted_kg) if prev.pullup_weighted_kg is not None else "—", _delta(pullup_weighted, prev.pullup_weighted_kg))
        table.add_row("Endurance (sec)", str(endurance) if endurance is not None else "—", str(prev.endurance_seconds) if prev.endurance_seconds is not None else "—", _delta(endurance, prev.endurance_seconds))
        table.add_row("Muscle-up livello", str(mu_level) if mu_level is not None else "—", str(prev.muscle_up_level) if prev.muscle_up_level is not None else "—", _delta(mu_level, prev.muscle_up_level))
        table.add_row("Pistol livello", str(pistol_level) if pistol_level is not None else "—", str(prev.pistol_level) if prev.pistol_level is not None else "—", _delta(pistol_level, prev.pistol_level))
        console.print(table)
    else:
        console.print(Panel(f"[green]Primo test registrato ({today})[/green]", border_style="green"))

    console.print("\n[dim]Usa [bold]python main.py adjust[/bold] per analizzare i risultati.[/dim]")
