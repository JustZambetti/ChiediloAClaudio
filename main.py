#!/usr/bin/env python3
"""
ChiediloAClaudio — Creative Brainstorming Pipeline
===================================================

Usage:
    python main.py                    # default 3 divergence rounds
    python main.py --rounds 2         # fewer rounds (faster)
    python main.py --rounds 4         # more rounds (deeper)
    python main.py --output report.md # also save report to file
"""

from __future__ import annotations
import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


def ui_print(text: str) -> None:
    """Render agent output with rich formatting."""
    if text.startswith("\n[") or text.startswith("["):
        # Agent label lines
        console.print(Text(text, style="dim cyan"))
    elif text.startswith("┌") or text.startswith("└") or text.startswith("│"):
        # Checkpoint box borders
        console.print(Text(text, style="bold yellow"))
    elif text.startswith("  ★"):
        console.print(Text(text, style="bold green"))
    elif text.startswith("  DIVERGENCE ROUND") or text.startswith("  Running"):
        console.print(Text(text, style="bold blue"))
    elif text.startswith("  SYNTHESIZING"):
        console.print(Text(text, style="bold magenta"))
    elif text.startswith("─" * 10):
        console.print(Rule(style="dim"))
    else:
        console.print(text)


def ui_input(prompt: str) -> str:
    """Styled input prompt."""
    return console.input(f"[bold green]{prompt}[/bold green]")


def print_header() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold]ChiediloAClaudio[/bold]\n"
        "[dim]Creative Brainstorming Pipeline[/dim]\n\n"
        "[dim]Agents:[/dim] Harvester → Critic + Researcher → Expander → Judge → Synthesizer\n"
        "[dim]Techniques:[/dim] Prompt chaining, parallel fan-out, SCAMPER, "
        "constraint inversion, iterative divergence loops",
        border_style="blue",
    ))
    console.print()


def print_report(report, output_path: str | None = None) -> None:
    md_text = report.to_markdown()

    console.print()
    console.print(Rule("[bold magenta]BRAINSTORM REPORT[/bold magenta]"))
    console.print()
    console.print(Markdown(md_text))

    if output_path:
        with open(output_path, "w") as f:
            f.write(md_text)
        console.print(f"\n[dim]Report saved to: {output_path}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Creative brainstorming pipeline powered by Claude agents"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of divergence rounds (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the final report as markdown",
    )
    args = parser.parse_args()

    if args.rounds < 1 or args.rounds > 5:
        console.print("[red]--rounds must be between 1 and 5[/red]")
        sys.exit(1)

    print_header()

    # Import here so missing deps give a clean error message
    try:
        from brainstorm.pipeline.runner import BrainstormPipeline
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        console.print("[dim]Run: pip install -r requirements.txt[/dim]")
        sys.exit(1)

    pipeline = BrainstormPipeline(
        max_rounds=args.rounds,
        ui_print=ui_print,
        ui_input=ui_input,
    )

    try:
        report = pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")
        sys.exit(0)

    print_report(report, output_path=args.output)


if __name__ == "__main__":
    main()
