"""Leaderboard — prints accuracy across all scored runs in results/scores/."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = _REPO_ROOT / "results" / "scores"

console = Console()


def print_leaderboard() -> None:
    if not SCORES_DIR.exists():
        console.print("[yellow]No scored runs yet. Run `make bench AGENT=… MODEL=…` first.[/yellow]")
        return

    entries = []
    for path in sorted(SCORES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries.append(data)

    if not entries:
        console.print("[yellow]No scored runs yet.[/yellow]")
        return

    entries.sort(key=lambda d: d.get("accuracy", 0), reverse=True)

    table = Table(title="BrowseComp Leaderboard")
    table.add_column("Agent", style="bold")
    table.add_column("Models", style="dim")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Run ID")

    for d in entries:
        table.add_row(
            str(d.get("agent", "?")),
            str(d.get("model_tag", "?")),
            f"{d.get('accuracy', 0):.3f}",
            str(d.get("correct", 0)),
            str(d.get("tasks", 0)),
            str(d.get("run_id", "?")),
        )
    console.print(table)
