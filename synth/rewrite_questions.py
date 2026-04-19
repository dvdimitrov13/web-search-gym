"""Retroactively rewrite existing filterbench questions in BrowseComp style.

Loads a JSONL, rebuilds Hop objects from each row's ideal_path, runs the same
polish pass used during generation, and writes a new JSONL. The original is
backed up with a `.before_polish` suffix.

Usage:
    uv run python -m synth.rewrite_questions bench/filterbench/test.jsonl
    uv run python -m synth.rewrite_questions bench/filterbench/test.jsonl --dry-run
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv

from core.console import console
from synth.gold_path_generation import Hop, polish_question

load_dotenv()


def _to_hops(ideal_path: list[dict]) -> list[Hop]:
    return [
        Hop(
            step=step.get("step", i + 1),
            query=step.get("query", ""),
            filters=step.get("filters") or {},
            expected=step.get("expected", ""),
        )
        for i, step in enumerate(ideal_path)
    ]


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Print rewrites without saving.")
def main(path: Path, dry_run: bool):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    client = anthropic.Anthropic()

    usage = {"sonnet_in": 0, "sonnet_out": 0}
    rewritten = 0
    unchanged = 0
    for row in rows:
        hops = _to_hops(row["ideal_path"])
        draft = row["question"]
        polished = polish_question(client, hops, draft, usage_tracker=usage)
        if polished.strip() != draft.strip():
            rewritten += 1
            console.print(f"[bold cyan]{row['id']}[/bold cyan] ({row['hop_count']}h, {row['filter_count']}f)")
            console.print(f"  [dim]before:[/dim] {draft}")
            console.print(f"  [green]after: [/green] {polished}")
            console.print()
            row["question"] = polished
        else:
            unchanged += 1

    cost = (usage["sonnet_in"] * 3.00 + usage["sonnet_out"] * 15.00) / 1_000_000

    if dry_run:
        console.print(
            f"[dim]DRY RUN — {rewritten} rewritten, {unchanged} unchanged. "
            f"Cost: ${cost:.3f}[/dim]"
        )
        return

    backup = path.with_suffix(path.suffix + ".before_polish")
    backup.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows if False) + "")
    # Proper backup: copy original bytes, not serialized-rows (avoids reordering diffs).
    backup.write_bytes(path.read_bytes())
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    console.print(
        f"[bold]{rewritten}/{len(rows)} rewritten[/bold] · {unchanged} unchanged · "
        f"cost ${cost:.3f}"
    )
    console.print(f"[dim]Backup: {backup}[/dim]")
    console.print(f"[dim]Wrote:  {path}[/dim]")


if __name__ == "__main__":
    main()
