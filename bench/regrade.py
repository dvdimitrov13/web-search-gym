"""Re-grade previously recorded agent answers with the current judge prompt.

Useful when the grader prompt changes and you want to update scores without
re-running the agents. Reads raw JSONL records (produced by `bench.runner`),
re-invokes the judge on the saved response text, overwrites the per-task
`is_correct` / `judge_raw` fields in place, and writes a fresh aggregate.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click
from dotenv import load_dotenv

from bench.browsecomp import Grader, answer_to_browsecomp_text
from core.console import console
from core.types import Answer

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"


def _render_response(agent_answer: dict) -> str:
    if not agent_answer:
        return ""
    ans = Answer(
        explanation=agent_answer.get("explanation", ""),
        exact_answer=agent_answer.get("exact_answer", ""),
        confidence=int(agent_answer.get("confidence", 0) or 0),
    )
    return answer_to_browsecomp_text(ans)


@click.command()
@click.argument("raw_path", type=click.Path(exists=True, path_type=Path))
def main(raw_path: Path):
    """Re-grade every record in a raw JSONL file using the current judge."""
    grader = Grader()
    records = []
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    console.print(f"[bold]{raw_path.name}[/bold]: re-grading {len(records)} records…")
    flipped = 0
    correct = 0
    for rec in records:
        if rec.get("error"):
            continue
        response = _render_response(rec.get("agent_answer") or {})
        if not response.strip():
            continue
        new = grader.grade(rec["question"], rec.get("correct_answer", ""), response)
        prev = bool(rec.get("is_correct"))
        if new.is_correct != prev:
            flipped += 1
            verb = "[green]→ yes[/green]" if new.is_correct else "[red]→ no[/red]"
            console.print(
                f"  flip idx={rec['idx']} {verb} "
                f"gold={rec.get('correct_answer', '')!r} "
                f"got={(rec.get('agent_answer') or {}).get('exact_answer', '')!r}"
            )
        rec["is_correct"] = new.is_correct
        rec["judge_raw"] = new.raw_judge_output
        if new.is_correct:
            correct += 1

    # Overwrite raw file in place.
    with raw_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Rewrite aggregate file.
    stem = raw_path.stem
    agg_path = RESULTS_DIR / "scores" / f"{stem}.json"
    if agg_path.exists():
        agg = json.loads(agg_path.read_text())
    else:
        agg = {"agent": "?", "run_id": "?"}
    agg["tasks"] = len(records)
    agg["correct"] = correct
    agg["accuracy"] = correct / max(len(records), 1)
    agg["errors"] = sum(1 for r in records if r.get("error"))
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(agg, indent=2))

    console.print(
        f"[bold]{correct}/{len(records)} = {agg['accuracy']:.3f}[/bold] "
        f"({flipped} flips). → {agg_path}"
    )


if __name__ == "__main__":
    main()
