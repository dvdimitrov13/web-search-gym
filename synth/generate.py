"""Generate searcher trajectories by running the canonical harness over a
question set. Traces are saved as JSON, ready for SFT conversion.

The teacher model (default: Claude Sonnet) drives the harness. Because synth,
harness, and inference all share the same `core/` code, there is zero prompt
or schema drift between data generation and training/inference — the whole
class of Issue 007 bugs is prevented structurally.

Invocation:
    python -m synth.generate --questions path/to/q.jsonl --model claude_sonnet
    python -m synth.generate --questions q.jsonl --model claude_sonnet --indices 0,1,2

Questions file format (JSONL, one per line):
    {"idx": <int>, "question": "<text>"}
    Optional extras (metadata, topic) are carried into the Trace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console

from agents.registry import load_model_config
from core.harness import SearcherHarness
from core.types import Task


console = Console()

_REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORIES_DIR = _REPO_ROOT / "trajectories"


def _load_questions(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(
                Task(
                    idx=row["idx"],
                    question=row["question"],
                    metadata={k: v for k, v in row.items() if k not in {"idx", "question"}},
                )
            )
    return tasks


def _build_harness(model_cfg: dict) -> SearcherHarness:
    return SearcherHarness(
        searcher_model=model_cfg["model_name"],
        provider=model_cfg.get("provider", "anthropic"),
        base_url=model_cfg.get("base_url", ""),
        api_key_env=model_cfg.get("api_key_env", ""),
        temperature=model_cfg.get("temperature", 0.2),
        thinking_budget=model_cfg.get("thinking_budget"),
        verbose=True,
    )


def generate(
    questions_path: Path,
    model_name: str,
    output_dir: Path | None = None,
    indices: list[int] | None = None,
) -> int:
    """Run the harness over all questions; save Traces. Returns #completed."""
    tasks = _load_questions(questions_path)
    if indices is not None:
        keep = set(indices)
        tasks = [t for t in tasks if t.idx in keep]

    model_cfg = load_model_config(model_name)
    harness = _build_harness(model_cfg)

    out_dir = output_dir or (TRAJECTORIES_DIR / f"synth_{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Generating {len(tasks)} trajectories with "
        f"[bold]{model_cfg['model_name']}[/bold] → {out_dir}"
    )

    completed = 0
    t0 = time.time()
    for task in tasks:
        try:
            trace = harness.run(task)
            trace.agent = "synth"
            trace.save(out_dir / f"idx-{task.idx}.json")
            completed += 1
            console.print(
                f"  [green]idx-{task.idx}[/green] "
                f"({trace.metadata.search_count} searches, "
                f"{len(trace.source_bank)} sources)"
            )
        except Exception as e:
            console.print(f"  [red]idx-{task.idx} failed:[/red] {e}")
    console.print(
        f"\n[bold]Done:[/bold] {completed}/{len(tasks)} trajectories in {time.time() - t0:.1f}s"
    )
    return completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True, help="JSONL of {idx, question}")
    p.add_argument("--model", required=True, help="Model config name from models/*.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--indices", default=None, help="Comma-separated idx subset")
    args = p.parse_args()

    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]

    generate(
        questions_path=args.questions,
        model_name=args.model,
        output_dir=args.output_dir,
        indices=indices,
    )


if __name__ == "__main__":
    main()
