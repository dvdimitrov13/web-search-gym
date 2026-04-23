"""Drive agent_dd across a task set and save enriched traces for SFT.

Traces include `turn_states` (per-turn live-state + tool-availability
snapshots) so `sft/convert.py --mode agent-dd-per-turn` can emit exact-
fidelity per-step training samples — each sample matches what the agent
actually saw at that step.

Unlike `bench.cli run`, this script does NOT grade answers. The grade is
useful for filtering trajectories AFTER generation (see
`synth/filter_traces.py`), but synthesis should run cheaply without
Gemini judge calls on every task.

Three task sources, selectable via `--source`:
  - jsonl         — read `{idx, question}` lines from a file
  - deepsearchqa  — DSQA splits: dev | smoke | domain2 | full
  - filterbench   — filterbench splits: dev | test
  - browsecomp    — BrowseComp splits from bench/splits.yaml

Usage:
    python -m synth.generate_agent_dd \\
        --model claude_sonnet \\
        --source deepsearchqa --split domain2 \\
        --concurrent 4

Traces land in `trajectories/agent_dd/idx-{N}.json` as the agent's normal
writeout — no dedicated output dir is needed.
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from agents.registry import load_agent
from core.types import Task

console = Console()

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tasks(source: str, path: Path | None, split: str | None, indices: list[int] | None) -> list[Task]:
    if source == "jsonl":
        if path is None:
            raise SystemExit("--path required for --source jsonl")
        import json
        tasks: list[Task] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tasks.append(Task(
                    idx=row["idx"],
                    question=row["question"],
                    answer=row.get("answer"),
                    metadata={k: v for k, v in row.items() if k not in {"idx", "question", "answer"}},
                ))
        if indices is not None:
            keep = set(indices)
            tasks = [t for t in tasks if t.idx in keep]
        return tasks

    if source == "deepsearchqa":
        from bench.deepsearchqa import load_tasks
        return load_tasks(split=split or "domain2", indices=indices)

    if source == "filterbench":
        from bench.filterbench import load_tasks
        return load_tasks(split=split or "test", indices=indices)

    if source == "browsecomp":
        import yaml
        from bench.browsecomp import load_tasks
        splits_yaml = _REPO_ROOT / "bench" / "splits.yaml"
        with open(splits_yaml) as f:
            split_data = yaml.safe_load(f) or {}
        split_indices = split_data.get(split or "dev", {}).get("indices")
        return load_tasks(indices=indices if indices is not None else split_indices)

    raise SystemExit(f"Unknown source: {source}")


def _run_one(agent, task: Task, dataset_tag: str) -> tuple[int, str | None]:
    """Call agent.answer(task). The agent saves its own trace. Returns (idx, err)."""
    # Tag dataset on metadata so agent_dd's per-task code paths behave the
    # same way they would under `bench.cli run` (e.g. extractor-skip logic).
    task.metadata["dataset"] = dataset_tag
    try:
        agent.answer(task)
        return task.idx, None
    except Exception as e:
        return task.idx, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def generate(
    *,
    agent_name: str,
    model_name: str,
    source: str,
    path: Path | None,
    split: str | None,
    indices: list[int] | None,
    concurrent: int,
    dataset_tag: str,
) -> int:
    tasks = _load_tasks(source, path, split, indices)
    if not tasks:
        console.print("[yellow]No tasks to process.[/yellow]")
        return 0

    agent = load_agent(agent_name, model=model_name)
    agent.setup()

    console.print(
        f"Synthesizing {len(tasks)} trajectories with "
        f"[bold]{agent.display_name}[/bold] / [bold]{model_name}[/bold] "
        f"@ concurrency={concurrent}\n"
    )

    t0 = time.time()
    completed = 0
    errors: list[tuple[int, str]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("trajectories", total=len(tasks))

        if concurrent <= 1:
            for task in tasks:
                idx, err = _run_one(agent, task, dataset_tag)
                if err:
                    errors.append((idx, err))
                else:
                    completed += 1
                progress.update(bar, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=concurrent) as ex:
                futures = {
                    ex.submit(_run_one, agent, task, dataset_tag): task
                    for task in tasks
                }
                for fut in as_completed(futures):
                    idx, err = fut.result()
                    if err:
                        errors.append((idx, err))
                    else:
                        completed += 1
                    progress.update(bar, advance=1)

    agent.teardown()
    elapsed = time.time() - t0

    console.print(
        f"\n[bold]Done:[/bold] {completed}/{len(tasks)} trajectories "
        f"in {elapsed:.1f}s"
        + (f" · [red]{len(errors)} errors[/red]" if errors else "")
    )
    if errors:
        for idx, err in errors[:5]:
            console.print(f"  [red]idx-{idx}:[/red] {err.splitlines()[0]}")
    return completed


def main():
    p = argparse.ArgumentParser(description="Agent-driven trajectory synthesis for SFT")
    p.add_argument("--agent", default="agent_dd", help="Agent name (default: agent_dd)")
    p.add_argument("--model", required=True, help="Model config name from models/*.yaml")
    p.add_argument(
        "--source",
        choices=["jsonl", "deepsearchqa", "filterbench", "browsecomp"],
        default="deepsearchqa",
    )
    p.add_argument("--path", type=Path, default=None, help="JSONL path (for --source jsonl)")
    p.add_argument("--split", default=None, help="Split name within the source")
    p.add_argument("--indices", default=None, help="Comma-separated idx subset")
    p.add_argument("--concurrent", type=int, default=4)
    p.add_argument(
        "--dataset-tag", default="",
        help=(
            "Tag written to task.metadata['dataset']. Some agents branch on "
            "this (e.g. extractor skip for DSQA). Defaults to the --source name."
        ),
    )
    args = p.parse_args()

    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]

    dataset_tag = args.dataset_tag or args.source
    generate(
        agent_name=args.agent,
        model_name=args.model,
        source=args.source,
        path=args.path,
        split=args.split,
        indices=indices,
        concurrent=args.concurrent,
        dataset_tag=dataset_tag,
    )


if __name__ == "__main__":
    main()
