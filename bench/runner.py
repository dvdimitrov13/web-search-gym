"""Run an agent on BrowseComp tasks, grade each answer, aggregate the run.

Saves:
- per-task record under results/raw/<agent>__<model_tag>__<run_id>.jsonl
- aggregated scores under results/scores/<agent>__<model_tag>__<run_id>.json

Writes agent-produced Answer records and the judge's verdict side by side so
runs are fully reproducible from the raw file alone.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from agents.base import BaseAgent
from bench.browsecomp import (
    GradeResult,
    Grader,
    answer_to_browsecomp_text,
)
from core.console import console
from core.types import Answer, RetryableAgentError, Task

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"


@dataclass
class TaskRecord:
    idx: int
    question: str
    correct_answer: str
    agent_answer: dict  # Answer.to_dict()
    is_correct: bool
    judge_raw: str
    error: str | None = None
    metrics: dict | None = None  # grader-specific per-task metrics (e.g. DSQA p/r/f1)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "agent_answer": self.agent_answer,
            "is_correct": self.is_correct,
            "judge_raw": self.judge_raw,
            "error": self.error,
            "metrics": self.metrics,
        }


def _answer_with_retry(agent: BaseAgent, task: Task) -> Answer:
    try:
        return agent.answer(task)
    except RetryableAgentError as e:
        console.print(
            f"  [yellow]Task {task.idx}: {e} — retrying once with fresh context[/yellow]"
        )
        return agent.answer(task)


def _grade(grader, task: Task, response_text: str):
    """Grader-agnostic dispatch.

    BrowseComp's Grader.grade(question, answer, response) → GradeResult.
    Other graders (DSQA) take the Task and response directly so they can use
    metadata like `answer_type`. We sniff the signature by trying the Task-
    based form first and falling back to the 3-arg form.
    """
    if hasattr(grader, "grade"):
        try:
            return grader.grade(task, response_text)  # DSQAGrader-style
        except TypeError:
            return grader.grade(task.question, task.answer or "", response_text)
    raise TypeError(f"Grader {type(grader).__name__} has no .grade method")


def _process_task(agent: BaseAgent, task: Task, grader) -> TaskRecord:
    try:
        ans = _answer_with_retry(agent, task)
    except Exception as e:
        return TaskRecord(
            idx=task.idx,
            question=task.question,
            correct_answer=task.answer or "",
            agent_answer={},
            is_correct=False,
            judge_raw="",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    # If the agent provided a prose answer AND the grader prefers prose
    # (DSQA-style), pass that directly; otherwise fall back to the
    # BrowseComp-formatted text that short-answer graders expect.
    prefers_prose = getattr(grader, "prefers_natural_response", False)
    if prefers_prose and ans.natural_text:
        response_text = ans.natural_text
    else:
        response_text = answer_to_browsecomp_text(ans)
    grade = _grade(grader, task, response_text)
    return TaskRecord(
        idx=task.idx,
        question=task.question,
        correct_answer=task.answer or "",
        agent_answer=ans.to_dict(),
        is_correct=grade.is_correct,
        judge_raw=grade.raw_judge_output,
        metrics=getattr(grade, "metrics", None),
    )


def _model_tag(agent: BaseAgent) -> str:
    """Short string describing the models the agent is using.

    Used in filenames so different model combos don't collide.
    """
    configs = getattr(agent, "model_configs", {}) or {}
    parts = []
    for stage in ("searcher", "extractor"):
        cfg = configs.get(stage)
        if cfg:
            parts.append(f"{stage}-{cfg.get('name', '?')}")
    return "__".join(parts) if parts else "default"


def run(
    agent: BaseAgent,
    tasks: list[Task],
    *,
    run_id: str,
    max_concurrent: int = 1,
    grader=None,
) -> dict:
    """Run the agent over tasks and grade each one. Returns an aggregate dict."""
    if grader is None:
        grader = Grader()

    model_tag = _model_tag(agent)
    stem = f"{agent.name}__{model_tag}__{run_id}"
    raw_path = RESULTS_DIR / "raw" / f"{stem}.jsonl"
    agg_path = RESULTS_DIR / "scores" / f"{stem}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Running [bold]{agent.display_name}[/bold] "
        f"([dim]{model_tag}[/dim]) on {len(tasks)} tasks, "
        f"{max_concurrent} concurrent…\n"
    )

    agent.setup()
    start_time = time.time()
    records: list[TaskRecord] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("graded", total=len(tasks))

        def _finish_one(rec: TaskRecord):
            records.append(rec)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            progress.update(bar, advance=1)

        if max_concurrent <= 1:
            for task in tasks:
                _finish_one(_process_task(agent, task, grader))
        else:
            with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
                futures = {
                    ex.submit(_process_task, agent, task, grader): task
                    for task in tasks
                }
                for fut in as_completed(futures):
                    _finish_one(fut.result())

    agent.teardown()
    elapsed = time.time() - start_time

    correct = sum(1 for r in records if r.is_correct)
    errors = [asdict(r) for r in records if r.error]
    aggregate = {
        "agent": agent.name,
        "model_tag": model_tag,
        "run_id": run_id,
        "tasks": len(records),
        "correct": correct,
        "accuracy": correct / max(len(records), 1),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
    }
    # Dataset-specific grader aggregation (e.g. DSQA precision/recall/F1).
    if hasattr(grader, "aggregate"):
        try:
            aggregate["grader_metrics"] = grader.aggregate([r.to_dict() for r in records])
        except Exception as e:
            console.print(f"[yellow]grader.aggregate failed: {e}[/yellow]")

    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    console.print(
        f"\n[bold]Accuracy:[/bold] {aggregate['accuracy']:.3f} "
        f"({correct}/{len(records)}) — {elapsed:.1f}s"
        + (f" · {aggregate['errors']} errors" if aggregate["errors"] else "")
    )
    gm = aggregate.get("grader_metrics")
    if gm and "f1" in gm:
        console.print(
            f"[bold]F1:[/bold] {gm['f1']:.3f} · "
            f"[bold]Precision:[/bold] {gm['precision']:.3f} · "
            f"[bold]Recall:[/bold] {gm['recall']:.3f}"
        )
    console.print(f"[dim]Raw: {raw_path}[/dim]")
    console.print(f"[dim]Aggregate: {agg_path}[/dim]")
    return aggregate
