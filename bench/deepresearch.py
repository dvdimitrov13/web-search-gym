"""DeepResearch Bench II loader.

Vendored at `vendor/DeepResearch-Bench-II/tasks_and_rubrics.jsonl` (132 tasks,
zh+en, 22 themes). Each task has a research prompt and a grading rubric with
`info_recall` and `analysis` dimensions — graded by a Gemini-based scorer in
the vendor repo (see `vendor/DeepResearch-Bench-II/run_evaluation.py`).

This loader is read-only: it maps rows to Task objects for inspection /
filtering. End-to-end grading is not wired — our current agents produce
short-answer output, not markdown research reports. To run the bench you
still need: a report-writing agent + the vendor Gemini evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.types import Task

_REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TASKS = _REPO_ROOT / "vendor" / "DeepResearch-Bench-II" / "tasks_and_rubrics.jsonl"


def load_tasks(
    language: str | None = None,
    theme: str | None = None,
    indices: list[int] | None = None,
) -> list[Task]:
    """Load DeepResearch-Bench-II tasks, optionally filtered.

    - `language`: "en" or "zh" — restrict to one language.
    - `theme`: e.g. "Finance & Business" — restrict to one theme.
    - `indices`: keep only these `idx` values (1-based in the source file).

    Task.question carries the research prompt; Task.answer is None (the bench
    grades against a rubric, not a single golden answer). Rubric + content +
    provenance live in `Task.metadata`.
    """
    if not VENDOR_TASKS.exists():
        raise FileNotFoundError(
            f"Missing {VENDOR_TASKS}. Did you clone the submodule? "
            "`git submodule update --init --recursive`"
        )

    tasks: list[Task] = []
    keep = set(indices) if indices is not None else None
    with open(VENDOR_TASKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if language and row.get("language") != language:
                continue
            if theme and row.get("theme") != theme:
                continue
            idx = row["idx"]
            if keep is not None and idx not in keep:
                continue
            meta = {k: v for k, v in row.items() if k != "prompt"}
            tasks.append(Task(idx=idx, question=row["prompt"], answer=None, metadata=meta))
    return tasks
