"""Trace schema — what the searcher emits for a single task.

The Trace is the interface between stages:
- Harness writes it when it finishes a task.
- Synth saves it as training data for SFT.
- Bench runner hands it to the extractor → grader.
- (Later) RL reward reads it to score retrieval quality.

`source_bank` is a first-class field, not an afterthought. Issue 007-6 in the
prior repo: the searcher's summaries weren't serialized, so downstream stages
had to re-fetch with the wrong query. Never again.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.types import SourceInfo


@dataclass
class SubmittedUrl:
    """One entry from the searcher's final `submit` call."""
    url: str
    score: float


@dataclass
class TraceMetadata:
    search_count: int = 0
    nudge_count: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""  # "submit" | "no_tool" | "nudge_exhausted" | "error"


@dataclass
class Trace:
    """Full record of one searcher run."""

    task_idx: int
    question: str
    messages: list[dict]  # raw conversation (assistant/user/tool_results)
    source_bank: dict[str, dict]  # url -> SourceInfo.to_dict()
    submitted: list[SubmittedUrl] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)
    # Optional identity info for downstream filtering / analysis.
    agent: str = ""
    searcher_model: str = ""
    # Optional: pre-synthesized answer from a non-harness searcher (e.g. exa_deep).
    synthesis: str = ""

    def to_dict(self) -> dict:
        return {
            "task_idx": self.task_idx,
            "question": self.question,
            "messages": self.messages,
            "source_bank": self.source_bank,
            "submitted": [asdict(s) for s in self.submitted],
            "metadata": asdict(self.metadata),
            "agent": self.agent,
            "searcher_model": self.searcher_model,
            "synthesis": self.synthesis,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Messages may contain Anthropic SDK Pydantic objects; serialize them
        # through model_dump / __dict__ on the fly so the JSON is portable.
        def _default(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=_default),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Trace":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            task_idx=raw["task_idx"],
            question=raw["question"],
            messages=raw["messages"],
            source_bank=raw.get("source_bank", {}),
            submitted=[SubmittedUrl(**s) for s in raw.get("submitted", [])],
            metadata=TraceMetadata(**raw.get("metadata", {})),
            agent=raw.get("agent", ""),
            searcher_model=raw.get("searcher_model", ""),
            synthesis=raw.get("synthesis", ""),
        )

    def sources_in_order(self) -> list[SourceInfo]:
        """Return SourceInfo in the order the searcher submitted them.

        Submitted URLs that aren't in source_bank are dropped (extractor/
        grader won't have content for them anyway).
        """
        out = []
        for s in self.submitted:
            entry = self.source_bank.get(s.url)
            if entry:
                out.append(SourceInfo(**entry))
        return out
