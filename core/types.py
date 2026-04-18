"""Shared dataclasses used across the gym."""

from dataclasses import dataclass, field


@dataclass
class Task:
    """A single question for a searcher agent to answer.

    `idx` + `question` are the minimum. Benchmarks can carry extra metadata
    via the `metadata` dict (e.g., BrowseComp's `canary`, `answer`, topic).
    """

    idx: int
    question: str
    answer: str | None = None  # ground truth (BrowseComp provides this)
    metadata: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """Alias: callers that already speak 'prompt' can use this."""
        return self.question


@dataclass
class SourceInfo:
    """A single source discovered by the searcher."""

    url: str
    title: str
    summary: str
    published: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "published": self.published,
        }


@dataclass
class Answer:
    """The extractor's final answer to a BrowseComp-style task."""

    explanation: str
    exact_answer: str
    confidence: int  # 0-100

    def to_dict(self) -> dict:
        return {
            "explanation": self.explanation,
            "exact_answer": self.exact_answer,
            "confidence": self.confidence,
        }


class RetryableAgentError(Exception):
    """Transient stuck-loop condition. Runner should retry once with fresh context."""
