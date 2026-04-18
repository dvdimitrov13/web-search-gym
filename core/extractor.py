"""Short-answer extractor — produces BrowseComp's required answer format.

Contract: given a question and a source bank (url → summary), emit:

    Explanation: ...
    Exact Answer: ...
    Confidence: 0-100

Why a separate stage (not "writer"):
- BrowseComp's grader regex-extracts `Exact Answer:` and checks it against the
  gold answer. A long-form writer's score would be dominated by synthesis
  quality rather than whether the searcher surfaced the right fact.
- Running a cheap, single-shot extractor (e.g. Haiku) keeps the benchmark
  signal dominated by retrieval — if the fact isn't in the source bank, no
  extractor can recover it.

Implementation notes:
- Uses the Anthropic SDK because it supports OpenAI-compatible endpoints too
  (via `base_url`), so a Qwen3-served extractor can plug in unchanged.
- No tools — single turn, text-only output.
"""

from __future__ import annotations

import os
import re

import anthropic

from core.prompts import EXTRACTOR_PROMPT
from core.types import Answer, SourceInfo


_OPENROUTER_BASE = "https://openrouter.ai/api"


def _make_client(provider: str, base_url: str = "", api_key_env: str = "") -> anthropic.Anthropic:
    """Create an Anthropic-compatible client for the given provider."""
    if provider == "openrouter":
        return anthropic.Anthropic(
            base_url=_OPENROUTER_BASE,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if base_url:
        return anthropic.Anthropic(
            base_url=os.path.expandvars(base_url),
            api_key=os.environ.get(api_key_env, "none") if api_key_env else "none",
        )
    return anthropic.Anthropic()


def _render_sources(sources: list[SourceInfo]) -> str:
    if not sources:
        return "(no sources found)"
    return "\n\n=====\n\n".join(
        f"Source: {s.title}\n"
        f"URL: {s.url}\n"
        f"Published: {s.published}\n"
        f"Summary: {s.summary}"
        for s in sources
    )


_EXACT_ANSWER_RE = re.compile(r"^Exact Answer:\s*(.+?)\s*$", re.MULTILINE)
_EXPLANATION_RE = re.compile(
    r"^Explanation:\s*(.+?)(?=^Exact Answer:|\Z)", re.MULTILINE | re.DOTALL,
)
_CONFIDENCE_RE = re.compile(r"^Confidence:\s*(\d{1,3})", re.MULTILINE)


def parse_answer(raw: str) -> Answer:
    """Parse the extractor's raw text output into an Answer.

    Be defensive: if any field is missing we fall back to the full output
    as explanation + empty exact answer + 0 confidence so the grader can still
    see what happened.
    """
    exact_m = _EXACT_ANSWER_RE.search(raw)
    expl_m = _EXPLANATION_RE.search(raw)
    conf_m = _CONFIDENCE_RE.search(raw)

    explanation = expl_m.group(1).strip() if expl_m else raw.strip()
    exact = exact_m.group(1).strip() if exact_m else ""
    try:
        confidence = int(conf_m.group(1)) if conf_m else 0
    except ValueError:
        confidence = 0
    confidence = max(0, min(100, confidence))
    return Answer(explanation=explanation, exact_answer=exact, confidence=confidence)


class Extractor:
    """Single-shot answer extractor."""

    def __init__(
        self,
        model_name: str,
        provider: str = "anthropic",
        base_url: str = "",
        api_key_env: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.client = _make_client(provider, base_url=base_url, api_key_env=api_key_env)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(
        self,
        question: str,
        sources: list[SourceInfo],
        synthesis: str = "",
    ) -> Answer:
        """Format a BrowseComp answer from sources and/or a pre-synthesized string.

        When `synthesis` is provided (e.g. from Exa deep's `output`), it is the
        primary input and the per-source summaries become optional grounding.
        When `synthesis` is empty (the lean_searcher path), behaviour is
        unchanged — sources are the only input.
        """
        parts = [f"Question:\n{question}"]
        if synthesis:
            parts.append(f"Research synthesis:\n{synthesis}")
        if sources:
            parts.append(f"Sources:\n{_render_sources(sources)}")
        elif not synthesis:
            parts.append("Sources:\n(no sources found)")
        parts.append("Answer in the exact format specified.")
        user = "\n\n".join(parts)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=EXTRACTOR_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
        return parse_answer(raw)
