"""The single Exa entry point. Always summary mode, never highlights.

Mismatch between training-time content format and inference-time content format
was Issue 007-5 in the prior repo. This module exists to make that regression
impossible: every caller (synth, harness, bench) calls `search` here, and it
always passes `contents={"summary": {"query": query}}`.

Returns a `SearchResults` dataclass with already-parsed `SourceInfo` objects,
so downstream code never touches the raw Exa SDK response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from exa_py import Exa

from core.types import SourceInfo


@dataclass
class SearchResults:
    """What the harness sees from one exa_search call."""

    query: str
    sources: list[SourceInfo]
    # Optional: the filters the searcher requested (for trace/debug).
    filters: dict = field(default_factory=dict)
    # Optional: deep-tier pre-synthesized answer (populated by deep_search).
    synthesis: str = ""

    def formatted(self) -> str:
        """Human-readable formatting for insertion into a tool_result."""
        if not self.sources:
            return "No results found."
        return "\n\n---\n\n".join(
            f"Title: {s.title}\n"
            f"URL: {s.url}\n"
            f"Published: {s.published}\n"
            f"Summary: {s.summary}"
            for s in self.sources
        )


class ExaClient:
    """Thin wrapper around `exa_py.Exa`. Owns the summary-mode contract."""

    def __init__(self, api_key: str | None = None, num_results: int = 5):
        self._exa = Exa(api_key or os.environ["EXA_API_KEY"])
        self.num_results = num_results

    def search(self, query: str, **filters) -> SearchResults:
        """Run a summary-mode search. Extra kwargs become Exa filters."""
        # We intentionally do NOT pass `highlights` or omit `contents`.
        # Training data uses summaries; inference must match.
        raw = self._exa.search(
            query,
            num_results=self.num_results,
            contents={"summary": {"query": query}},
            **filters,
        )
        sources = [
            SourceInfo(
                url=r.url,
                title=r.title or "(untitled)",
                summary=r.summary or "(no summary)",
                published=r.published_date or "unknown",
            )
            for r in raw.results
        ]
        return SearchResults(query=query, sources=sources, filters=filters)

    def deep_search(self, query: str, tier: str) -> SearchResults:
        """Run a multi-step deep tier — 'deep-lite' | 'deep' | 'deep-reasoning'.

        Exa's deep tiers do their own multi-hop query plan + synthesis. We ask
        for the synthesized answer via `output_schema` (text shape) and pass
        that back as `SearchResults.synthesis` — the downstream extractor uses
        the synthesis as its primary input rather than per-page summaries.

        We intentionally skip `contents` here: the summary-mode contract in
        `search()` exists to keep lean-searcher training and inference aligned;
        `deep_search` is eval-only and never feeds the training pipeline, so
        using the synthesis path instead of summary-mode creates no mismatch.
        """
        raw = self._exa.search(
            query,
            type=tier,
            output_schema={
                "type": "text",
                "description": (
                    "A concise, factual answer to the question. Include the "
                    "specific value requested (name, number, date, phrase, "
                    "etc.) plus a brief explanation of the supporting evidence."
                ),
            },
        )
        synthesis = ""
        output = getattr(raw, "output", None)
        if output is not None:
            content = getattr(output, "content", output)
            synthesis = content if isinstance(content, str) else str(content)
        sources = [
            SourceInfo(
                url=r.url,
                title=r.title or "(untitled)",
                summary="",  # deep_search skips per-page summary; synthesis is primary
                published=getattr(r, "published_date", None) or "unknown",
            )
            for r in raw.results
        ]
        return SearchResults(
            query=query,
            sources=sources,
            filters={"type": tier},
            synthesis=synthesis,
        )

    def backfill(self, urls: list[str], query: str) -> list[SourceInfo]:
        """Fetch summaries for URLs that weren't returned by search.

        Only used as a fallback when a submit contains URLs missing from the
        source bank (which shouldn't normally happen; cheap safety net).
        """
        if not urls:
            return []
        raw = self._exa.get_contents(urls, summary={"query": query})
        return [
            SourceInfo(
                url=r.url,
                title=r.title or "(untitled)",
                summary=r.summary or "(no summary)",
                published=r.published_date or "unknown",
            )
            for r in raw.results
        ]
