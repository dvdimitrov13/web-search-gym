"""The single Exa entry point.

Two contracts:
- `search()` — summary mode. Always uses `contents={"summary": {"query": ...}}`.
  The summary query defaults to the search query, but can be overridden via
  `summary_query` for broad-retrieve / narrow-summarize patterns. Both forms
  are part of the lean_searcher training/inference contract (the tool schema
  exposes `summary_query`, so training data captures whichever the model used).
  Every summary query is wrapped with `_SUMMARY_DETAIL_PREAMBLE` before
  hitting Exa, which pushes Gemini Flash into extractive enumeration (dense,
  verbatim, list-preserving) rather than a condensed paragraph — empirically
  ~2x denser summaries on multi-item pages. This wrapper is fixed and part
  of the contract; the model's `summary_query` becomes `{query}` inside it.
  What's NOT allowed: switching the `contents` shape itself (e.g. swapping
  summary for highlights). That mismatch was Issue 007-5 in the prior repo.
- `search_highlights()` — eval-only, returns chunked highlights per URL. Used
  by chroma_agent's search/grep/prune harness. NOT part of the lean_searcher
  training contract; do not call from synth/lean.

Returns a `SearchResults` dataclass with already-parsed `SourceInfo` objects,
so downstream code never touches the raw Exa SDK response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from exa_py import AsyncExa, Exa

from core.types import SourceInfo

# Extractive-enumeration preamble wrapped around every `contents.summary.query`
# before it hits Gemini Flash. Empirically (see experiment on DSQA idx=737)
# this roughly doubles per-URL summary density vs. asking "summarize for {q}"
# alone — Gemini flips from a generic paragraph into itemized extraction of
# names, quantities, dates, list items, step details. Set Answer tasks (65%
# of DSQA) were under-recalling because summaries collapsed multi-item pages
# to one representative fact; this preamble pushes the summarizer to enumerate
# every page-stated specific relevant to the query.
_SUMMARY_DETAIL_PREAMBLE = (
    "Extract every specific fact on this page (named entities, quantities, "
    "dates, numbers, list items, step details) relevant to the question "
    "below. Enumerate verbatim — do NOT abstract or paraphrase. If the page "
    "contains a list, reproduce the list. If the page has a step-by-step "
    "procedure, include each step. Prefer completeness over brevity.\n\n"
    "Question: {query}"
)


@dataclass
class HighlightChunk:
    """One reranked highlight block extracted from a URL."""

    url: str
    title: str
    text: str
    published: str = "unknown"


@dataclass
class HighlightResults:
    """Chunked search results for the chroma_agent / agent_dd paths."""

    query: str
    chunks: list[HighlightChunk]
    filters: dict = field(default_factory=dict)
    # url -> full readable page text. Populated when the Exa call requests
    # `contents.text` alongside highlights. Used by agent_dd's
    # browse_page tool — lets us cache the page at search time and drop
    # the Jina Reader round-trip on browse.
    pages: dict[str, str] = field(default_factory=dict)


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

    def __init__(
        self,
        api_key: str | None = None,
        num_results: int = 5,
        search_type: str | None = None,
    ):
        key = api_key or os.environ["EXA_API_KEY"]
        self._exa = Exa(key)
        # Lazy — only instantiated when an `_async` method is first used.
        self._exa_async: AsyncExa | None = None
        self._api_key = key
        self.num_results = num_results
        # None → let Exa pick (currently `auto`). Set to e.g. "instant" / "fast"
        # / "neural" / "keyword" to pin the retrieval backend. `deep*` tiers go
        # through `deep_search()`, not here.
        self.search_type = search_type

    @property
    def async_exa(self) -> AsyncExa:
        if self._exa_async is None:
            self._exa_async = AsyncExa(self._api_key)
        return self._exa_async

    def search(
        self,
        query: str,
        summary_query: str | None = None,
        **filters,
    ) -> SearchResults:
        """Run a summary-mode search. Extra kwargs become Exa filters.

        `query` shapes URL retrieval (what Exa ranks against).
        `summary_query` shapes per-URL summaries (what the Gemini Flash
        summarizer focuses on). Defaults to `query` when omitted so single-
        query callers behave as before. Training data captures both fields,
        so split-query calls are part of the contract, not a shim.
        """
        summary_q = summary_query if summary_query else query
        # Always wrap with the extractive-detail preamble — every summary goes
        # through Gemini Flash, and we want dense enumerated output by default
        # so Set Answer / multi-item tasks don't lose gold items to abstraction.
        summary_q_wrapped = _SUMMARY_DETAIL_PREAMBLE.format(query=summary_q)
        sdk_kwargs: dict = {
            "num_results": self.num_results,
            "contents": {"summary": {"query": summary_q_wrapped}},
        }
        if self.search_type:
            sdk_kwargs["type"] = self.search_type
        raw = self._exa.search(query, **sdk_kwargs, **filters)
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

    def _build_highlights_kwargs(
        self,
        query: str,
        *,
        num_results: int,
        num_sentences: int,
        max_characters: int | None,
        highlights_per_url: int,
        highlights_query: str | None,
        include_text: bool,
        text_max_chars: int | None,
    ) -> dict:
        hq = highlights_query or query
        # `maxCharacters` is the preferred knob as of Feb 2026 (composes better
        # across diverse page structures than a sentence count).
        highlight_spec: dict = {
            "highlights_per_url": highlights_per_url,
            "query": hq,
        }
        if max_characters is not None:
            highlight_spec["maxCharacters"] = max_characters
        else:
            highlight_spec["num_sentences"] = num_sentences

        contents_spec: dict = {"highlights": highlight_spec}
        if include_text:
            # Exa returns `text` alongside highlights in one call. We cache
            # this per-URL so a downstream browse_page tool can extract from
            # memory without a second network round-trip.
            if text_max_chars is not None:
                contents_spec["text"] = {"maxCharacters": text_max_chars}
            else:
                contents_spec["text"] = True

        sdk_kwargs: dict = {
            "num_results": num_results,
            "contents": contents_spec,
        }
        if self.search_type:
            sdk_kwargs["type"] = self.search_type
        return sdk_kwargs

    @staticmethod
    def _parse_highlight_response(
        raw, query: str, include_text: bool, filters: dict,
    ) -> HighlightResults:
        chunks: list[HighlightChunk] = []
        pages: dict[str, str] = {}
        for r in raw.results:
            url = r.url
            title = r.title or "(untitled)"
            published = r.published_date or "unknown"
            highlights = getattr(r, "highlights", None) or []
            if include_text:
                page_text = getattr(r, "text", None) or ""
                if page_text:
                    pages[url] = page_text
            if not highlights:
                continue
            for h in highlights:
                text = h if isinstance(h, str) else getattr(h, "text", None) or str(h)
                chunks.append(
                    HighlightChunk(url=url, title=title, text=text, published=published)
                )
        return HighlightResults(
            query=query, chunks=chunks, filters=filters, pages=pages,
        )

    def search_highlights(
        self,
        query: str,
        *,
        num_results: int = 5,
        num_sentences: int = 3,
        max_characters: int | None = None,
        highlights_per_url: int = 5,
        highlights_query: str | None = None,
        include_text: bool = False,
        text_max_chars: int | None = None,
        **filters,
    ) -> HighlightResults:
        """Search Exa and return reranked highlight chunks per URL.

        Used by chroma_agent and agent_dd. Sync counterpart of
        `search_highlights_async` — same semantics.
        """
        sdk_kwargs = self._build_highlights_kwargs(
            query,
            num_results=num_results,
            num_sentences=num_sentences,
            max_characters=max_characters,
            highlights_per_url=highlights_per_url,
            highlights_query=highlights_query,
            include_text=include_text,
            text_max_chars=text_max_chars,
        )
        raw = self._exa.search(query, **sdk_kwargs, **filters)
        return self._parse_highlight_response(raw, query, include_text, filters)

    async def search_highlights_async(
        self,
        query: str,
        *,
        num_results: int = 5,
        num_sentences: int = 3,
        max_characters: int | None = None,
        highlights_per_url: int = 5,
        highlights_query: str | None = None,
        include_text: bool = False,
        text_max_chars: int | None = None,
        **filters,
    ) -> HighlightResults:
        """Async counterpart of `search_highlights`."""
        sdk_kwargs = self._build_highlights_kwargs(
            query,
            num_results=num_results,
            num_sentences=num_sentences,
            max_characters=max_characters,
            highlights_per_url=highlights_per_url,
            highlights_query=highlights_query,
            include_text=include_text,
            text_max_chars=text_max_chars,
        )
        raw = await self.async_exa.search(query, **sdk_kwargs, **filters)
        return self._parse_highlight_response(raw, query, include_text, filters)

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
