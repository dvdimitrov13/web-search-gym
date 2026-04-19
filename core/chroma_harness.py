"""Chroma-style multi-turn searcher harness.

Tools: search / grep / prune / submit (see core/chroma_tools.py).

Differences from SearcherHarness:
- Search returns chunked highlights (not per-URL summaries).
- grep regex-searches across all surfaced chunks this run.
- prune edits the past tool_result history in place: pruned URLs' chunk blocks
  become `[pruned: <reason>]` placeholders on the next API call. This is the
  "self-editing context" the Chroma Context-1 paper describes.
- No scratchpad. Thinking is the only interior state.

Shared with SearcherHarness:
- thinking_passthrough handling via in-place strip in `_inject_live_state`.
- force_tools nudging.
- Client construction (Anthropic / OpenRouter / custom).
- Extended thinking kwargs.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date
from typing import Any

import anthropic

from core.chroma_tools import ANTHROPIC_CHROMA_TOOLS
from core.console import console
from core.context import exa_api_block
from core.exa_client import ExaClient, HighlightChunk
from core.harness import _RETRY_DELAYS, _llm_call, make_client
from core.prompts import CHROMA_SEARCHER_PROMPT, THINKING_INSTRUCTION
from core.trace import SubmittedUrl, Trace, TraceMetadata
from core.types import RetryableAgentError, SourceInfo, Task


def _estimate_tokens(text: str) -> int:
    # Rough 4-char-per-token heuristic (same as core/context.estimate_tokens).
    return max(1, len(text) // 4)


class ChromaHarness:
    """Chroma-style search/grep/prune harness."""

    def __init__(
        self,
        searcher_model: str,
        provider: str = "anthropic",
        base_url: str = "",
        api_key_env: str = "",
        temperature: float = 0.2,
        thinking_budget: int | None = None,
        thinking_instruction: bool = True,
        thinking_passthrough: bool = True,
        force_tools: bool = False,
        max_searches: int = 5,
        num_results_per_search: int = 5,
        num_sentences_per_highlight: int = 3,
        highlights_per_url: int = 5,
        max_nudges: int = 5,
        verbose: bool = False,
        exa_client: ExaClient | None = None,
    ):
        self.client = make_client(provider, base_url=base_url, api_key_env=api_key_env)
        self.searcher_model = searcher_model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.thinking_instruction = thinking_instruction
        self.thinking_passthrough = thinking_passthrough
        self.force_tools = force_tools
        self.max_searches = max_searches
        self.num_results_per_search = num_results_per_search
        self.num_sentences_per_highlight = num_sentences_per_highlight
        self.highlights_per_url = highlights_per_url
        self.max_nudges = max_nudges
        self.verbose = verbose
        self.exa = exa_client or ExaClient()

    def run(self, task: Task) -> Trace:
        t0 = time.time()

        system = self._build_system()
        messages: list[dict] = [
            {"role": "user", "content": f"Research task:\n\n{task.question}"}
        ]

        # Chunk store: each chunk gets an id (chunk-{N}) and tracks URL.
        # live_chunks: the set of chunk_ids still visible in the self-edited
        # context. Pruning moves chunk_ids out of live_chunks; history is
        # rewritten on the next `_inject_live_state` call.
        all_chunks: dict[str, HighlightChunk] = {}
        live_chunks: set[str] = set()
        pruned_urls: dict[str, str] = {}  # url -> reason

        search_count = 0
        nudge_count = 0
        submitted: list[SubmittedUrl] = []
        stop_reason = "no_tool"

        max_iterations = self.max_searches * 3 + self.max_nudges

        for _ in range(max_iterations):
            call_messages = self._prepare_call_messages(
                messages, live_chunks, pruned_urls, search_count
            )
            response = self._call_llm(system, call_messages, ANTHROPIC_CHROMA_TOOLS)

            if self.verbose:
                for b in response.content:
                    if getattr(b, "type", None) == "thinking":
                        text = getattr(b, "thinking", "")
                        preview = " ".join(text.split())[:220]
                        if preview:
                            console.print(f"  [magenta]💭 {preview}…[/magenta]")

            has_tool_use = any(
                getattr(b, "type", None) == "tool_use" for b in response.content
            )
            if not has_tool_use:
                if not self.force_tools:
                    stop_reason = "no_tool"
                    break
                nudge_count += 1
                if nudge_count > self.max_nudges:
                    stop_reason = "nudge_exhausted"
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You MUST use a tool. Call search/grep/prune or "
                        "submit if you have enough URLs."
                    ),
                })
                continue

            messages.append({"role": "assistant", "content": response.content})
            nudge_count = 0

            tool_results, done, deltas = self._dispatch_tools(
                response,
                all_chunks=all_chunks,
                live_chunks=live_chunks,
                pruned_urls=pruned_urls,
                search_count=search_count,
                submitted=submitted,
            )
            search_count = deltas["search_count"]
            messages.append({"role": "user", "content": tool_results})
            if done:
                stop_reason = "submit"
                break

        elapsed = time.time() - t0

        # Build the source bank from the URLs the agent submitted. For each
        # submitted URL, aggregate its live chunks into a summary block so the
        # downstream extractor gets the actually-useful text.
        source_bank: dict[str, dict] = {}
        for cid, c in all_chunks.items():
            if cid not in live_chunks:
                continue
            source_bank.setdefault(
                c.url,
                {"url": c.url, "title": c.title, "summary": "", "published": c.published},
            )
        # Concatenate chunks per URL into the summary field.
        for cid, c in all_chunks.items():
            if cid not in live_chunks:
                continue
            entry = source_bank[c.url]
            entry["summary"] = (entry["summary"] + "\n\n" + c.text).strip()

        return Trace(
            task_idx=task.idx,
            question=task.question,
            messages=messages,
            source_bank=source_bank,
            submitted=submitted,
            metadata=TraceMetadata(
                search_count=search_count,
                elapsed_seconds=round(elapsed, 2),
                stop_reason=stop_reason,
            ),
            agent="chroma_agent",
            searcher_model=self.searcher_model,
        )

    # ── Context assembly ────────────────────────────────────────────

    def _build_system(self) -> str:
        system = CHROMA_SEARCHER_PROMPT.format(
            date=date.today().isoformat(),
            max_searches=self.max_searches,
        )
        if self.thinking_instruction:
            system = f"{system}\n\n{THINKING_INSTRUCTION}"
        system += f"\n\n{exa_api_block()}"
        return system

    def _prepare_call_messages(
        self,
        messages: list[dict],
        live_chunks: set[str],
        pruned_urls: dict[str, str],
        search_count: int,
    ) -> list[dict]:
        """Build the per-call message list with live-state + pruned-chunk edits.

        Two transformations applied in order:
        1. Strip thinking blocks from older assistant turns (if passthrough off).
        2. Rewrite past tool_result contents so any chunk whose id is no longer
           in live_chunks becomes a `[pruned: <reason>]` placeholder.
        """
        call_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "assistant" and isinstance(content, list) and not self.thinking_passthrough:
                content = [
                    b for b in content
                    if not (isinstance(b, dict) and b.get("type") == "thinking")
                    and not (hasattr(b, "type") and getattr(b, "type", None) == "thinking")
                ]
            call_messages.append({"role": role, "content": content})

        # Rewrite tool_result chunk blocks per live_chunks.
        call_messages = [self._rewrite_tool_results(m, live_chunks, pruned_urls) for m in call_messages]

        # Inject live budget block into the last user message.
        budget = self._budget_block(search_count, live_chunks)
        if call_messages and call_messages[-1].get("role") == "user":
            last = dict(call_messages[-1])
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = [
                    {"type": "text", "text": content},
                    {"type": "text", "text": budget},
                ]
            else:
                last["content"] = list(content) + [{"type": "text", "text": budget}]
            call_messages[-1] = last
        else:
            call_messages.append({"role": "user", "content": budget})
        return call_messages

    def _rewrite_tool_results(
        self, msg: dict, live_chunks: set[str], pruned_urls: dict[str, str]
    ) -> dict:
        """If this is a user message carrying tool_result blocks, rewrite their
        chunk placeholders so pruned chunk_ids become `[pruned]` markers."""
        if msg.get("role") != "user":
            return msg
        content = msg.get("content")
        if not isinstance(content, list):
            return msg
        new_content = []
        for b in content:
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                new_content.append(b)
                continue
            text = b.get("content", "")
            if not isinstance(text, str):
                new_content.append(b)
                continue
            rewritten = self._strip_pruned_chunks(text, live_chunks, pruned_urls)
            new_content.append({**b, "content": rewritten})
        return {**msg, "content": new_content}

    _CHUNK_BLOCK_RE = re.compile(
        r"\[chunk-(\d+) url=(.+?)\]\n(.+?)(?=\n\[chunk-|\Z)",
        re.DOTALL,
    )

    def _strip_pruned_chunks(
        self, text: str, live_chunks: set[str], pruned_urls: dict[str, str]
    ) -> str:
        def replace(match: re.Match) -> str:
            cid = f"chunk-{match.group(1)}"
            url = match.group(2)
            if cid in live_chunks:
                return match.group(0)
            reason = pruned_urls.get(url, "pruned")
            return f"[chunk-{match.group(1)} url={url} — PRUNED: {reason}]"
        return self._CHUNK_BLOCK_RE.sub(replace, text)

    def _budget_block(self, search_count: int, live_chunks: set[str]) -> str:
        remaining = self.max_searches - search_count
        return (
            "<budget>\n"
            f"searches: {search_count}/{self.max_searches} used "
            f"({remaining} remaining)\n"
            f"live chunks: {len(live_chunks)}\n"
            "</budget>"
        )

    def _call_llm(self, system: str, messages: list[dict], tools: list[dict]):
        kwargs: dict[str, Any] = dict(
            model=self.searcher_model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=messages,
            tools=tools,
        )
        if self.thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        return _llm_call(self.client, **kwargs)

    # ── Tool dispatch ───────────────────────────────────────────────

    def _dispatch_tools(
        self,
        response,
        *,
        all_chunks: dict[str, HighlightChunk],
        live_chunks: set[str],
        pruned_urls: dict[str, str],
        search_count: int,
        submitted: list[SubmittedUrl],
    ) -> tuple[list[dict], bool, dict]:
        tool_results: list[dict] = []
        done = False

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            if block.name == "search":
                if search_count >= self.max_searches:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"REJECTED: search budget exhausted "
                            f"({search_count}/{self.max_searches}). Use grep "
                            "across what you have, prune, or submit."
                        ),
                        "is_error": True,
                    })
                    continue
                query = block.input.get("query", "")
                filters = {k: v for k, v in block.input.items() if k != "query"}
                try:
                    results = self.exa.search_highlights(
                        query,
                        num_results=self.num_results_per_search,
                        num_sentences=self.num_sentences_per_highlight,
                        highlights_per_url=self.highlights_per_url,
                        **filters,
                    )
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Search error: {e}",
                        "is_error": True,
                    })
                    continue
                # Register chunks in the chunk store.
                new_ids: list[str] = []
                for c in results.chunks:
                    cid = f"chunk-{len(all_chunks)}"
                    all_chunks[cid] = c
                    live_chunks.add(cid)
                    new_ids.append(cid)
                text = self._format_chunks(results.query, new_ids, all_chunks)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                })
                search_count += 1
                if self.verbose:
                    console.print(
                        f"  [cyan]Search {search_count}/{self.max_searches}:[/cyan] "
                        f"[dim]{query[:90]}[/dim] ({len(new_ids)} chunks)"
                    )

            elif block.name == "grep":
                pattern = block.input.get("pattern", "")
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Invalid regex: {e}",
                        "is_error": True,
                    })
                    continue
                matches: list[str] = []
                for cid, c in all_chunks.items():
                    if cid not in live_chunks:
                        continue
                    if regex.search(c.text):
                        matches.append(self._format_one_chunk(cid, c))
                        if len(matches) >= 5:
                            break
                text = (
                    "No matches.\n"
                    if not matches
                    else f"Found {len(matches)} match(es):\n\n" + "\n\n".join(matches)
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                })
                if self.verbose:
                    console.print(
                        f"  [yellow]Grep:[/yellow] [dim]{pattern[:60]}[/dim] "
                        f"({len(matches)} match(es))"
                    )

            elif block.name == "prune":
                urls = block.input.get("urls") or []
                reason = (block.input.get("reason") or "").strip()
                dropped = 0
                for cid, c in list(all_chunks.items()):
                    if c.url in urls and cid in live_chunks:
                        live_chunks.discard(cid)
                        dropped += 1
                for u in urls:
                    pruned_urls[u] = reason or "off-topic"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": (
                        f"Pruned {dropped} chunk(s) from {len(urls)} URL(s). "
                        f"{len(live_chunks)} chunk(s) live."
                        + (f" Reason: {reason}" if reason else "")
                    ),
                })
                if self.verbose:
                    console.print(
                        f"  [red]Prune:[/red] {dropped} chunks from {len(urls)} URLs "
                        f"[dim]{reason[:60]}[/dim]"
                    )

            elif block.name == "submit":
                urls = block.input.get("urls", [])
                for u in urls:
                    submitted.append(
                        SubmittedUrl(url=u.get("url", ""), score=u.get("score", 0.0))
                    )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "URLs submitted. Research complete.",
                })
                done = True
                if self.verbose:
                    console.print(f"  [green]Submit:[/green] {len(urls)} URL(s)")

        return tool_results, done, {"search_count": search_count}

    def _format_chunks(
        self, query: str, new_ids: list[str], all_chunks: dict[str, HighlightChunk]
    ) -> str:
        if not new_ids:
            return f"No chunks returned for query {query!r}."
        parts = [f"{len(new_ids)} chunk(s) for query {query!r}:"]
        for cid in new_ids:
            c = all_chunks[cid]
            parts.append(self._format_one_chunk(cid, c))
        return "\n\n".join(parts)

    def _format_one_chunk(self, cid: str, c: HighlightChunk) -> str:
        return (
            f"[{cid} url={c.url}]\n"
            f"Title: {c.title}\n"
            f"Text: {c.text}"
        )
