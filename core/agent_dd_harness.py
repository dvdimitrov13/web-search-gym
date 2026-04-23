"""Multi-turn agent_dd harness: search → browse_page → answer.

Single-agent shape — the model drives retrieval AND synthesis in one
rollout, then calls `answer` to end. No separate extractor stage.

Key differences from `core/harness.py` (lean_searcher):
- Tools: search / browse_page / answer (not exa_search / commit_memory / submit)
- Search returns HIGHLIGHT CHUNKS (extractive, maxCharacters=200), not
  per-URL summaries.
- Citations are first-class: each chunk + browse result gets a short id
  (`S_*` / `B_*`) surfaced to the model for `answer.citations`.
- No scratchpad. The model's own thinking is the working memory.
- Exit condition: `answer` tool call. Final answer is extracted from the
  tool_use input directly — no downstream extractor.

Cycle-based budget: one cycle = one assistant turn with ≥1 non-error
search or browse_page call. Parallel calls in one turn share a cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import date

import anthropic

from core.browse import HaikuBrowseExtractor
from core.console import console
from core.context import estimate_tokens, live_state_block
from core.exa_client import ExaClient
from core.harness import _fuzzy_replace  # reuse lean's scratchpad fuzzy matcher
from core.agent_dd_prompts import AGENT_DD_SYSTEM_PROMPT
from core.agent_dd_tools import AGENT_DD_ANTHROPIC_TOOLS
from core.trace import SubmittedUrl, Trace, TraceMetadata
from core.types import Answer, RetryableAgentError, Task

_RETRY_DELAYS = [15, 30, 45]
_OPENROUTER_BASE = "https://openrouter.ai/api"


def _make_client(
    provider: str,
    base_url: str = "",
    api_key_env: str = "",
) -> anthropic.Anthropic:
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


def _llm_call(client, **kwargs):
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            console.print(
                f"  [yellow]Retry {attempt + 1}/{len(_RETRY_DELAYS)}: "
                f"{type(e).__name__}, waiting {delay}s…[/yellow]"
            )
            time.sleep(delay)
    return client.messages.create(**kwargs)


def _short_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


class AgentDDHarness:
    """Single-rollout DR Tulu-style harness (search / browse_page / answer)."""

    def __init__(
        self,
        searcher_model: str,
        provider: str = "anthropic",
        base_url: str = "",
        api_key_env: str = "",
        temperature: float = 0.2,
        thinking_budget: int | None = None,
        thinking_passthrough: bool = True,
        max_cycles: int = 5,
        results_per_query: int = 5,
        highlight_max_chars: int = 200,
        highlights_per_url: int = 3,
        exa_search_type: str | None = "instant",
        browse_extractor_model: str = "claude-haiku-4-5-20251001",
        browse_extractor_max_tokens: int = 320,
        scratchpad_max_tokens: int = 1024,
        max_shrink_attempts: int = 2,
        max_nudges: int = 3,
        verbose: bool = False,
        exa_client: ExaClient | None = None,
    ):
        self.client = _make_client(provider, base_url=base_url, api_key_env=api_key_env)
        self.searcher_model = searcher_model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.thinking_passthrough = thinking_passthrough
        self.max_cycles = max_cycles
        self.results_per_query = results_per_query
        self.highlight_max_chars = highlight_max_chars
        self.highlights_per_url = highlights_per_url
        self.scratchpad_max_tokens = scratchpad_max_tokens
        self.max_shrink_attempts = max_shrink_attempts
        self.max_nudges = max_nudges
        self.verbose = verbose

        self.exa = exa_client or ExaClient(
            num_results=results_per_query, search_type=exa_search_type,
        )
        self.browse_extractor = HaikuBrowseExtractor(
            model=browse_extractor_model,
            max_tokens=browse_extractor_max_tokens,
        )

    # ── Public entry ────────────────────────────────────────────────

    def run(self, task: Task) -> tuple[Trace, Answer]:
        """Run one rollout. Returns (Trace, final Answer)."""
        t0 = time.time()

        system = AGENT_DD_SYSTEM_PROMPT.format(
            date=date.today().isoformat(),
            max_cycles=self.max_cycles,
        )
        messages: list[dict] = [
            {"role": "user", "content": f"Research task:\n\n{task.question}"},
        ]

        # Source bank keyed by snippet id. Values carry url/title/text/type.
        source_bank: dict[str, dict] = {}
        # Per-task page cache: url → full-page text captured at search time.
        # Read by browse_page so we never round-trip to Jina for a URL we've
        # already fetched.
        page_cache: dict[str, dict] = {}
        # Persistent scratchpad (commit_memory). Free — no cycle cost.
        scratchpad = ""
        must_shrink = False
        shrink_attempts = 0
        cycle_count = 0
        searches_issued = 0
        browse_count = 0
        nudge_count = 0
        stop_reason = "no_tool"
        final_answer: Answer | None = None

        # One turn per iteration, capped so we can't loop forever even if
        # the model refuses to call `answer`. Generous multiplier because
        # commit_memory turns don't consume cycles but do consume iterations.
        max_iterations = self.max_cycles * 4 + self.max_nudges

        for _ in range(max_iterations):
            call_messages = self._inject_live_state(
                messages, cycle_count, scratchpad, searches_issued=searches_issued,
            )

            # Restrict tools to commit_memory only while the scratchpad is
            # over budget — model must shrink before anything else.
            turn_tools = (
                [t for t in AGENT_DD_ANTHROPIC_TOOLS if t["name"] == "commit_memory"]
                if must_shrink
                else AGENT_DD_ANTHROPIC_TOOLS
            )
            response = self._call_llm(system, call_messages, tools=turn_tools)

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
                nudge_count += 1
                if nudge_count > self.max_nudges:
                    stop_reason = "nudge_exhausted"
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You MUST call a tool. Use `search` or `browse_page` "
                        "to gather evidence, or `answer` to submit your "
                        "final answer if you have enough."
                    ),
                })
                continue

            messages.append({"role": "assistant", "content": response.content})
            nudge_count = 0

            dispatch_out = self._dispatch(
                response,
                task=task,
                cycle_count=cycle_count,
                source_bank=source_bank,
                page_cache=page_cache,
                scratchpad=scratchpad,
                must_shrink=must_shrink,
                shrink_attempts=shrink_attempts,
            )
            tool_results = dispatch_out["tool_results"]
            consumed_cycle = dispatch_out["consumed_cycle"]
            answer_from_tool = dispatch_out["answer"]
            searches_issued += dispatch_out["searches_in_turn"]
            browse_count += dispatch_out["browses_in_turn"]
            scratchpad = dispatch_out["scratchpad"]
            must_shrink = dispatch_out["must_shrink"]
            shrink_attempts = dispatch_out["shrink_attempts"]
            if consumed_cycle:
                cycle_count += 1

            messages.append({"role": "user", "content": tool_results})

            if answer_from_tool is not None:
                final_answer = answer_from_tool
                stop_reason = "answer"
                break

        elapsed = time.time() - t0

        if final_answer is None:
            # Model never called `answer` — synthesize a placeholder so the
            # grader still sees something.
            final_answer = Answer(
                explanation=f"Harness stopped without answer (stop_reason={stop_reason})",
                exact_answer="",
                confidence=0,
            )

        # Build a Trace. We repurpose existing Trace fields:
        # - `source_bank` is keyed by snippet id here (not url).
        # - `submitted` mirrors the citations from the final answer.
        submitted = [
            SubmittedUrl(url=source_bank[sid]["url"], score=1.0)
            for sid in (final_answer.natural_text.split("\n") if False else [])
        ]

        trace = Trace(
            task_idx=task.idx,
            question=task.question,
            messages=messages,
            source_bank=source_bank,
            submitted=submitted,
            metadata=TraceMetadata(
                search_count=cycle_count,
                searches_issued=searches_issued,
                nudge_count=nudge_count,
                elapsed_seconds=round(elapsed, 2),
                stop_reason=stop_reason,
            ),
            searcher_model=self.searcher_model,
            synthesis=final_answer.natural_text,
        )
        return trace, final_answer

    # ── LLM call + message mgmt ────────────────────────────────────

    def _call_llm(self, system: str, messages: list[dict], tools: list[dict] | None = None):
        kwargs = dict(
            model=self.searcher_model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=messages,
            tools=tools if tools is not None else AGENT_DD_ANTHROPIC_TOOLS,
        )
        if self.thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        return _llm_call(self.client, **kwargs)

    def _inject_live_state(
        self,
        messages: list[dict],
        cycle_count: int,
        scratchpad: str,
        *,
        searches_issued: int | None = None,
    ) -> list[dict]:
        """Append live <budget>+<commit_memory> text to the last user message.

        Same pattern as the lean harness: live state sits next to the model's
        generation point, keeping the system prompt cacheable across turns.
        """
        live = live_state_block(
            search_count=cycle_count,
            max_searches=self.max_cycles,
            scratchpad=scratchpad,
            scratchpad_max_tokens=self.scratchpad_max_tokens,
            label="search cycles",
            searches_issued=searches_issued,
        )
        call_messages = self._strip_thinking(messages)
        if call_messages and call_messages[-1].get("role") == "user":
            last = dict(call_messages[-1])
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = [
                    {"type": "text", "text": content},
                    {"type": "text", "text": live},
                ]
            else:
                last["content"] = list(content) + [{"type": "text", "text": live}]
            call_messages[-1] = last
        else:
            call_messages.append({"role": "user", "content": live})
        return call_messages

    def _strip_thinking(self, messages: list[dict]) -> list[dict]:
        """Drop thinking blocks from API-bound assistant history if configured."""
        if self.thinking_passthrough:
            return messages
        out = []
        for m in messages:
            if m["role"] == "assistant" and isinstance(m.get("content"), list):
                out.append({
                    "role": m["role"],
                    "content": [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") == "thinking")
                        and not (hasattr(b, "type") and getattr(b, "type", None) == "thinking")
                    ],
                })
            else:
                out.append(m)
        return out

    # ── Tool dispatch ──────────────────────────────────────────────

    def _dispatch(
        self,
        response,
        *,
        task: Task,
        cycle_count: int,
        source_bank: dict[str, dict],
        page_cache: dict[str, dict],
        scratchpad: str,
        must_shrink: bool,
        shrink_attempts: int,
    ) -> dict:
        """Sync entry: defers to an async dispatch so parallel tool calls
        (search, browse_page) actually execute concurrently rather than
        serializing inside a for-loop."""
        return asyncio.run(self._dispatch_async(
            response,
            task=task,
            cycle_count=cycle_count,
            source_bank=source_bank,
            page_cache=page_cache,
            scratchpad=scratchpad,
            must_shrink=must_shrink,
            shrink_attempts=shrink_attempts,
        ))

    async def _dispatch_async(
        self,
        response,
        *,
        task: Task,
        cycle_count: int,
        source_bank: dict[str, dict],
        page_cache: dict[str, dict],
        scratchpad: str,
        must_shrink: bool,
        shrink_attempts: int,
    ) -> dict:
        """Execute all tool_use blocks from one response.

        Phases (to respect the page_cache → browse dependency):
          1. `commit_memory` calls (synchronous, update scratchpad state).
          2. `search` calls in parallel. They populate page_cache.
          3. `browse_page` calls in parallel. They read page_cache.
          4. `answer` (at most one, synchronous — no I/O).

        Returns a dict with tool_results, consumed_cycle, answer,
        searches_in_turn, browses_in_turn, scratchpad, must_shrink,
        shrink_attempts — the full side-effect state for the run loop.
        """
        # Keep original response-order so tool_results is reconstructed in
        # the same order the model emitted them (cleaner transcript / trace).
        indexed: list[tuple[int, object]] = [
            (i, b) for i, b in enumerate(response.content)
            if getattr(b, "type", None) == "tool_use"
        ]
        scratchpads = [(i, b) for i, b in indexed if b.name == "commit_memory"]
        searches = [(i, b) for i, b in indexed if b.name == "search"]
        browses = [(i, b) for i, b in indexed if b.name == "browse_page"]
        answers = [(i, b) for i, b in indexed if b.name == "answer"]

        budget_exhausted = cycle_count >= self.max_cycles
        results_by_idx: dict[int, dict] = {}
        searches_in_turn = 0
        browses_in_turn = 0
        consumed_cycle = False
        final_answer: Answer | None = None

        # ── Phase 0: commit_memory (sync, no I/O, no cycle) ────────
        for i, b in scratchpads:
            tr, scratchpad, must_shrink, shrink_attempts = self._handle_scratchpad(
                b,
                scratchpad=scratchpad,
                must_shrink=must_shrink,
                shrink_attempts=shrink_attempts,
            )
            results_by_idx[i] = tr

        # ── Phase 1: searches (parallel) ───────────────────────────
        if searches and budget_exhausted:
            for i, b in searches:
                results_by_idx[i] = self._budget_error(b, cycle_count)
        elif searches:
            search_trs = await asyncio.gather(*[
                self._handle_search_async(b, cycle_count, source_bank, page_cache)
                for _, b in searches
            ])
            for (i, _), tr in zip(searches, search_trs):
                results_by_idx[i] = tr
                if not tr.get("is_error"):
                    consumed_cycle = True
                    searches_in_turn += 1

        # ── Phase 2: browses (parallel, run after phase 1 so the page
        #             cache from any same-turn search is visible here).
        # Any turn that issues search OR browse consumes one cycle, so
        # browses are budgeted alongside searches. ──
        if browses and budget_exhausted:
            for i, b in browses:
                results_by_idx[i] = self._budget_error(b, cycle_count)
        elif browses:
            browse_trs = await asyncio.gather(*[
                self._handle_browse_async(b, source_bank, page_cache)
                for _, b in browses
            ])
            for (i, _), tr in zip(browses, browse_trs):
                results_by_idx[i] = tr
                if not tr.get("is_error"):
                    consumed_cycle = True
                    browses_in_turn += 1

        # ── Phase 3: answer (sync, no I/O) ─────────────────────────
        for i, b in answers:
            tr, final_answer = self._handle_answer(b, source_bank)
            results_by_idx[i] = tr

        # Reassemble tool_results in the model's original emission order.
        tool_results = [results_by_idx[i] for i, _ in indexed]
        return {
            "tool_results": tool_results,
            "consumed_cycle": consumed_cycle,
            "answer": final_answer,
            "searches_in_turn": searches_in_turn,
            "browses_in_turn": browses_in_turn,
            "scratchpad": scratchpad,
            "must_shrink": must_shrink,
            "shrink_attempts": shrink_attempts,
        }

    def _handle_answer(self, block, source_bank: dict[str, dict]) -> tuple[dict, Answer]:
        inp = block.input or {}
        final = str(inp.get("final_answer", "")).strip()
        expl = str(inp.get("explanation", "")).strip()
        cites = inp.get("citations") or []
        if not isinstance(cites, list):
            cites = []
        missing = [c for c in cites if c not in source_bank]
        natural = final
        if expl:
            natural = f"{expl}\n\nFinal answer: {final}"
        answer = Answer(
            explanation=expl or final,
            exact_answer=final,
            confidence=80,
            natural_text=natural,
        )
        tr = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": (
                f"Answer accepted. Citations: {len(cites) - len(missing)} "
                f"valid, {len(missing)} invalid."
                + (f" Invalid: {missing}" if missing else "")
            ),
        }
        if self.verbose:
            console.print(
                f"  [green]Answer:[/green] [b]{final[:80]}[/b] "
                f"(citations: {len(cites)}, missing: {len(missing)})"
            )
        return tr, answer

    def _budget_error(self, block, cycle_count: int) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": (
                f"REJECTED: Search-cycle budget exhausted "
                f"({cycle_count}/{self.max_cycles} cycles used). "
                "Call `answer` now with the evidence you have."
            ),
            "is_error": True,
        }

    async def _handle_search_async(
        self,
        block,
        cycle_count: int,
        source_bank: dict[str, dict],
        page_cache: dict[str, dict],
    ) -> dict:
        query = (block.input or {}).get("query", "")
        if not query:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "search error: missing `query`",
                "is_error": True,
            }
        try:
            # Always pull full-page text alongside highlights — no cap —
            # so browse_page can read from the cache without any external
            # fetch. If a page is truly huge, the Haiku extractor caps
            # its own input (max_page_chars in HaikuBrowseExtractor).
            results = await self.exa.search_highlights_async(
                query,
                num_results=self.results_per_query,
                max_characters=self.highlight_max_chars,
                highlights_per_url=self.highlights_per_url,
                include_text=True,
                text_max_chars=None,
            )
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"search error: {e}",
                "is_error": True,
            }

        # Cache full-page text for later browse_page (so we don't need Jina).
        # Title carried alongside so browse can construct a useful header.
        for c in results.chunks:
            if c.url in results.pages and c.url not in page_cache:
                page_cache[c.url] = {
                    "title": c.title,
                    "text": results.pages[c.url],
                }

        # Chunks get snippet ids. Model cites these later.
        lines: list[str] = []
        if not results.chunks:
            lines.append("No highlight chunks returned.")
        else:
            for c in results.chunks:
                sid = _short_id("S", c.url, c.text[:80])
                if sid not in source_bank:
                    source_bank[sid] = {
                        "url": c.url,
                        "title": c.title,
                        "text": c.text,
                        "type": "highlight",
                    }
                lines.append(
                    f"[{sid}] {c.title} — {c.url}\n  \"{c.text}\""
                )

        if self.verbose:
            console.print(
                f"  [cyan]search (cycle {cycle_count + 1}/{self.max_cycles}):[/cyan] "
                f"[dim]{query[:90]}[/dim] ({len(results.chunks)} chunks)"
            )

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": "\n\n".join(lines) if lines else "No results.",
        }

    async def _handle_browse_async(
        self,
        block,
        source_bank: dict[str, dict],
        page_cache: dict[str, dict],
    ) -> dict:
        inp = block.input or {}
        url = inp.get("url", "")
        question = inp.get("question", "")
        if not url or not question:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "browse_page error: both `url` and `question` are required",
                "is_error": True,
            }

        cached = page_cache.get(url)
        if cached is None:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": (
                    "browse_page error: URL is not in the page cache. "
                    "browse_page only works on URLs returned by a prior "
                    "`search` call in this rollout. Search first, then "
                    "browse a URL from the results."
                ),
                "is_error": True,
            }

        try:
            extracted = await self.browse_extractor.extract_async(
                url=url,
                title=cached["title"],
                question=question,
                content=cached["text"],
            )
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"browse_page error: {e}",
                "is_error": True,
            }

        bid = _short_id("B", url, question[:80])
        source_bank[bid] = {
            "url": url,
            "title": cached["title"],
            "text": extracted,
            "type": "browse",
            "question": question,
        }
        if self.verbose:
            console.print(
                f"  [cyan]browse_page:[/cyan] [dim]{url[:80]}[/dim] "
                f"({len(extracted)} chars, cached={len(cached['text'])} chars)"
            )

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": (
                f"[{bid}] {cached['title']} — {url}\n"
                f"(extracted against: \"{question}\")\n\n"
                f"{extracted}"
            ),
        }

    def _handle_scratchpad(
        self,
        block,
        *,
        scratchpad: str,
        must_shrink: bool,
        shrink_attempts: int,
    ) -> tuple[dict, str, bool, int]:
        """Port of lean harness's commit_memory handler. Token-budgeted,
        fuzzy-match edit-in-place. Over-budget → restrict next turn to
        commit_memory only until shrunk."""
        old_text = block.input.get("old_text")
        new_text = block.input.get("new_text", "")

        if old_text is not None:
            candidate, matched = _fuzzy_replace(scratchpad, old_text, new_text)
            op = "edited" if matched else "old_text not found, no change"
        else:
            candidate = new_text
            matched = True
            op = "written"

        candidate_tokens = estimate_tokens(candidate)
        over_budget = candidate_tokens > self.scratchpad_max_tokens
        scratchpad = candidate

        if over_budget:
            if must_shrink and matched:
                shrink_attempts += 1
            elif not must_shrink:
                shrink_attempts = 1
            must_shrink = True
            if shrink_attempts > self.max_shrink_attempts:
                raise RetryableAgentError(
                    f"scratchpad stuck over budget for {self.max_shrink_attempts} "
                    f"consecutive forced-shrink turns "
                    f"(~{candidate_tokens} vs {self.scratchpad_max_tokens} limit)"
                )
            over_by = candidate_tokens - self.scratchpad_max_tokens
            outcome = (
                f"WRITE APPLIED but scratchpad is now "
                f"~{candidate_tokens}/{self.scratchpad_max_tokens} tokens "
                f"(over by ~{over_by})."
                if matched
                else (
                    f"NO-OP: old_text not found, nothing changed. "
                    f"Still ~{candidate_tokens}/{self.scratchpad_max_tokens} tokens "
                    f"(over by ~{over_by}). Re-read the <commit_memory> block and "
                    "copy old_text exactly."
                )
            )
            content = (
                f"{outcome} You MUST shrink the scratchpad before doing "
                "anything else. Your next turn will only have `commit_memory` "
                "available — all other tools are disabled until you get back "
                f"under {self.scratchpad_max_tokens} tokens. "
                "Use old_text + new_text=\"\" to delete sections, or replace "
                "verbose chunks with shorter summaries."
            )
            return (
                {"type": "tool_result", "tool_use_id": block.id, "content": content},
                scratchpad,
                must_shrink,
                shrink_attempts,
            )

        # In budget — clear shrink lock.
        must_shrink = False
        shrink_attempts = 0
        content = (
            f"commit_memory {op} successfully "
            f"(~{candidate_tokens}/{self.scratchpad_max_tokens} tokens used). "
            "See the <commit_memory> block at the end of the user message for current contents."
        )
        if self.verbose:
            console.print(
                f"  [cyan]commit_memory:[/cyan] {op} "
                f"(~{candidate_tokens}/{self.scratchpad_max_tokens} tokens)"
            )
        return (
            {"type": "tool_result", "tool_use_id": block.id, "content": content},
            scratchpad,
            must_shrink,
            shrink_attempts,
        )
