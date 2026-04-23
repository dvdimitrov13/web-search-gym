"""The canonical searcher loop. Model-agnostic (any Anthropic-compatible endpoint).

Responsibilities:
- Multi-turn tool-calling loop over `exa_search` / `scratchpad` / `submit`.
- Budget enforcement (search count, scratchpad tokens).
- Nudging the model back into tool use if it stops without calling a tool.
- Producing a `Trace` with full messages + source_bank + metadata for downstream
  stages (synth, sft, bench, rl).

The loop never depends on the specific benchmark shape — it ends when the
searcher calls `submit` and hands off a source bank. BrowseComp's short-answer
format is produced by `core/extractor.py` in a separate stage.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date

import anthropic

from core.console import console
from core.context import (
    estimate_tokens,
    exa_api_block,
    live_state_block,
)
from core.exa_client import ExaClient
from core.prompts import (
    SEARCHER_PROMPT,
    SEARCHER_PROMPT_NO_SCRATCHPAD,
    THINKING_INSTRUCTION,
)
from core.tools import ANTHROPIC_TOOLS
from core.trace import SubmittedUrl, Trace, TraceMetadata
from core.types import RetryableAgentError, SourceInfo, Task

_RETRY_DELAYS = [15, 30, 45]
_OPENROUTER_BASE = "https://openrouter.ai/api"


# ── Client construction ─────────────────────────────────────────────


def make_client(
    provider: str,
    base_url: str = "",
    api_key_env: str = "",
) -> anthropic.Anthropic:
    """Create an Anthropic-compatible client.

    - `anthropic` (default): native Anthropic.
    - `openrouter`: routes to OpenRouter with the Anthropic-compatible schema.
    - Custom `base_url` (e.g. a vLLM proxy): uses a fake API key if the env
      var isn't set.
    """
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
    """Sync call with backoff on rate-limit / connection errors."""
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


# ── Scratchpad edit primitive ───────────────────────────────────────


def _fuzzy_replace(text: str, old: str, new: str) -> tuple[str, bool]:
    """Replace `old` in `text` with relaxed whitespace matching.

    Three tiers, in order:
    1. Exact substring match.
    2. Whitespace-normalized match, mapped back to original text.
    3. rapidfuzz partial-ratio alignment with 95% similarity floor.

    Returns (new_text, matched).
    """
    if old in text:
        return text.replace(old, new, 1), True

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    norm_old = normalize(old)
    norm_text = normalize(text)
    idx = norm_text.find(norm_old)

    if idx != -1:
        # Map normalized index back to original positions.
        char_map: list[int] = []
        in_ws = False
        norm_pos = 0
        for oi, c in enumerate(text):
            if c in " \t\n\r":
                if not in_ws and norm_pos > 0:
                    char_map.append(oi)
                    norm_pos += 1
                in_ws = True
            else:
                in_ws = False
                char_map.append(oi)
                norm_pos += 1

        if idx < len(char_map):
            orig_start = char_map[idx]
            end_idx = idx + len(norm_old)
            if end_idx <= len(char_map):
                orig_end = char_map[end_idx - 1] + 1
            elif char_map:
                orig_end = char_map[-1] + 1
            else:
                orig_end = len(text)
            return text[:orig_start] + new + text[orig_end:], True

    # Tier 3: rapidfuzz
    try:
        from rapidfuzz import fuzz
        alignment = fuzz.partial_ratio_alignment(old, text, score_cutoff=95)
        if (
            alignment is not None
            and alignment.score >= 95
            and alignment.dest_end > alignment.dest_start
        ):
            return (
                text[: alignment.dest_start] + new + text[alignment.dest_end :],
                True,
            )
    except ImportError:
        pass

    return text, False


# ── Harness ─────────────────────────────────────────────────────────


class SearcherHarness:
    """Multi-turn Exa-search loop.

    Instantiate once, call `run(task)` per task. Stateless across tasks.
    """

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
        scratchpad_max_tokens: int = 512,
        results_per_query: int = 5,
        max_nudges: int = 5,
        max_shrink_attempts: int = 2,
        use_scratchpad: bool = True,
        verbose: bool = False,
        exa_client: ExaClient | None = None,
        exa_search_type: str | None = None,
    ):
        self.client = make_client(provider, base_url=base_url, api_key_env=api_key_env)
        self.searcher_model = searcher_model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.thinking_instruction = thinking_instruction
        self.thinking_passthrough = thinking_passthrough
        self.force_tools = force_tools
        self.use_scratchpad = use_scratchpad

        self.max_searches = max_searches
        self.scratchpad_max_tokens = scratchpad_max_tokens
        self.max_nudges = max_nudges
        self.max_shrink_attempts = max_shrink_attempts
        self.verbose = verbose

        self.exa = exa_client or ExaClient(
            num_results=results_per_query, search_type=exa_search_type,
        )

    # ── Public entry point ──────────────────────────────────────────

    def run(self, task: Task) -> Trace:
        """Run the searcher on a single task. Always returns a Trace.

        Raises RetryableAgentError only if the scratchpad stays over budget
        for more than max_shrink_attempts consecutive forced-shrink turns —
        a stuck-loop condition the runner should retry once from scratch.
        """
        t0 = time.time()

        system = self._build_system()
        messages: list[dict] = [
            {"role": "user", "content": f"Research task:\n\n{task.question}"}
        ]

        scratchpad = ""
        # Cycle-based budget: `search_count` is CYCLES USED (assistant turns
        # that issued ≥1 non-error exa_search). Parallel exa_search calls
        # within one turn share a single cycle. `searches_issued` is the raw
        # count of individual calls, kept for trace metadata & cost analysis.
        search_count = 0
        searches_issued = 0
        nudge_count = 0
        submitted: list[SubmittedUrl] = []
        source_bank: dict[str, dict] = {}
        must_shrink = False
        shrink_attempts = 0
        stop_reason = "no_tool"

        max_iterations = self.max_searches * 3 + self.max_nudges

        for _ in range(max_iterations):
            call_messages = self._inject_live_state(
                messages, search_count, scratchpad, searches_issued=searches_issued,
            )

            # When the scratchpad is over budget, restrict tools to scratchpad
            # only until the model shrinks it back in range. When the scratchpad
            # is disabled entirely, strip it from the tool list.
            base_tools = (
                [t for t in ANTHROPIC_TOOLS if t["name"] != "commit_memory"]
                if not self.use_scratchpad
                else ANTHROPIC_TOOLS
            )
            turn_tools = (
                [t for t in ANTHROPIC_TOOLS if t["name"] == "commit_memory"]
                if must_shrink
                else base_tools
            )

            response = self._call_llm(system, call_messages, turn_tools)

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
                # Nudge the model back into tool use.
                nudge_count += 1
                if nudge_count > self.max_nudges:
                    stop_reason = "nudge_exhausted"
                    break
                messages.append(self._assistant_message(response))
                messages.append({
                    "role": "user",
                    "content": (
                        "You MUST use a tool. Do not respond with text. "
                        "Call exa_search to find sources, or call submit "
                        "if you have collected enough URLs."
                    ),
                })
                if self.verbose:
                    console.print(
                        f"  [yellow]Nudging model to use tools "
                        f"({nudge_count}/{self.max_nudges})…[/yellow]"
                    )
                continue

            messages.append(self._assistant_message(response))
            nudge_count = 0  # reset on successful tool call

            tool_results, done, updates = self._dispatch_tools(
                response,
                search_count=search_count,
                searches_issued=searches_issued,
                scratchpad=scratchpad,
                must_shrink=must_shrink,
                shrink_attempts=shrink_attempts,
                source_bank=source_bank,
                submitted=submitted,
            )
            search_count = updates["search_count"]
            searches_issued = updates["searches_issued"]
            scratchpad = updates["scratchpad"]
            must_shrink = updates["must_shrink"]
            shrink_attempts = updates["shrink_attempts"]

            messages.append({"role": "user", "content": tool_results})

            if done:
                stop_reason = "submit"
                break

        elapsed = time.time() - t0

        return Trace(
            task_idx=task.idx,
            question=task.question,
            messages=messages,
            source_bank=source_bank,
            submitted=submitted,
            metadata=TraceMetadata(
                search_count=search_count,
                searches_issued=searches_issued,
                nudge_count=nudge_count,
                elapsed_seconds=round(elapsed, 2),
                stop_reason=stop_reason,
            ),
            searcher_model=self.searcher_model,
        )

    # ── Prompt / message building ───────────────────────────────────

    def _build_system(self) -> str:
        """System prompt: SEARCHER_PROMPT + thinking_instruction + exa_api block.

        Static across turns (no live state) so prompt caching can kick in.
        """
        prompt_template = SEARCHER_PROMPT if self.use_scratchpad else SEARCHER_PROMPT_NO_SCRATCHPAD
        system = prompt_template.format(
            date=date.today().isoformat(),
            max_searches=self.max_searches,
        )
        if self.thinking_instruction:
            system = f"{system}\n\n{THINKING_INSTRUCTION}"
        system += f"\n\n{exa_api_block()}"
        return system

    def _inject_live_state(
        self,
        messages: list[dict],
        search_count: int,
        scratchpad: str,
        *,
        searches_issued: int | None = None,
    ) -> list[dict]:
        """Append live <budget>+<scratchpad> text to the last user message.

        Mutates a per-call copy; does not modify `messages` in place (so older
        turns don't accumulate stale scratchpad copies).
        """
        live = live_state_block(
            search_count=search_count,
            max_searches=self.max_searches,
            scratchpad=scratchpad,
            scratchpad_max_tokens=self.scratchpad_max_tokens,
            label="search cycles",
            searches_issued=searches_issued,
        )
        call_messages = list(messages)
        # Strip thinking blocks from older assistant turns so thinking stays
        # momentary (Sonnet re-thinks per turn instead of carrying prior
        # deliberation). The full thinking is preserved in `messages` — which
        # is what gets saved to the Trace — so this only affects what the API
        # sees on the next call.
        if not self.thinking_passthrough:
            call_messages = [
                {
                    "role": m["role"],
                    "content": (
                        [b for b in m["content"]
                         if not (isinstance(b, dict) and b.get("type") == "thinking")
                         and not (hasattr(b, "type") and getattr(b, "type", None) == "thinking")]
                        if m["role"] == "assistant" and isinstance(m.get("content"), list)
                        else m.get("content")
                    ),
                }
                for m in call_messages
            ]
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

    def _call_llm(self, system: str, messages: list[dict], tools: list[dict]):
        kwargs = dict(
            model=self.searcher_model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=messages,
            tools=tools,
        )
        if self.thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        return _llm_call(self.client, **kwargs)

    def _assistant_message(self, response) -> dict:
        """Convert an Anthropic response into an assistant message for replay.

        Always preserves thinking blocks so the saved Trace carries the full
        reasoning for debugging. `thinking_passthrough=False` strips thinking
        only from the API-bound view (see `_inject_live_state`), not from the
        trace record.
        """
        return {"role": "assistant", "content": response.content}

    # ── Tool dispatch ───────────────────────────────────────────────

    def _dispatch_tools(
        self,
        response,
        *,
        search_count: int,
        searches_issued: int,
        scratchpad: str,
        must_shrink: bool,
        shrink_attempts: int,
        source_bank: dict[str, dict],
        submitted: list[SubmittedUrl],
    ) -> tuple[list[dict], bool, dict]:
        """Execute all tool_use blocks from a response. Returns (tool_results, done, updates).

        Cycle semantics: `search_count` = cycles used. Multiple parallel
        exa_search blocks in ONE response all see the same pre-dispatch
        `search_count` in `_handle_exa`, so they all succeed or all fail
        against the same budget gate. After dispatch, if any non-error
        exa_search was issued, `search_count` increments by exactly 1.
        """
        tool_results: list[dict] = []
        done = False
        any_exa_success = False

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            if block.name == "exa_search":
                tool_results.append(
                    self._handle_exa(
                        block, search_count=search_count, source_bank=source_bank,
                    )
                )
                if not tool_results[-1].get("is_error"):
                    any_exa_success = True
                    searches_issued += 1

            elif block.name == "commit_memory":
                tr, scratchpad, must_shrink, shrink_attempts = (
                    self._handle_scratchpad(
                        block,
                        scratchpad=scratchpad,
                        must_shrink=must_shrink,
                        shrink_attempts=shrink_attempts,
                    )
                )
                tool_results.append(tr)

            # Disabled — kept for a future re-enablement pass.
            # elif block.name == "prune":
            #     urls = block.input.get("urls") or []
            #     reason = (block.input.get("reason") or "").strip()
            #     dropped = [u for u in urls if u in source_bank]
            #     for u in dropped:
            #         source_bank.pop(u, None)
            #     tool_results.append({
            #         "type": "tool_result",
            #         "tool_use_id": block.id,
            #         "content": (
            #             f"Pruned {len(dropped)}/{len(urls)} URL(s) from source bank. "
            #             f"{len(source_bank)} source(s) remain."
            #             + (f" Reason: {reason}" if reason else "")
            #         ),
            #     })
            #     if self.verbose:
            #         console.print(
            #             f"  [yellow]Prune:[/yellow] {len(dropped)}/{len(urls)} URLs "
            #             f"([dim]{reason[:80]}[/dim])"
            #         )

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
                if self.verbose:
                    console.print(
                        f"  [green]Submit:[/green] {len(urls)} URL(s)"
                    )
                done = True

        # One cycle = one response containing ≥1 non-error exa_search.
        if any_exa_success:
            search_count += 1

        return tool_results, done, {
            "search_count": search_count,
            "searches_issued": searches_issued,
            "scratchpad": scratchpad,
            "must_shrink": must_shrink,
            "shrink_attempts": shrink_attempts,
        }

    def _handle_exa(
        self,
        block,
        *,
        search_count: int,
        source_bank: dict[str, dict],
    ) -> dict:
        if search_count >= self.max_searches:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": (
                    f"REJECTED: Search-cycle budget exhausted "
                    f"({search_count}/{self.max_searches} cycles used). "
                    "You must call submit now with your collected URLs."
                ),
                "is_error": True,
            }
        query = block.input.get("query", "")
        summary_query = block.input.get("summary_query")
        # Everything except the two query fields becomes an Exa filter.
        filters = {
            k: v for k, v in block.input.items()
            if k not in ("query", "summary_query")
        }
        try:
            results = self.exa.search(query, summary_query=summary_query, **filters)
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Search error: {e}",
                "is_error": True,
            }
        # Update source bank (first-seen wins for duplicate URLs).
        for s in results.sources:
            if s.url not in source_bank:
                source_bank[s.url] = s.to_dict()
        if self.verbose:
            console.print(
                f"  [cyan]Search (cycle {search_count + 1}/{self.max_searches}):[/cyan] "
                f"[dim]{query[:90]}[/dim] ({len(results.sources)} results)"
            )
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": results.formatted(),
        }

    def _handle_scratchpad(
        self,
        block,
        *,
        scratchpad: str,
        must_shrink: bool,
        shrink_attempts: int,
    ) -> tuple[dict, str, bool, int]:
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
                    f"(over by ~{over_by}). Re-read the <scratchpad> block and "
                    "copy old_text exactly."
                )
            )
            content = (
                f"{outcome} "
                "You MUST shrink the scratchpad before doing anything else. "
                "Your next turn will only have `scratchpad` available — all other "
                f"tools are disabled until you get back under "
                f"{self.scratchpad_max_tokens} tokens. "
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
            f"Scratchpad {op} successfully "
            f"(~{candidate_tokens}/{self.scratchpad_max_tokens} tokens used). "
            "See the <scratchpad> block at the end of the user message for current contents."
        )
        return (
            {"type": "tool_result", "tool_use_id": block.id, "content": content},
            scratchpad,
            must_shrink,
            shrink_attempts,
        )

    # ── Convenience: normalize Trace sources into ordered SourceInfo list ──

    @staticmethod
    def sources_from_trace(trace: Trace) -> list[SourceInfo]:
        """Return SourceInfo in the order the searcher submitted them."""
        return trace.sources_in_order()
