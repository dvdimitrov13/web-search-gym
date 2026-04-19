"""Synthetic filter-use benchmark generator (v3 — iterative chain-builder).

The design lesson from v2: asking a model to generate a multi-hop question
in one turn invites hop inflation (splitting one retrieval into multiple
pseudo-hops). v3 fixes this by building the chain one hop at a time, with a
cold Exa retrievability check after each hop. Structural anti-inflation:
every hop is a real search.

## Flow per question

For each hop i in 1..N:
    Turn runs with tools=[exa_search, submit_hop], tool_choice={"type": "any"}.
    The agent may call exa_search any number of times to explore, then calls
    submit_hop to end the turn with:
        { query, expected_answer, required_exa_filters? }
    On submit_hop:
        - Cold retrievability check: re-run the declared query+filters and
          fuzzy-match expected_answer in the results.
        - Pass → commit hop, advance to i+1.
        - Fail → feed the failure back, let the agent retry the same hop.
After N hops are committed:
    Compose turn: tools=[submit_question], tool_choice forces that tool.
    Agent writes the natural-language question that threads all hops,
    hiding filter names behind organic phrasing.

Final gate: preflight (schema, hop count, canonical filter count, no literal
filter names in question) + one more cold retrievability check on the last
hop to catch compose-time drift. No separate LLM verifier — cold checks +
preflight cover the quality gates.

Cells: (hop_count, filter_count) where hops ∈ {1,2,3,4}, filters ∈ {0,1,2}.

Usage:
    uv run python -m synth.gold_path_generation --mode dev    # 3 varied cells × 1
    uv run python -m synth.gold_path_generation --mode test   # 12 cells × 1
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv

from core.console import console
from core.exa_client import ExaClient

load_dotenv()


_REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _REPO_ROOT / "bench" / "filterbench"

MODEL = "claude-sonnet-4-5"

# filter_count <= hop_count — a hop can carry at most one filtered search,
# so (1h, 2f) is architecturally impossible.
ALL_CELLS: list[tuple[int, int]] = [
    (h, f) for h in (1, 2, 3, 4) for f in (0, 1, 2) if f <= h
]
DEV_CELLS: list[tuple[int, int]] = [(1, 0), (2, 1), (4, 2)]

SEED_TOPICS = [
    "Science & Research",
    "Business & Finance",
    "Technology & Software",
    "Politics & Government",
    "Sports",
    "History",
    "Arts & Entertainment",
    "Health & Medicine",
]


# ── Tool schemas ───────────────────────────────────────────────────────

_FILTER_PROPERTIES = {
    "include_domains": {"type": "array", "items": {"type": "string"}},
    "exclude_domains": {"type": "array", "items": {"type": "string"}},
    "start_published_date": {
        "type": "string",
        "description": "ISO 8601, e.g. 2024-01-01T00:00:00.000Z",
    },
    "end_published_date": {"type": "string"},
    "category": {
        "type": "string",
        "enum": [
            "company",
            "research paper",
            "news",
            "pdf",
            "personal site",
            "financial report",
            "people",
        ],
    },
    "include_text": {"type": "array", "items": {"type": "string"}},
    "exclude_text": {"type": "array", "items": {"type": "string"}},
}

EXA_SEARCH_TOOL = {
    "name": "exa_search",
    "description": (
        "Search the web via Exa neural search. Returns up to 5 results "
        "(title, URL, published date, summary). Use filters to narrow."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, **_FILTER_PROPERTIES},
        "required": ["query"],
    },
}

SUBMIT_HOP_TOOL = {
    "name": "submit_hop",
    "description": (
        "Commit this hop to the chain. Call ONCE per hop after you have "
        "verified the expected entity/fact with exa_search. The system will "
        "re-run your declared query+filters cold and check that the expected "
        "answer is retrievable. If the cold check fails, you will be asked "
        "to refine this hop."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The exa_search query that will resolve this hop — what a "
                    "downstream benchmark agent would search to find the answer."
                ),
            },
            "expected_answer": {
                "type": "string",
                "description": (
                    "The short phrase this hop should resolve to (name, number, "
                    "date, or short phrase). The cold retrievability check will "
                    "fuzzy-match this string in the Exa results."
                ),
            },
            "required_exa_filters": {
                "type": "object",
                "description": (
                    "OPTIONAL. Exa filters strictly necessary for this hop to "
                    "return the right answer. Leave empty (omit or {}) if no "
                    "filter is needed. Include ONLY filters whose removal "
                    "would cause the hop to fail."
                ),
                "properties": _FILTER_PROPERTIES,
            },
        },
        "required": ["query", "expected_answer"],
    },
}

SUBMIT_QUESTION_TOOL = {
    "name": "submit_question",
    "description": (
        "Compose the final multi-hop natural-language question from the "
        "accumulated chain. Call ONCE at the end."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Natural-language multi-hop question that threads through "
                    "all hops. Must NOT name Exa filter types literally."
                ),
            },
        },
        "required": ["question"],
    },
}


# ── Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are building a synthetic multi-hop research benchmark question, one hop at \
a time. The benchmark tests an agent's ability to chain Exa web searches and \
apply filters strategically.

Today's date is {date}.

## Target for this question

- **{hop_count} hops** total. Each hop = ONE unknown that an agent resolves \
with ONE Exa search. Hop N's answer MUST be an input to hop N+1.
- **{filter_count} filtered hops** — meaning {filter_count} of your {hop_count} \
hops must call exa_search with a non-empty required_exa_filters object. The \
other hops must pass NO filters (empty or omitted required_exa_filters). \
Each filtered hop can use any combination of filter keys (e.g., \
`category=news` + `start_published_date` + `end_published_date` on the same \
hop counts as ONE filtered hop, not multiple).
- A filtered hop only counts if the filters are STRICTLY NECESSARY — removing \
them would cause the hop to fail or return a wrong answer.
- Seed topic: {seed_topic}.

## Per-hop workflow

You will build the chain in a conversation. On each turn:
1. Use `exa_search` as many times as needed to explore and lock the next \
hop's entity/fact. Query variations, filter experiments — use it freely.
2. When you have a concrete entity/fact and know the exact filters (if any) \
that reliably surface it, call `submit_hop` with:
   - `query`: the exact Exa query that resolves this hop
   - `expected_answer`: the short phrase the cold check will look for
   - `required_exa_filters`: OPTIONAL — only include if strictly necessary
3. The system will run your query+filters cold and fuzzy-match the expected \
answer in the top results. If it lands, the hop commits and you advance to \
the next hop. If not, you retry this hop.

**You MUST use a tool on every turn** (either exa_search or submit_hop) — \
text-only responses are not accepted.

## Hop-chaining discipline (critical)

- **Each hop must require a new search.** If the answer to hop N+1 would \
already be visible in hop N's search results, it's not a real new hop — \
it's hop inflation. Pivot to a genuinely new property of the resolved entity.
- **Pivot, don't re-extract.** Good pivot: "Hassabis's advisor was Maguire → \
what was Maguire's famous 2000 study?". Bad: "Hassabis's PhD advisor → what \
year did Hassabis earn his PhD?" (both in the same bio paragraph).

## Filter discipline — EXACTLY {filter_count} FILTERED HOP(S)

Of your {hop_count} hops, EXACTLY {filter_count} must use required_exa_filters \
(any combination within a hop counts as one filtered hop); the rest must use \
no filters. Under- or over-filtering is a spec violation.
- **Counting rule: one hop with ANY filter combination = one filtered hop.** \
  A single hop with `category=news + start_published_date + end_published_date` \
  is still ONE filtered hop.
- Reliable filter combinations (each = one filtered hop):
  - `category=news` + `published_date` window — time-bounded events.
  - `include_domains=[".gov"/".org"]` + optional date — authoritative facts.
  - `include_domains=["en.wikipedia.org"]` — biographies.
- AVOID: `include_domains=["arxiv.org"]` + narrow date window (Exa's arXiv \
date index is flaky); `include_domains=["sec.gov"]` + specific dollar \
figures (Exa indexes filing metadata, not 10-K line items).

## Final compose turn

After the chain is built, you will be asked to write a natural-language \
multi-hop question that threads through all hops. Rules:
- The question must NOT contain literal Exa API terms ("include_domains", \
"category", etc.). Use organic hints ("in a recent news article", "on the \
official government site").
- The golden_answer is the last hop's expected_answer.
- Emit via `submit_question`.
"""


HOP_PROMPT = """\
Build hop {hop_idx} of {hop_count}.

{context}

Use exa_search to explore and lock the entity, then call submit_hop."""


COMPOSE_PROMPT = """\
The chain is complete. Here is what you built:

{chain_summary}

Now compose a single natural-language question that threads through ALL \
{hop_count} hops in order. The question's terminal answer is the last hop's \
expected_answer: {golden!r}.

Rules:
- Don't name Exa filter types literally. Hint organically.
- The question should read like BrowseComp — natural, curiosity-driven, \
multi-clause.
- Do NOT change any facts from the chain.

Emit via submit_question."""


RETRY_HOP_FEEDBACK = """\
The cold retrievability check FAILED for your submit_hop.

What we ran:
  query  = {query!r}
  filters = {filters}
  expected_answer = {expected!r}

Judge feedback:
  {reason}

Address the judge's diagnosis directly — tighten the query, drop or adjust \
filters, or pick a different pivot. Then retry submit_hop for this hop."""


# ── Data ───────────────────────────────────────────────────────────────


@dataclass
class Hop:
    step: int
    query: str
    filters: dict
    expected: str


@dataclass
class QuestionRecord:
    id: str
    hop_count: int
    filter_count: int
    filter_types: list[str]
    question: str
    ideal_path: list[dict]
    golden_answer: str
    accepted: bool = False
    preflight_issues: list[str] = field(default_factory=list)
    seed_topic: str = ""


# ── Canonicalization + preflight ──────────────────────────────────────


def canon_filter_key(k: str) -> str:
    if k in ("start_published_date", "end_published_date"):
        return "published_date"
    if k in ("start_crawl_date", "end_crawl_date"):
        return "crawl_date"
    return k


def canonical_filter_types(hops: list[Hop]) -> list[str]:
    """Record of distinct filter-type families used across ALL hops (for analysis)."""
    seen: set[str] = set()
    for hop in hops:
        for k in (hop.filters or {}).keys():
            seen.add(canon_filter_key(k))
    return sorted(seen)


def filtered_hop_count(hops: list[Hop]) -> int:
    """Number of hops that use any non-empty filter combination."""
    return sum(1 for h in hops if h.filters)


_FORBIDDEN_QUESTION_TERMS = [
    "include_domains", "exclude_domains",
    "start_published_date", "end_published_date",
    "start_crawl_date", "end_crawl_date",
    "include_text", "exclude_text",
    "Exa filter", "category=", "category:",
]


def preflight(
    question: str,
    hops: list[Hop],
    hop_count: int,
    filter_count: int,
) -> list[str]:
    issues: list[str] = []
    if not question.strip():
        issues.append("Empty question.")
    if len(hops) != hop_count:
        issues.append(f"hops = {len(hops)} but spec requires {hop_count}.")
    fc = filtered_hop_count(hops)
    if fc != filter_count:
        issues.append(
            f"filtered hops = {fc} but spec requires {filter_count}."
        )
    lc_q = question.lower()
    for term in _FORBIDDEN_QUESTION_TERMS:
        if term.lower() in lc_q:
            issues.append(f"question contains literal filter term {term!r}.")
    return issues


# ── Helpers ────────────────────────────────────────────────────────────


def _make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _exa_search_formatted(exa: ExaClient, query: str, filters: dict) -> str:
    args = dict(filters or {})
    args.setdefault("type", "neural")
    try:
        return exa.search(query, **args).formatted()
    except Exception as e:
        return f"Search error: {e}"


COLD_CHECK_MODEL = "claude-haiku-4-5-20251001"

_COLD_CHECK_TOOL = {
    "name": "judge_retrievability",
    "description": "Judge whether Exa evidence corroborates the expected answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no"]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
    },
}

_COLD_CHECK_SYSTEM = """\
You judge whether a block of Exa search results CONTAINS or CORROBORATES a \
specific expected answer. Be pragmatic:
- Minor wording differences are fine ("UK" vs "United Kingdom", "$383.285B" vs "$383 billion").
- For compound answers ("X and Y"), accept if both parts appear even in \
different order or phrasing.
- Synonym-level equivalence is fine.
- Factual drift beyond trivial rounding is not.

Emit your verdict via judge_retrievability:
- `verdict`: "yes" or "no".
- `reason`: brief diagnostic. When NO, be actionable — say WHY the evidence \
doesn't support the expected answer (e.g., "the actual year was 1983, not \
1981"; "results are all about a different person with the same surname"; \
"the filter returned off-topic results — drop category=news for this query"). \
The generator will use your reason to refine on retry."""


def _cold_check(
    client: anthropic.Anthropic,
    evidence: str,
    expected: str,
    usage_tracker: dict | None = None,
) -> tuple[bool, str]:
    """Cheap Haiku judgment: does the evidence corroborate the expected answer?

    Optionally accumulates Haiku token usage into `usage_tracker`.
    """
    try:
        resp = client.messages.create(
            model=COLD_CHECK_MODEL,
            max_tokens=200,
            temperature=0.0,
            system=_COLD_CHECK_SYSTEM,
            messages=[{"role": "user", "content": (
                f"EXPECTED ANSWER: {expected!r}\n\n"
                f"EXA SEARCH RESULTS:\n{evidence}"
            )}],
            tools=[_COLD_CHECK_TOOL],
            tool_choice={"type": "tool", "name": "judge_retrievability"},
        )
    except Exception as e:
        return False, f"cold-check API error: {e}"
    if usage_tracker is not None:
        u = getattr(resp, "usage", None)
        if u is not None:
            usage_tracker["haiku_in"] = usage_tracker.get("haiku_in", 0) + (getattr(u, "input_tokens", 0) or 0)
            usage_tracker["haiku_out"] = usage_tracker.get("haiku_out", 0) + (getattr(u, "output_tokens", 0) or 0)
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            i = block.input
            verdict = (i.get("verdict") or "").lower().strip()
            reason = (i.get("reason") or "").strip()
            return verdict == "yes", reason
    return False, "no judgment returned"


# ── Iterative Generator ───────────────────────────────────────────────


class IterativeGenerator:
    """Builds a chain of hops turn-by-turn, then composes the final question."""

    # Pricing ($ per 1M tokens). Class-level so a caller can aggregate easily.
    PRICES = {
        "sonnet": {"input": 3.00, "output": 15.00},
        "haiku":  {"input": 1.00, "output": 5.00},
    }

    def __init__(self, model: str = MODEL, verbose: bool = False):
        self.client = _make_client()
        self.model = model
        self.exa = ExaClient()
        self.verbose = verbose
        self.usage = {
            "sonnet_in": 0, "sonnet_out": 0,
            "haiku_in": 0, "haiku_out": 0,
        }

    def _track(self, resp, kind: str) -> None:
        """kind: 'sonnet' or 'haiku'."""
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.usage[f"{kind}_in"] += getattr(u, "input_tokens", 0) or 0
        self.usage[f"{kind}_out"] += getattr(u, "output_tokens", 0) or 0

    def cost_usd(self) -> float:
        p = self.PRICES
        return (
            self.usage["sonnet_in"]  * p["sonnet"]["input"]  / 1_000_000
            + self.usage["sonnet_out"] * p["sonnet"]["output"] / 1_000_000
            + self.usage["haiku_in"]  * p["haiku"]["input"]  / 1_000_000
            + self.usage["haiku_out"] * p["haiku"]["output"] / 1_000_000
        )

    def build(
        self,
        hop_count: int,
        filter_count: int,
        seed_topic: str,
        max_hop_retries: int = 1,
        max_turns_per_hop: int = 3,
    ) -> tuple[list[Hop], str] | None:
        """Run the full per-question pipeline. Returns (hops, question) or None."""
        system = SYSTEM_PROMPT.format(
            date=date.today().isoformat(),
            hop_count=hop_count,
            filter_count=filter_count,
            seed_topic=seed_topic,
        )
        messages: list[dict] = []
        chain: list[Hop] = []

        for hop_idx in range(1, hop_count + 1):
            fhc = filtered_hop_count(chain)
            context_lines = []
            if chain:
                context_lines.append("Chain so far:")
                for h in chain:
                    mark = " [filtered]" if h.filters else ""
                    context_lines.append(
                        f"  hop {h.step}: query={h.query!r} "
                        f"filters={json.dumps(h.filters)} → {h.expected!r}{mark}"
                    )
                context_lines.append("")
            remaining_budget = filter_count - fhc
            remaining_hops = hop_count - hop_idx + 1
            context_lines.append(
                f"Filtered hops so far: {fhc}/{filter_count}. "
                f"Remaining hops including this one: {remaining_hops}."
            )
            if remaining_budget == 0:
                context_lines.append(
                    "Budget FULL — this hop and any remaining hops must use "
                    "NO filters (empty/omitted required_exa_filters)."
                )
            elif remaining_budget == remaining_hops:
                context_lines.append(
                    "You MUST use a filter combination on this hop (and every "
                    "remaining hop) to hit the budget exactly."
                )
            else:
                context_lines.append(
                    f"You still owe {remaining_budget} filtered hop(s) across "
                    f"{remaining_hops} remaining hop(s). Plan accordingly."
                )
            if hop_idx == hop_count:
                context_lines.append(
                    "This is the LAST hop. Its expected_answer becomes the "
                    "golden answer — keep it short (name/number/date/phrase)."
                )
            prompt = HOP_PROMPT.format(
                hop_idx=hop_idx,
                hop_count=hop_count,
                context="\n".join(context_lines),
            )
            pre_hop_msg_count = len(messages)
            messages.append({"role": "user", "content": prompt})

            committed = self._run_hop_loop(
                system, messages, hop_idx,
                max_hop_retries, max_turns_per_hop,
            )
            if committed is None:
                return None
            chain.append(committed)
            # Prune exploration history from this hop — the chain summary is
            # regenerated in the next hop's HOP_PROMPT, so context stays lean.
            del messages[pre_hop_msg_count:]

        # ── Compose final question ──
        chain_summary = "\n".join(
            f"  hop {h.step}: query={h.query!r} filters={json.dumps(h.filters)} "
            f"→ {h.expected!r}"
            for h in chain
        )
        compose_prompt = COMPOSE_PROMPT.format(
            chain_summary=chain_summary,
            hop_count=hop_count,
            golden=chain[-1].expected,
        )
        messages.append({"role": "user", "content": compose_prompt})

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.5,
            system=system,
            messages=messages,
            tools=[SUBMIT_QUESTION_TOOL],
            tool_choice={"type": "tool", "name": "submit_question"},
        )
        self._track(resp, "sonnet")
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_question":
                return chain, (block.input.get("question") or "").strip()
        return None

    # ── Hop loop (two structured attempts) ──
    def _run_hop_loop(
        self,
        system: str,
        messages: list[dict],
        hop_idx: int,
        max_retries: int,
        max_turns: int,
    ) -> Hop | None:
        """Attempt 1: up to max_turns search turns + forced submit.
        Attempt 2 (if Attempt 1's submit fails cold check): 1 search + forced submit.
        """
        hop, reason = self._one_attempt(system, messages, hop_idx, search_turns=max_turns)
        if hop:
            return hop
        if self.verbose:
            console.print(
                f"  [yellow]h{hop_idx} attempt 1 failed — retrying with 1 search + forced submit[/yellow]"
            )
        messages.append({"role": "user", "content": (
            f"ATTEMPT 1 FAILED for this hop. The cold-check judge rejected "
            f"your submission. Reason:\n  {reason}\n\n"
            "You now have ONE more search turn to investigate, followed by "
            "a FORCED submit_hop (no further searches after that). Use the "
            "search specifically to address the judge's diagnosis above — "
            "then submit a corrected hop."
        )})
        hop, _ = self._one_attempt(system, messages, hop_idx, search_turns=1)
        return hop

    def _one_attempt(
        self,
        system: str,
        messages: list[dict],
        hop_idx: int,
        search_turns: int,
    ) -> tuple[Hop | None, str]:
        """N search-or-submit turns, then a forced-submit if agent hasn't submitted.
        Returns (Hop, '') on success or (None, reason) on failure."""
        tools = [EXA_SEARCH_TOOL, SUBMIT_HOP_TOOL]
        submit_block = None

        for _ in range(search_turns):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.6,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice={"type": "any"},
            )
            self._track(resp, "sonnet")

            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == "exa_search":
                    args = dict(block.input)
                    query = args.pop("query", "")
                    result = _exa_search_formatted(self.exa, query, args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    if self.verbose:
                        flt = {k: v for k, v in args.items() if k != "type"}
                        flt_s = f" {json.dumps(flt)}" if flt else ""
                        console.print(
                            f"  [magenta]h{hop_idx} search:[/magenta] [dim]{query[:80]}[/dim]{flt_s}"
                        )
                elif block.name == "submit_hop":
                    submit_block = block
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Submitted — running cold retrievability check.",
                    })

            messages.append({"role": "assistant", "content": resp.content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if submit_block is not None:
                break

        # Force a submit if agent didn't
        if submit_block is None:
            if self.verbose:
                console.print(
                    f"  [yellow]h{hop_idx}: search turns used — forcing submit[/yellow]"
                )
            messages.append({"role": "user", "content": (
                "Search budget for this attempt is used up. You MUST call "
                "submit_hop NOW with your best candidate based on what "
                "you've already searched."
            )})
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.4,
                system=system,
                messages=messages,
                tools=[SUBMIT_HOP_TOOL],
                tool_choice={"type": "tool", "name": "submit_hop"},
            )
            self._track(resp, "sonnet")
            messages.append({"role": "assistant", "content": resp.content})
            for b in resp.content:
                if getattr(b, "type", None) == "tool_use" and b.name == "submit_hop":
                    submit_block = b
                    messages.append({"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": "Submitted — running cold retrievability check.",
                    }]})
                    break
            if submit_block is None:
                return None, "agent did not produce a submit_hop even when forced"

        # Cold check
        inp = dict(submit_block.input)
        query = (inp.get("query") or "").strip()
        expected = (inp.get("expected_answer") or "").strip()
        filters = inp.get("required_exa_filters") or {}
        evidence = _exa_search_formatted(self.exa, query, filters)
        ok, reason = _cold_check(self.client, evidence, expected, self.usage)
        if self.verbose:
            status = "[green]✓ cold check[/green]" if ok else "[red]✗ cold check[/red]"
            console.print(
                f"  h{hop_idx} submit: query={query[:60]!r} "
                f"expected={expected!r} filters={json.dumps(filters)} {status} "
                f"[dim]{reason[:80]}[/dim]"
            )
        if ok:
            return Hop(step=hop_idx, query=query, filters=filters, expected=expected), ""
        return None, (
            f"cold check NO. query={query!r} filters={json.dumps(filters)} "
            f"expected={expected!r} — judge said: {reason}"
        )

# ── Orchestrator ───────────────────────────────────────────────────────


def generate_one(
    hop_count: int,
    filter_count: int,
    seed_topic: str,
    idx: int,
    verbose: bool = True,
    usage_accum: dict | None = None,
) -> QuestionRecord | None:
    console.rule(
        f"[bold]cell({hop_count}h,{filter_count}f) #{idx}[/bold] "
        f"seed=[magenta]{seed_topic}[/magenta]"
    )

    gen = IterativeGenerator(verbose=verbose)
    result = gen.build(hop_count, filter_count, seed_topic)
    if result is None:
        console.print("  [red]generator failed to build chain[/red]")
        return None
    chain, question = result
    golden = chain[-1].expected

    console.print(f"  Q: [cyan]{question[:120]}[/cyan]")
    console.print(f"  gold: [green]{golden!r}[/green]")

    # ── Preflight ──
    pre_issues = preflight(question, chain, hop_count, filter_count)
    if pre_issues:
        console.print("  [yellow]preflight issues:[/yellow]")
        for iss in pre_issues:
            console.print(f"    - {iss}")
        return None  # no second chance after iterative build; fresh generation next cell

    # ── Final cold retrievability check (compose-drift guard) ──
    last = chain[-1]
    evidence = _exa_search_formatted(gen.exa, last.query, last.filters)
    ok, reason = _cold_check(gen.client, evidence, last.expected, gen.usage)
    if not ok:
        console.print(f"  [red]final cold check failed:[/red] [dim]{reason[:120]}[/dim]")
        return None

    if usage_accum is not None:
        for k, v in gen.usage.items():
            usage_accum[k] = usage_accum.get(k, 0) + v
    if verbose:
        console.print(
            f"  [dim]tokens: sonnet {gen.usage['sonnet_in']}in/{gen.usage['sonnet_out']}out · "
            f"haiku {gen.usage['haiku_in']}in/{gen.usage['haiku_out']}out · "
            f"cell cost ${gen.cost_usd():.4f}[/dim]"
        )
    console.print("  [green]✓ accepted[/green]")

    canon = canonical_filter_types(chain)
    return QuestionRecord(
        id="",
        hop_count=hop_count,
        filter_count=filter_count,
        filter_types=canon,
        question=question,
        ideal_path=[
            {"step": h.step, "query": h.query, "filters": h.filters, "expected": h.expected}
            for h in chain
        ],
        golden_answer=golden,
        accepted=True,
        seed_topic=seed_topic,
    )


def _load_existing(path: Path) -> tuple[list[dict], set[tuple[int, int]]]:
    """Load existing JSONL (if any) and return (records, set of done (h,f) pairs)."""
    records: list[dict] = []
    done: set[tuple[int, int]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(r)
            done.add((r["hop_count"], r["filter_count"]))
    return records, done


@click.command()
@click.option(
    "--mode", type=click.Choice(["dev", "test"]), default="dev",
    help="dev: 3 varied cells × 1. test: all valid cells × 1.",
)
@click.option("--n-per-cell", type=int, default=1)
@click.option("--seed", type=int, default=42)
@click.option(
    "--skip-existing/--no-skip-existing", default=False,
    help="Skip cells already present in the output JSONL (for restarts).",
)
@click.option("--verbose/--quiet", default=True)
def main(mode, n_per_cell, seed, skip_existing, verbose):
    """Generate synthetic filter-use benchmark questions (iterative, incremental write)."""
    random.seed(seed)

    cells = DEV_CELLS if mode == "dev" else ALL_CELLS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{mode}.jsonl"

    existing_records, done_cells = _load_existing(output_path)
    if skip_existing and done_cells:
        console.print(
            f"[dim]skip-existing: {len(done_cells)} cells already have records; "
            f"skipping those.[/dim]"
        )

    t0 = time.time()
    cumulative_usage: dict = {
        "sonnet_in": 0, "sonnet_out": 0, "haiku_in": 0, "haiku_out": 0,
    }
    # Open for append so each accepted record survives interrupts.
    mode_open = "a" if skip_existing else "w"
    if mode_open == "w":
        existing_records = []  # we're overwriting
    f_out = open(output_path, mode_open, encoding="utf-8")

    accepted_count = len(existing_records)
    try:
        for hop, filt in cells:
            if skip_existing and (hop, filt) in done_cells:
                if verbose:
                    console.print(
                        f"[dim]skipping cell({hop}h,{filt}f) — already done[/dim]"
                    )
                continue
            for i in range(n_per_cell):
                topic = random.choice(SEED_TOPICS)
                rec = generate_one(
                    hop, filt, topic, idx=i + 1, verbose=verbose,
                    usage_accum=cumulative_usage,
                )
                if rec:
                    rec.id = f"synth_{mode}_{accepted_count:04d}"
                    accepted_count += 1
                    f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    f_out.flush()
    finally:
        f_out.close()

    elapsed = time.time() - t0
    total = len(cells) * n_per_cell

    p = IterativeGenerator.PRICES
    total_cost = (
        cumulative_usage["sonnet_in"]  * p["sonnet"]["input"]  / 1_000_000
        + cumulative_usage["sonnet_out"] * p["sonnet"]["output"] / 1_000_000
        + cumulative_usage["haiku_in"]  * p["haiku"]["input"]  / 1_000_000
        + cumulative_usage["haiku_out"] * p["haiku"]["output"] / 1_000_000
    )

    console.rule("[bold]done[/bold]")
    console.print(
        f"[bold]Accepted:[/bold] {accepted_count}/{total} in {elapsed:.1f}s"
    )
    console.print(
        f"[bold]Tokens:[/bold] Sonnet "
        f"{cumulative_usage['sonnet_in']:,}in / "
        f"{cumulative_usage['sonnet_out']:,}out · Haiku "
        f"{cumulative_usage['haiku_in']:,}in / "
        f"{cumulative_usage['haiku_out']:,}out"
    )
    console.print(f"[bold]Cost:[/bold] ${total_cost:.4f} total · ${total_cost/max(accepted_count,1):.4f}/accepted")
    console.print(f"[dim]→ {output_path}[/dim]")


if __name__ == "__main__":
    main()
