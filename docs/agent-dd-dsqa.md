# agent_dd on DeepSearchQA — Architecture & Findings

**Motivation.** DeepSearchQA (`google/deepsearchqa` on HuggingFace; 900 prompts,
17 domains, 65% Set Answer, 35% Single Answer) is a multi-step research QA
benchmark where the gold answer is often an *enumeration* of entities, and
the judge (`gemini-2.5-flash` autorater, pinned by the authors) scores
precision / recall / F1 on that enumeration plus a strict "fully correct"
flag. The structure punishes agents that (a) can't verify multiple gold
items per task, (b) add excessive non-gold items, or (c) lose partial
progress across search cycles.

We used the `domain2` split (2 tasks per problem_category, 32 tasks, seeded)
to iterate harness designs. All numbers below are from this 32-task slice
with a `gemini-2.5-flash` judge, `concurrency=4`.

## Why a new agent

`lean_searcher` on DSQA hit 45.8% F1 but only 25% fully-correct. The failure
pattern clustered around Set Answer tasks: the model retrieved the right
URLs, but the downstream `Extractor` stage (single-line `Exact Answer:`
field) collapsed 5+ item enumerations into one entity. Even skipping the
extractor for DSQA, summary-mode highlights weren't surfacing every gold
item from dense list pages (Gemini Flash summaries cap around ~1K tokens /
~4k chars — a ceiling that bites on list-heavy pages).

`agent_dd` is a separate harness shape that attacks these specific
bottlenecks.

## Architecture

**Single-agent, single-rollout.** Unlike `lean_searcher` (searcher → extractor
pipeline), `agent_dd` drives retrieval AND synthesis in one rollout, ending
with a structured `answer` tool call. Inspired by DR Tulu
(arXiv:2511.19399), adapted for our Anthropic-native infra.

**Four tools:**

| Tool | Behavior | Cost model |
|---|---|---|
| `search` | Exa **highlights** mode (`maxCharacters=200`, `highlights_per_url=3`, `type=instant`). Also pulls full page `text` alongside highlights in the same Exa call; caches it per-URL. Returns `S_*` snippet ids for citation. | 1 cycle per turn with ≥1 search |
| `browse_page` | Reads `url`'s already-cached page text (no Jina, no network) and runs Haiku (`claude-haiku-4-5-20251001`, `max_tokens=320`, `max_page_chars=128000`) with a detail-preservation prompt against the caller's `question`. Returns `B_*` ids. | 1 cycle per turn with ≥1 browse |
| `commit_memory` | Token-budgeted edit-in-place scratchpad (reused from lean's schema). Fuzzy-match `old_text`/`new_text`; over-budget turns lock to commit_memory only until shrunk. | **Free** (no cycle) |
| `answer` | Structured `{final_answer, explanation, citations}`. Ends the rollout. Harness validates each citation id against the source bank. | N/A |

**Budget:** `max_cycles=8`. Any turn with ≥1 non-error search OR browse = 1
cycle. `commit_memory` and `answer` don't consume cycles. Parallel calls
within one turn share a single cycle.

**Async parallel execution.** `_dispatch_async` phases tool calls so they
actually run concurrently rather than serializing in a for-loop:

1. All `commit_memory` blocks (sync, no I/O).
2. All `search` blocks (concurrent via `AsyncExa.search` + `asyncio.gather`).
   They populate the per-task page cache.
3. All `browse_page` blocks (concurrent via `AsyncAnthropic.messages.create`
   + `asyncio.gather`). They read from the page cache.
4. `answer` block (sync).

On the smoke task (idx 737, 3 parallel searches + 3 parallel browses), the
same rollout went from **147.5s sequential → 35.3s with async** (4.2×).

**Snippet ids:** `S_<8hex>` for highlights (`sha1(url + text[:80])`),
`B_<8hex>` for browses (`sha1(url + question[:80])`). Deterministic, stable
across retries within a rollout, prefix-separated so the model (and
downstream consumers) can distinguish highlight vs browse evidence.

## Design decisions traced through experiments

### 1. Summary-mode detail preamble (before agent_dd)
On `lean_searcher`, every summary query passes through a fixed extractive-
enumeration preamble before reaching Exa. Empirically doubled per-URL
summary density (~1,175 → ~2,417 chars) on multi-item pages by pushing
Gemini Flash from paragraph-summary mode into verbatim enumeration. All
`search()` callers benefit — training contract preserves the wrapper so
training data and inference see the same shape.

### 2. Cycle budget semantics (before agent_dd)
Changed `max_searches` from "5 individual Exa calls" to "5 turns that
include any search or browse." Parallel searches in one turn share a
cycle. Prompt updated to encourage parallel emission for independent
sub-queries. Strong adoption: ~70 parallel-search turns across 32 tasks
on `lean_searcher`.

### 3. Exa `contents.text` caching → drop Jina
Initial `browse_page` design used Jina Reader (`r.jina.ai`) + Haiku. Hit
repeated 30s timeouts on heavy pages (`wellplated.com/steak-fries/`).
Observation: Exa's `contents` object accepts `text` alongside `highlights`
in one call. `search` now pulls full page text per URL into a per-task
`page_cache`; `browse_page` extracts from memory. Dropped Jina entirely —
no external key, no timeouts, no rate-limit path.

### 4. Single-agent (no extractor)
DSQA's grader reads prose naturally. Passing the agent's `{explanation,
final_answer}` directly to the Gemini judge beats running an
intermediate extractor that collapses Set Answer enumerations.
Implemented by adding `Answer.natural_text` and `prefers_natural_response`
on the grader.

### 5. Async SDK wiring
Both `exa_py.AsyncExa` and `anthropic.AsyncAnthropic` exist. Harness
is sync on the surface (`_dispatch` calls `asyncio.run(_dispatch_async)`)
so the runner, agent, and bench CLI stay sync. Thread-pool equivalent
was the alternative — picked asyncio because we're calling two libraries
that natively expose async clients.

### 6. Cycle cap at 8 (not 5)
With `max_cycles=5`, 28/32 tasks hit the cap; accuracy 21.9%, F1 36.8%.
Bumping to 8: 26/32 at the cap, accuracy 28.1%, F1 46.4%. DSQA causal
chains routinely need 6-8 cycles when the model splits between retrieval
and drill-down.

### 7. Adding `commit_memory` back
Initial agent_dd didn't include commit_memory. Re-added with the same
schema as `lean_searcher`. Doesn't consume a cycle. All 32 tasks used it,
mean 3.3 calls per task. Biggest single-step improvement in the whole
experiment: **+15.7pt fully-correct, +4pt F1** just from adding the
scratchpad.

## Results — DSQA domain2 (32 tasks)

| Agent config | Fully correct | F1 | Precision | Recall | Wall (c=4) | Avg/task |
|---|---:|---:|---:|---:|---:|---:|
| exa_deep (no-extract) | 25.0% (8/32) | 36.1% | 40.9% | 34.9% | 118s | ~15s |
| lean 5-search (orig budget) | 25.0% (8/32) | 41.9% | 46.8% | 41.8% | 1147s | ~143s |
| lean 5-cycle | 25.0% (8/32) | 45.8% | 49.8% | 44.1% | 1702s | ~213s |
| lean 5-cycle + instant + split-query | 18.8% (6/32) | 42.4% | 45.4% | 41.7% | 1449s | ~181s |
| agent_dd 5-cycle | 21.9% (7/32) | 36.8% | 47.0% | 34.0% | 490s | ~61s |
| agent_dd 8-cycle | 28.1% (9/32) | 46.4% | 54.1% | 43.5% | 750s | ~94s |
| **agent_dd 8-cycle + commit_memory** | **43.8% (14/32)** | **50.4%** | **53.8%** | **49.5%** | **1046s** | **~131s** |

**Per-task timings** are the full-bench wall-clock divided by 32, with
`concurrency=4`. The wall-clock number is what you'd actually wait for;
the per-task number approximates what each task costs if you were running
serially but amortized across 4 concurrent workers. To estimate raw
single-task time, multiply by ~4.

The headline:
- **agent_dd 8-cycle + mem is best** on Fully-correct, F1, and Precision.
- **~1.6× faster than lean's cycle run** (1046s vs 1702s) at higher quality.
- `exa_deep` is the only sub-minute option on average, but at ~7pt lower F1.

## Behavioral findings (from the 32 traces)

### Parallel adoption

| | 5-cycle | 8-cycle | **8-cycle + mem** |
|---|---:|---:|---:|
| Turns with parallel search | 62 | 96 | 98 |
| Turns with parallel browse | 19 | 22 | 29 |

Parallel emission is comfortable once the prompt affords it. Async dispatch
turns this into real wall-clock savings (3 concurrent browses = ~3× faster
than serialized).

### Tool usage

| | 5-cycle | 8-cycle | **8-cycle + mem** |
|---|---:|---:|---:|
| Cycles used (mean / max) | 4.7 / 5 | 7.2 / 8 | 6.8 / 8 |
| Searches issued (mean / max) | 4.8 / 10 | 7.1 / 18 | 8.2 / 17 |
| Browses issued (mean / max) | 3.0 / 9 | 3.9 / 9 | 4.4 / 11 |
| commit_memory calls (mean / total) | — | — | 3.3 / 105 |
| Tasks that used commit_memory at all | — | — | **32/32** |

### Citation discipline

- 31/32 answers emitted explicit `citations` on the 5-cycle run.
- All citations verified against `source_bank`; no invented ids reported
  back in recent runs.
- Citation format matches DR Tulu's `<cite id="...">` / `\boxed{}`
  semantically, using native Anthropic tool schema rather than XML.

### Failure modes observed

- **Google-style `site:` operators** in query strings (`site:example.com`) —
  Exa doesn't parse them; they just become noise tokens in the embedding.
  The `search` tool description explicitly warns against this now.
- **Budget exhaustion at cycle cap** (23/32 at cap for 8-cycle + mem) —
  further gains likely if cycles bumped to 10-12, but diminishing returns
  expected. Not run yet.

## What this tells us

1. **Harness choice matters more than model choice** for this task class.
   Same Sonnet 4.5, four different harness shapes, 19-44% accuracy span.
2. **State tracking (commit_memory) is the single biggest lever** — +56%
   relative improvement on fully-correct from adding it to an otherwise
   identical agent_dd. DSQA's causal-chain structure rewards explicit
   constraint tracking between cycles.
3. **Browse-on-cached-text beats separate browse tool.** Caching page
   text at search-time eliminates a whole class of failure (Jina timeouts,
   rate limits) and enables unlimited-depth extraction without a second
   network call per drill-down.
4. **Async parallel dispatch is load-bearing** for latency when the
   model uses parallel tool calls (which it does readily under our
   prompt). Sequential dispatch gave us correctness at 4× the cost.
5. **Single-agent > searcher+extractor** for grade-the-prose benchmarks.
   DSQA's judge reads natural responses; the lean_searcher extractor
   actively hurt Set Answer scores by compressing enumerations.

## Next experiments (not run)

- Cycle cap at 10-12 — squeeze the last few points out. Returns likely
  diminishing.
- Drop `type=instant` for `auto` — on lean this was a 3-4pt regression;
  agent_dd has instant on by default for retrieval-latency reasons that
  matter less once we have the page cache. Worth a direct A/B.
- Parallel-friendly `commit_memory` — currently sync; could be issued
  alongside searches/browses in one turn for zero extra latency.
- Run the 900-task full split as a publication-quality number.

## Files

- `agents/agent_dd/` — agent class, config.
- `core/agent_dd_harness.py` — async dispatch loop, scratchpad + shrink
  state machine, phased concurrent tool execution.
- `core/agent_dd_tools.py` — schema for `search`, `browse_page`,
  `commit_memory`, `answer`.
- `core/agent_dd_prompts.py` — system prompt.
- `core/browse.py` — Haiku extractor (sync + async).
- `core/exa_client.py` — `AsyncExa`, page-text caching, highlight helpers.
- `bench/deepsearchqa.py` — HF loader, domain2 stratified split, Gemini
  autorater per the official Kaggle starter notebook.
