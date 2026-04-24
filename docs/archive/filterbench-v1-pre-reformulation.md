# Filterbench v1 — Pre-Reformulation Results

**Dataset**: `bench/filterbench/test.jsonl` (33 questions, 11 cells, hop_count ∈
{1..4} × filter_count ∈ {0..2} with `filter_count ≤ hop_count`).

**Important caveat**: these scores are on the **pre-polish** question wording.
The questions leaked hints heavily — named intermediate entities upfront,
quoted filter dates in plain English ("news reports from March 2024"), and
narrated the full chain in textbook prose before asking the terminal question.
v2 rewrites every question in BrowseComp style with the entity/filter hints
hidden. The pre-polish file is preserved at `test.jsonl.before_polish` for
comparison. Models should be re-benched on v2 before any conclusions about
searcher capability.

## Leaderboard

Sorted by accuracy, then wall-clock time.

| Rank | Agent | Searcher | Correct | Accuracy | Time |
|---:|---|---|---:|---:|---:|
| 1 | `exa_deep` | Exa deep (standard tier) | 31/33 | **93.9%** | 117s |
| 1 | `exa_deep` | Exa deep-reasoning | 31/33 | **93.9%** | 224s |
| 3 | `lean_searcher` | Claude Sonnet 4.5 (thinking_budget=2000) | 30/33 | 90.9% | 1010s |
| 4 | `lean_searcher` | Qwen3-14B (OpenRouter) | 28/33 | 84.8% | 938s |
| 5 | `lean_searcher` | Qwen3-8B (OpenRouter) | 27/33 | 81.8% | 1523s |
| 6 | `exa_deep` | Exa deep-lite | 26/33 | 78.8% | 109s |

Extractor for every row is Claude Sonnet 4.5 (temperature=0, max_tokens=1024),
pinned by the fix in agent yaml precedence so the extractor stays deterministic
even when the shared searcher model config enables sampling for thinking.

## Judge

OpenAI-compatible judge via OpenRouter (`gpt-4.1-mini`), grader template ported
from openai/simple-evals but **loosened after the initial run**. The strict
template rejected `"Lineage Logistics"` against golden `"Lineage"` (same IPO
issuer, pre-/post-rebrand). The loosened judge accepts common aliases,
full-vs-short forms, tickers, and standard numeric abbreviations. Only the two
Qwen runs actually used the loosened judge end-to-end; the other four were
graded once by the strict judge — their accuracy should be read with "up to
~5% understated" in mind.

## Observations

- **exa_deep (standard) is the best value**: ties exa_deep_reasoning on
  accuracy at half the wall-clock.
- **lean_searcher + Sonnet is competitive with deep tiers**, but at ~9× the
  wall-clock of exa_deep — extended thinking + iterative tool-calls are
  expensive.
- **Qwen3-14B edges 8B by 3 pp** — small but not trivial. Both still hit the
  tool-use floor: the force_tools nudge is required, and OpenRouter's
  Anthropic-compatible endpoint returns tool_use blocks without setting
  `stop_reason="tool_use"` for Qwen (already handled in `core/harness.py`).
- **exa_deep_lite floors the board at 78.8%** — same agent harness as the
  standard tier, just the cheaper Exa tier; all failures are retrieval-quality
  downgrades, not pipeline issues.

## What gets re-benched on v2

All six runs above — same agents, same models, same splits, same judge, but on
the polished questions. Expect absolute accuracy to drop across the board (the
pre-polish questions leaked chain structure in ways that obviated the
multi-hop). If the ranking holds, the ranking is credible; if it scrambles,
some of the v1 deltas were artifacts of question wording.
