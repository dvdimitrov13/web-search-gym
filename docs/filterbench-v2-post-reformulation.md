# Filterbench v2 — Post-Reformulation Results

**Dataset**: `bench/filterbench/test.jsonl` (33 questions, same paths as v1),
with every question rewritten in BrowseComp style by
`synth/rewrite_questions.py` using the polish pass now baked into
`synth/gold_path_generation.polish_question()`. The rewrite preserves the
ideal_path / golden_answer exactly; only the natural-language question changes.

**What the polish does**: hides intermediate entities discovered in earlier
hops (refers by role/property), buries filter mechanics (no "news reports from
March 2024" or "on the city's official website"), collapses narrative preamble
into a single compact clause, and removes enumerators ("and given that..."
/ "and what year was...") that would otherwise reveal the hop count.

**Judge**: OpenAI-compatible via OpenRouter (`gpt-4.1-mini`), loosened template
(aliases / full-vs-short forms / tickers / numeric abbreviations accepted).

## Leaderboard

Sorted by v2 accuracy, then wall-clock. `Δ` is absolute percentage points vs v1.

| Rank | Agent | Searcher | v2 Correct | v2 Accuracy | Time | Δ vs v1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `exa_deep` | Exa deep | 26/33 | **78.8%** | 151s | −15.1 |
| 1 | `exa_deep` | Exa deep-reasoning | 26/33 | **78.8%** | 397s | −15.1 |
| 3 | `lean_searcher` | Claude Sonnet 4.5 (thinking) | 23/33 | 69.7% | 1195s | −21.2 |
| 4 | `lean_searcher` | Qwen3-8B | 21/33 | 63.6% | 1421s | −18.2 |
| 5 | `exa_deep` | Exa deep-lite | 20/33 | 60.6% | 148s | −18.2 |
| 6 | `lean_searcher` | Qwen3-14B | 19/33 | 57.6% | 817s | −27.3 |

## v1 → v2 delta table

| Agent | v1 Acc | v2 Acc | Δ |
|---|---:|---:|---:|
| exa_deep | 93.9% | 78.8% | −15.1 |
| exa_deep_reasoning | 93.9% | 78.8% | −15.1 |
| lean_searcher + Sonnet | 90.9% | 69.7% | −21.2 |
| lean_searcher + Qwen3-8B | 81.8% | 63.6% | −18.2 |
| lean_searcher + Qwen3-14B | 84.8% | 57.6% | −27.3 |
| exa_deep_lite | 78.8% | 60.6% | −18.2 |

Every agent dropped. The benchmark is genuinely harder without the leaked
entity/filter hints.

## Observations

- **Ranking largely preserved at the top**: the two strong Exa tiers still tie
  for first, and lean_sonnet is still in the top half. But:
- **Qwen3-14B flipped below Qwen3-8B** (57.6% vs 63.6%). v1's 14B-over-8B
  ordering was artifact: the larger model apparently got more of the free lunch
  from the leaked hints (longer questions, more phrases to key off). On the
  polished questions where you have to actually *find* the intermediate
  entity, 14B isn't ahead. That's a meaningful insight — worth digging into
  with per-question failure analysis.
- **exa_deep_lite and Sonnet diverged** — same agents whose v1 scores were 12
  pp apart closed to 9 pp now, but more importantly, deep-lite (cheap) now
  beats qwen14b. The pricing calculus changes.
- **exa_deep and exa_deep_reasoning tie again** at 78.8%. Reasoning tier's 2.6×
  slower wall-clock remains not worth it for this benchmark.
- **Absolute ceiling sits at 79%** — no agent cleared 80% on the polished
  questions. That's about right for a multi-hop+filter benchmark on a modern
  model stack. Room to improve.

## What this says about v1

The 15-to-27 pp drops mean v1 scores were inflated by **5-10 questions
per agent** that were solvable just from the question text (entity name +
filter hint). The strong searchers still rank highly on v2, so v1's ranking of
the top tier is partially credible — but deltas within the middle tier (qwen
8b vs 14b, lean_sonnet vs deep-lite) were noise from question leakage, not
searcher capability.

## Reproducibility

```bash
# Regenerate the polished questions from any raw v1 file:
uv run python -m synth.rewrite_questions bench/filterbench/test.jsonl

# Re-run any agent:
uv run python -m bench.cli run \
  --agent lean_searcher --model claude_sonnet \
  --dataset filterbench --split test --concurrent 3 --run-id fb_v2
```

All raw results at `results/raw/*fb_v2.jsonl`, aggregates at
`results/scores/*fb_v2.json`.
