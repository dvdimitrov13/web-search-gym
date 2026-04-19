# Chroma-style Agent on Filterbench — End-to-End Eval

**Motivation.** Chroma Context-1 (March 2026) is a 20B MoE agent trained for
multi-hop retrieval with a 4-tool harness: `search_corpus`, `grep_corpus`,
`read_document`, `prune_chunks`. Self-editing context (prune rewrites past
tool_results in place) is the paper's headline contribution. We wanted to
port that harness shape onto Exa and see whether a strong untrained model
(Claude Sonnet 4.5 with thinking) would pick up the grep/prune behaviors
organically, or whether they're strictly trained behaviors.

## Tool mapping: Chroma → Exa

| Chroma Context-1 | Our Exa port | Why |
|---|---|---|
| `search_corpus(q)` | `search(q, filters)` | Exa neural + filters returns ranked highlight chunks per URL |
| `read_document(id)` | *(dropped)* | Exa highlights already do server-side chunk reranking — read adds no information on a 4-5 turn budget |
| `grep_corpus(p)` | `grep(p)` | Regex over the chunks surfaced this run; returns up to 5 matches |
| `prune_chunks(ids)` | `prune(urls)` | Drops URLs from working context; past tool_results are rewritten to show `[PRUNED: reason]` placeholders |
| — | `submit(urls)` | BrowseComp-shaped hand-off for the downstream extractor |

Self-editing context is genuine: `_prepare_call_messages` walks the message
history and rewrites tool_result chunk blocks into placeholders for any
chunk whose URL was pruned. This frees token budget on the next call, just
like Chroma's design.

Thinking config: `thinking_passthrough=false` (momentary), `thinking_budget=1024`.
Extractor: same Claude Sonnet 4.5 we use everywhere.

## Results — 33-question filterbench test set

| Agent | Accuracy | Wall-clock | Thinking tokens/task |
|---|---:|---:|---:|
| v3a lean_searcher (scratchpad on) | 69.7% (23/33) | 1074s | 1,098 |
| v3b lean_searcher (scratchpad off) | **75.8% (25/33)** | 900s | 767 |
| v3c chroma_agent (search+grep+prune) | 72.7% (24/33) | **578s** | 680 |

Middle of the road on accuracy. **2× faster wall-clock** than v3a — chunked
highlights keep the per-turn context smaller than summary mode.

## Tool usage across 33 traces

| Tool | Total calls | Avg/task | Median | Tasks that used it |
|---|---:|---:|---:|---:|
| search | 92 | 2.79 | 3 | 33/33 |
| submit | 32 | 0.97 | 1 | 32/33 (1 budget-exhausted) |
| **grep** | **6** | **0.18** | **0** | **4/33 (12%)** |
| **prune** | **0** | **0** | **0** | **0/33 (0%)** |

**grep fired 6 times total across 4 tasks**. Every one was a late-trajectory
cross-reference — e.g., "check if X appears in prior chunks before
searching again." When grep was used, it contributed; it just didn't happen
often.

**prune fired zero times.** Despite:
- Budget block showing live-chunk count every turn.
- Prompt line: "Prune aggressively. When a URL is off-topic ... remove it."
- Note that grep/prune don't count against search budget.

Sonnet never reached for it. The model preferred to keep accumulating chunks
even when the context filled with off-topic material — the same pattern we
saw with the stand-alone `prune` tool on lean_searcher earlier in the session.

## Why the tools aren't used (interpretation)

The research pass on 2024–26 agentic-search literature (see session
transcript) surfaced the precise finding: **grep and prune are *trained*
behaviors in Chroma Context-1, not emergent ones.** Chroma built a curriculum
and rewarded the model for self-editing context mid-search. Out-of-the-box
frontier models don't reach for these tools because their policy prior is
"issue another search" — it's what almost every pre-training corpus example
of tool use does.

Our evidence here is consistent:
- lean_searcher with a `prune` tool added → 0 prune calls on 3 smoke tasks.
- chroma_agent with search/grep/prune as a unified vocabulary → 0 prune
  calls, 6 grep calls over 33 tasks.
- The accuracy lift from adding these tools is **not positive** (72.7% vs
  75.8% baseline), which is expected when the tools aren't used: we replaced
  per-URL summaries with chunked highlights, which is a net information loss
  at the same turn budget if the agent doesn't substitute grep for re-search.

## What this means for the project

1. **The scaffolding works.** Self-editing context works end-to-end: prune
   would actually reduce token footprint if called. The chunk store + regex
   dispatch are correct.
2. **chroma_agent is the right shape for SFT/RL, not prompting.** To make
   this agent shape beat lean_searcher, we'd need to:
   - Generate synthetic trajectories where grep and prune are exercised
     (e.g., force-insert a grep call after every 2 searches, with verified
     ground-truth benefit), or
   - RL the model with per-step rewards favoring grep-before-search and
     prune-on-irrelevance, as Chroma did.
3. **Wall-clock is a nice bonus.** 2× faster than lean_searcher at comparable
   accuracy is worth keeping on the list for baseline comparisons, even
   without the pruning payoff.
4. **Prompt coercion is probably a dead end.** We've seen through 3+
   ablations now that Sonnet doesn't adopt "unusual" tools on prompt alone.
   No amount of "use this tool when X" will reliably change behavior. RL
   or SFT is required.

## Relation to lean_searcher scratchpad result (v3a → v3b, +6 pp)

v3b showed that removing the scratchpad tool *improved* accuracy on the same
benchmark. The hypothesis — bookkeeping tools the model doesn't have a
trained policy for are net-negative, because they introduce overhead without
corresponding gain — is reinforced by v3c. Strong frontier models benefit
from *fewer* state-management tools unless those tools are trained for.

## Reproducibility

```bash
uv run python -m bench.cli run \
  --agent chroma_agent --model claude_sonnet \
  --dataset filterbench --split test --concurrent 3 --run-id fb_v3c
```

Raw at `results/raw/chroma_agent__*__fb_v3c.jsonl`, aggregate at
`results/scores/chroma_agent__*__fb_v3c.json`, traces at
`trajectories/chroma_agent/idx-*.json`.
