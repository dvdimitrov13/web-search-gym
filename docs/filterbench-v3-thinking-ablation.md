# Filterbench v3 — Thinking / Scratchpad Ablation

Same polished 33-question test set as v2. Changing searcher-side knobs to probe
where the accuracy actually comes from.

## Config delta v2 → v3

- `thinking_passthrough: true` → **false** (momentary thinking — prior turns'
  thinking blocks are stripped from the next API call; still saved to trace).
- `thinking_budget: 2000` → **1024**.
- Filter schema camelCase → snake_case (`core/tools.py`), so `startPublishedDate`
  style errors from v2 no longer waste a search turn.

Everything else (extractor, judge, Exa client, prompt text, concurrent=3) held
constant.

## v3a: lean_searcher + Sonnet (momentary thinking, scratchpad on)

**Accuracy: 23/33 = 69.7%** — identical to v2. Wall-clock 1074s vs v2's 1195s.

| Metric | v2 | v3a | Δ |
|---|---:|---:|---:|
| Accuracy | 69.7% | 69.7% | 0 |
| Wall-clock | 1195s | 1074s | −10% |
| Thinking tokens / task | ~4,400 | ~1,098 | **−4×** |
| Thinking tokens / block | ~935 | ~146 | −6× |
| Thinking blocks / task | 4.7 | 7.5 | +60% |
| Scratchpad edits / task | 0.94 | 3.61 | **+4×** |
| Tasks with ≥1 scratchpad edit | 9/33 (27%) | 32/33 (97%) | +70 pp |

## What changed behaviorally

1. **Thinking became short and tactical**. From ~935 tokens per block to ~146.
   Stripping prior-turn thinking removed the "restate the deliberation so far"
   context, so each turn thinks to decide the next action and stops.
2. **Scratchpad became the state-holder**. 97% of tasks maintain an explicit
   constraints table now. In v2 the model kept state inside its internal
   monologue; with momentary thinking it has to externalize.
3. **Accuracy unchanged**. 3,300 tokens of extra v2 thinking per task were
   rehashing, not reasoning. Correctness didn't move.

## Implications

- **Better SFT data**: v3 trajectories are shorter, more tool-use-dense, and
  externalize state — easier for a student model to imitate than long interior
  monologues.
- **Lower cost per task**: ~4× less thinking billed.
- **More auditable**: scratchpad is now a real artifact; can reconstruct the
  searcher's state at any turn from tool-use history + scratchpad snapshots.

## v3b: lean_searcher + Sonnet (scratchpad disabled)

Next run in this doc. Tests whether removing the scratchpad externalization
forces thinking back up (and whether accuracy survives without any explicit
state-tracking tool).
