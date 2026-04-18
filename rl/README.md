# RL — later phase

This directory is intentionally empty except for this README and `configs/`
placeholders. The full GRPO port lands when the SFT phase is solid.

## Why not port now

Dead code drifts. The upstream RL stack (`~/Projects/agent-gym`) moves faster
than we can maintain a cargo-culted copy. We'll re-port fresh at the moment
the SFT checkpoint is ready for RL warmup, so the code that lands matches the
state of the upstream repo and the trained checkpoint.

## Port plan (for the future us)

Source: `~/Projects/agent-gym/src/`.

### Port nearly as-is

| From | To | Notes |
|---|---|---|
| `training/tito.py` | `rl/tito.py` | TI/TO token-splicing primitives. Swap snippet-ID tracking (`[S1]`, `[R1]`) for URL tracking keyed off `core.trace.Trace.source_bank`. |
| `training/tito_trainer.py` | `rl/tito_trainer.py` | `_encode_tool_result` must re-tokenize the full prompt each turn, because our system prompt embeds dynamic `<exa_api>` / `<budget>` / `<scratchpad>` blocks. Agent-gym's pure splicing won't work. |
| `training/thinking_budget.py` | `rl/thinking_budget.py` | `ThinkingBudgetProcessor` logits processor. Port untouched. |
| `rewards/base.py` (RewardComposer) | `rl/rewards/composer.py` | Multiplicative mode is the one we want. |
| `rewards/thinking_reward.py` | `rl/rewards/thinking.py` | Port untouched. |

### Rewrite

| From | To | Notes |
|---|---|---|
| `rewards/llm_judge_reward.py` | `rl/rewards/judge.py` | Rewrite around **BrowseComp-style short-answer correctness**. Given the searcher's source bank, run the `core.extractor.Extractor` to produce an Answer, then LLM-judge against the gold answer (yes/no). This replaces agent-gym's open-ended retrieval quality judge with a direct pass/fail signal that matches the bench. |
| `rewards/format_reward.py` | `rl/rewards/hallucination.py` | Replace snippet-ID validity check with **URL-hallucination check**: fraction of submitted URLs that exist in the accumulated `source_bank`. Soft multiplier, not a hard gate. |

### Skip

- `training/offpolicy_trainer.py` + `training/rollout_server.py` — untested
  upstream, not needed for v1.
- DuckDuckGo / Serper providers — we use Exa.
- The full 200-question training set — regenerate fresh from BrowseComp's
  training-question distribution via `synth/generate.py`.

## Non-negotiables when the port lands

1. **SFT warmup first.** Agent-gym's V3 (no-think, no warmup) collapsed to 59%
   submit rate due to format failures. Our SFT checkpoint fixes that before RL.
2. **Multiplicative reward, not additive.** Agent-gym's V1 (additive) let good
   retrieval compensate for broken format. V2 (multiplicative) fixed this
   within a single eval. Same decision applies here.
3. **Rewards read `core.trace.Trace`.** Any new reward component must consume
   the Trace schema — not a bespoke rollout format. If the shape doesn't fit,
   extend `core.trace` rather than forking.
4. **Token-in / token-out.** Keep the rollout loop in token space (no
   decode/re-encode round-trips). This was the training-stability fix from
   SID-1 that agent-gym validated in V2.

## Related

- `../sft/` — produces the warmup checkpoint this phase builds on.
- `../bench/` — BrowseComp is the same eval the reward judge compares against.
- Upstream: `~/Projects/agent-gym/REPORT.md` for V1/V2/V3 findings.
