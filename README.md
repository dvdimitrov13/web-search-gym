# web-search-gym

Training gym for web-search agents. Qwen3 (or Gemma) searcher + Exa, trained via SFT
then GRPO, evaluated on [BrowseComp](https://openai.com/index/browsecomp/) and
[DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa).

## Why

Earlier attempts split attention across benchmarks that conflated searcher quality with
writer quality. This repo narrows the scope: train the agent's *retrieval* behavior,
measure it on two complementary benches — BrowseComp (1,266 short-answer needle-in-
haystack tasks) and DSQA (900 multi-step research tasks with rubric-graded Set Answer
enumerations).

## Structure

```
core/       canonical harness, tools, prompts, Exa client — single source of truth
agents/     harness shapes (NOT models). Registry auto-discovers.
models/     model configs. Orthogonal to agents: any model runs any agent.
bench/      BrowseComp loader, grader, runner, CLI.
synth/      trajectory generator (uses core.harness with a teacher model).
sft/        convert → train → serve.
rl/         stub; full GRPO port lands after SFT phase is solid.
tests/      parity tests (critical) + harness + bench + convert.
```

## Quickstart

```bash
make setup                                                      # uv sync + .env
# edit .env: ANTHROPIC_API_KEY, EXA_API_KEY, OPENAI_API_KEY
make test                                                        # sanity
make bench AGENT=lean_searcher MODEL=claude_sonnet SPLIT=dev    # 10-task smoke
```

## Invocation model

```
AGENT   = which harness (lean_searcher is the one shipping)
MODEL   = which model powers the whole harness
SEARCHER_MODEL / EXTRACTOR_MODEL = override per stage
```

Adding a new baseline or checkpoint is one YAML file in `models/`, never a new agent
directory.

## Background

- Prior repo [exa-DeepBench](../exa-DeepBench) — built the harness and SFT plumbing.
- Prior repo [agent-gym](../agent-gym) — built the GRPO RL stack.
- `rl/README.md` documents the port plan from agent-gym.

## Progress — agent_dd identified as lead SFT candidate

We iterated four harness shapes against DSQA's `domain2` split (32 tasks, 2 per
problem_category, graded by the official `gemini-2.5-flash` autorater) and
converged on **`agent_dd`** — a single-rollout harness with four tools
(`search`, `browse_page`, `commit_memory`, `answer`) and async parallel dispatch.
Full write-up: [docs/agent-dd-dsqa.md](docs/agent-dd-dsqa.md).

### DSQA domain2 leaderboard (Sonnet 4.5, concurrency=4)

| Harness | Correct | F1 | Precision | Recall | Wall-clock | Avg/task |
|---|---:|---:|---:|---:|---:|---:|
| exa_deep (no-extract) | 25.0% (8/32) | 36.1% | 40.9% | 34.9% | 118s | ~15s |
| lean_searcher 5-search | 25.0% (8/32) | 41.9% | 46.8% | 41.8% | 1147s | ~143s |
| lean_searcher 5-cycle | 25.0% (8/32) | 45.8% | 49.8% | 44.1% | 1702s | ~213s |
| agent_dd 8-cycle + mem (mem=1024, no cache) | 43.8% (14/32) | 50.4% | 53.8% | 49.5% | 1046s | ~131s |
| **agent_dd 8-cycle + mem (mem=512, cached)** | **46.9% (15/32)** | **60.7%** | **62.9%** | **61.2%** | **982s** | **~123s** |

The cached configuration adds prompt caching (`cache_control` on the system
block and the last stable message block) and tightens `scratchpad_max_tokens`
from 1024 → 512. Quality moved **+3.1pt correct / +10.3pt F1** vs the prior
best; the tighter scratchpad budget nudges the model toward constraints-table
notes rather than expansive prose.

### Cost (measured from trace `usage` fields, 32-task DSQA domain2 run)

| | tokens |
|---:|---:|
| Sonnet input (uncached) | 102k |
| Sonnet output | 163k |
| Sonnet cache write | 289k |
| **Sonnet cache read** | **2.35M** |
| Haiku input (extractor) | 1.58M |
| Haiku output | 34k |

- **85.7% of Sonnet input mass served from cache** — per-turn caching of the
  system block + stable message prefix hits cleanly across turns in a rollout.
- **$6.29 for the 32-task bench** ($0.197/task, measured not estimated).
  Split: Sonnet $4.54 · Haiku $1.75.
- **400-task synthesis run projected at ~$79** — back-computed from real
  per-task cost, not a ceiling estimate.

`TraceMetadata` and `TurnState` now carry `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens` per turn plus
rolled-up totals (with a separate `extractor_*` quartet for Haiku). Every
synthesis rollout emits exact costs; no estimation from request counts.

### Gemma 4 student baselines (same harness, same DSQA split)

Same 32-task `domain2` split, same `agent_dd` 8-cycle + commit_memory
harness. Only the model is swapped. Establishes the floor a trained
Gemma 4 4B student would need to beat.

| Backend | Searcher thinking | Correct | F1 | Answered | Wall |
|---|---|---:|---:|---:|---:|
| Sonnet 4.5 (Anthropic direct) | ON (budget=1024) | **46.9%** | **0.607** | 32/32 | 982s |
| Gemma 4 26B-A4B (OpenRouter, Anthropic shim) | OFF | 16.7% (5/30)* | 0.306 | **16/30** | stuck |
| Gemma 4 E4B (self-hosted vLLM) | OFF | 6.2% | 0.156 | 31/32 | 320s |
| **Gemma 4 E4B (self-hosted vLLM)** | **ON (extractor OFF)** | 9.4% | **0.260** | 32/32 | 732s |

*OpenRouter run was 30/32 tasks — 2 stuck on HTTP hangs, killed at 40
min. 47% of completed tasks hit `nudge_exhausted` because OpenRouter's
Anthropic shim mangles Gemma's tool-use format. Native serving fixed
this (0% / 3% nudge_exhausted on the self-hosted runs).

**Headline takeaways:**
- **Native tool serving matters more than model size.** Gemma E4B on
  vLLM `--tool-call-parser gemma4` answers 100% of tasks; the shimmed
  26B-A4B only 53%. Same model family, different serving layer.
- **Thinking helps, but you must turn it OFF for the extractor stage.**
  Gemma 4 with thinking on puts extraction into `<thinking>…</thinking>`,
  which the reasoning parser strips → 97.6% empty extracts. Per-call
  `chat_template_kwargs={"enable_thinking": False}` on the extractor
  while keeping the searcher's server-default ON is the right shape,
  mirroring our Sonnet+Haiku setup (Sonnet thinks, Haiku doesn't).
- **Gemma E4B floor is F1 0.260 — 43% of Sonnet's 0.607.** That's the
  pre-SFT distance to close.

Self-hosting notes (vLLM on RTX 5090 via Vast.ai):
```bash
vllm serve google/gemma-4-E4B-it \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --gpu-memory-utilization 0.90 --host 0.0.0.0 --port 8000
# Then from local: ssh -p <port> -L 8000:localhost:8000 -N -f root@<host>
```
See [docs/agent-dd-dsqa.md](docs/agent-dd-dsqa.md) §9 for the full
setup and repro steps.

### Why `agent_dd` is a good SFT target

- **Single rollout, single reward.** The whole trajectory (search → browse →
  memory edits → cited final answer) lives in one completion. One reward
  signal applied to one contiguous rollout — a much cleaner gradient than
  training a searcher + a separate extractor.
- **Citation-native answer.** The `answer` tool's `citations` field
  references snippet IDs from the source bank. The harness validates them,
  so a reward can directly credit cited-and-correct and penalize
  hallucinated IDs — a grounded, hackable-resistant signal.
- **Uniform tool vocabulary.** Four tools, each with a structured JSON
  schema, render identically to Anthropic and OpenAI function-calling
  formats (no prompt-format drift across providers or inference backends).
- **Learned behaviors worth distilling, not noise.** Every trace in the
  winning run used `commit_memory` (32/32 tasks, mean 3.3 calls each).
  98 turns contained parallel `search` calls; 29 had parallel `browse_page`
  calls. These are substantive policy patterns — the model isn't just
  improvising; it's applying consistent strategies worth transferring to
  a smaller student.
- **Tractable rollout cost.** ~131s/task at concurrency=4 (~33s at c=16)
  makes SFT trajectory generation feasible at 10K-trajectory scale in
  a day on a single node without additional GPU, since the agent is
  Anthropic-hosted during synthesis.
- **Orthogonal to the model.** `agent_dd` is a harness; point `--model`
  at any Anthropic-compatible backend (Sonnet now, Qwen3-8B or Gemma 3/4
  when we train) and the exact same tool schema and reward signal apply.

### What's next

- Close the outstanding 23/32 tasks that hit the cycle cap — test cap=10/12.
- Generate teacher trajectories (`synth/generate`) with the agent_ddharness +
  Sonnet 4.5 against DSQA + filterbench.
- SFT a Qwen3-8B / Gemma 3-12B student on the teacher traces and
  re-benchmark against the Sonnet teacher baseline.
- Port the existing GRPO stack (`rl/`) to use agent_dd's rollout format.

## Status

Harness phase complete: `lean_searcher`, `chroma_agent`, `exa_deep`, `agent_dd`
shipping, with `agent_dd` as the lead candidate for SFT. Data synthesis and
training runs land next.
