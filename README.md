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
| **agent_dd 8-cycle + commit_memory** | **43.8% (14/32)** | **50.4%** | **53.8%** | **49.5%** | **1046s** | **~131s** |

`agent_dd` beats the next-best harness by **+18.8pt fully-correct and +4.6pt F1**
while being **~1.6× faster** end-to-end. F1 crossed 50% for the first time on
DSQA domain2.

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
