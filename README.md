# web-search-gym

Training gym for web-search agents. Qwen3 searcher + Exa, trained via SFT then GRPO,
evaluated on [BrowseComp](https://openai.com/index/browsecomp/).

## Why

Earlier attempts split attention across benchmarks that conflated searcher quality with
writer quality. This repo narrows the scope: train only the searcher, measure only
searcher-dependent answer accuracy on BrowseComp's 1,266 short-answer tasks.

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

## Status

Scaffold only. SFT and RL land in later phases.
