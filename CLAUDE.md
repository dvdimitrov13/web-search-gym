# web-search-gym

Training gym for web-search agents. Qwen3 + Exa searcher, trained via SFT then GRPO,
evaluated on BrowseComp.

## Tech stack

- Python ≥3.10, `uv` for env management
- Anthropic SDK (searcher calls via native or OpenRouter)
- `exa-py` for web search
- `trl` + `peft` + `transformers` for SFT (and later RL)
- `click` + `rich` for CLI / output
- `pytest` for tests

## Non-negotiable architectural rules

1. **`core/` is the single source of truth.** `agents/`, `synth/`, `sft/`, `bench/`,
   and (later) `rl/` all import prompts, tool schemas, the Exa client, the dynamic
   context builder, and the harness loop from `core/`. No reimplementations.
2. **Never fork a prompt.** If you need a variant, parameterize — don't duplicate.
   `tests/test_core_parity.py` enforces this.
3. **Agents ≠ models.** A directory under `agents/` is a harness shape. Adding a
   new model means one YAML in `models/`, not a new agent directory.
4. **Scratchpad tool returns status only.** Never inline full scratchpad content
   into tool results (the `<scratchpad>` system block shows it).
5. **Exa client uses `search(contents={"summary": {"query": query}})` only.**
   No `highlights`, no `get_contents` with a different query.
6. **Source bank is a first-class Trace field.** Never re-fetch content; downstream
   stages read from the Trace.

## Conventions

- Keep comments minimal; only explain WHY (non-obvious constraints, gotchas).
- Prefer editing over creating files.
- Imports: core stdlib, then third-party, then local, alphabetized within groups.
- Tool schemas live in `core/tools.py`; they're converted to Anthropic or OpenAI
  function-calling format by helper functions, not duplicated.

## Key commands

```bash
make setup
make test
make bench AGENT=lean_searcher MODEL=claude_sonnet SPLIT=dev
make list-agents
make list-models
```

## Task splits (in `bench/splits.yaml`)

- `dev` — 10 tasks, smoke tests
- `smoke` — 50 tasks, SFT iteration
- `full` — 1266 tasks, final eval

## Current state

Scaffold only. No agents trained, no data generated.
