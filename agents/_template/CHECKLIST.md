# Checklist for adding a new agent

Before opening a PR for a new agent under `agents/<your_name>/`:

- [ ] `agent.py` defines exactly one `BaseAgent` subclass
- [ ] Set `TOOL_SCHEMAS` on the class (or `None` if no LLM tool calls,
      e.g. exa_deep)
- [ ] Set `SYSTEM_PROMPT` on the class (or `None`)
- [ ] If your agent exposes a `search` (or `exa_search`) tool, it MUST
      compose `**EXA_SEARCH_FILTERS` from `core.tools` — never re-author
      filter parameters
- [ ] If your agent uses `commit_memory`, import `COMMIT_MEMORY_SPEC`
      from `core.tools` rather than redefining the schema
- [ ] Tools / prompts / harness live in
      `agents/<your_name>/{tools,prompts,harness}.py`. Anything that's
      *truly* shared with another agent goes in `core/` (and probably
      already does — check there first)
- [ ] `uv run pytest tests/test_core_parity.py` passes locally —
      including the parametrized check that picks up your agent
      automatically via `discover_agents()`
- [ ] If your agent emits trajectories with novel fields, document the
      schema and update `sft/convert.py` to handle them — do NOT fork
      the converter

These rules exist because synth → SFT → trained-model is a one-way
pipeline. Schema drops at any layer corrupt all downstream artifacts
and are only fixable by full re-runs. The parity test catches the most
common drop (filter set on `search`); the rest depend on author
discipline.
