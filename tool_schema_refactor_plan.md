# Agent Refactor Plan — BaseAgent ownership across all 4 agents

**Status:** ready to execute. Big-bang refactor that puts every agent's
tool schemas, prompts, and harness inside `agents/<name>/` and gives
`BaseAgent` class-level ownership of those assets.

**Scope:** structural refactor only. Filter-drop fix lands inside Phase A.
No synth re-run. No model retraining.

---

## Why this exists

### 1. The proximate bug

`core/agent_dd_tools.py:32-52` defines agent_dd's `search` tool with
`query` as the only parameter. The shared filter set in
`core/tools.py:17-50` (`_EXA_FILTERS`: category, dates,
include/exclude_domains) is composed by `core/tools.py::CANONICAL_TOOLS`
and `core/chroma_tools.py::CANONICAL_CHROMA_TOOLS` but NOT by
`agent_dd`.

Consequence: the 400-question synth run produced trajectories where
Sonnet (the teacher) never had the option to use Exa filters. That
schema gap propagated through `sft/convert.py` into
`sft/data/agent_dd_*.jsonl`, and the LoRA-tuned student trained on
filter-less traces.

### 2. The structural cause

Each agent has its tool schemas / prompts / harness scattered across
`core/` with no class-level ownership:

| Agent          | Tools                          | Prompts                                | Harness                  |
|----------------|--------------------------------|----------------------------------------|--------------------------|
| lean_searcher  | `core/tools.py`                | `core/prompts.py`                      | `core/harness.py`        |
| agent_dd       | `core/agent_dd_tools.py`       | `core/agent_dd_prompts.py`             | `core/agent_dd_harness.py` |
| chroma_agent   | `core/chroma_tools.py`         | `core/prompts.py` (`CHROMA_SEARCHER_PROMPT`) | `core/chroma_harness.py` |
| exa_deep       | n/a (no tool calls)            | n/a                                    | n/a (single Exa call)    |

Three problems:

1. **Inconsistent placement.** `agent_dd` has its own prompts file;
   `chroma_agent`'s prompt is mixed into shared `core/prompts.py`. Same
   asset, different homes. Authors guess.
2. **No class-level ownership.** Tool schemas live as module-level dicts.
   To check "every agent that exposes a search tool has the canonical
   filter set," you'd need a hand-maintained registry. There is none.
3. **`core/harness.py` mixes shared helpers (`make_client`, `_llm_call`,
   `_fuzzy_replace`, `_RETRY_DELAYS`) with lean_searcher's
   `SearcherHarness`.** chroma_harness and agent_dd_harness import the
   helpers from there. CLAUDE.md mandates "agents ≠ models" but in
   practice the harness IS the agent shape.

### 3. The fix

After this plan:

1. Each agent owns its tools, prompts, and harness inside
   `agents/<name>/`.
2. `BaseAgent` has class-level slots (`TOOL_SCHEMAS`, `SYSTEM_PROMPT`)
   that every agent populates. The parity test iterates all discovered
   agents and asserts conformance.
3. `core/` keeps only the genuinely shared assets: filter fragment,
   universal helpers, context builders, extractor, types, exa client.
4. `agents/_template/CHECKLIST.md` makes the contract explicit for
   future authors.

---

## Out of scope

- **Re-running synth/SFT.** User has additional pre-rerun work first.
- **Restoring filter mentions in `agent_dd_prompts.py`.** Prompt content
  change for the next synth run, separate PR.
- **Touching exa_deep's tool/prompt slots.** It has none — sets both to
  `None`.

---

## Final architecture

```
core/
  llm.py             # NEW — `make_client`, `_llm_call`, `_RETRY_DELAYS`
                     # (extracted from core/harness.py; shared by 3 harnesses)
  scratchpad.py      # NEW — `_fuzzy_replace` (shared by 2+ harnesses)
  tools.py           # SHRUNK — only `EXA_SEARCH_FILTERS` (renamed from `_EXA_FILTERS`)
                     # plus tiny render helpers if every agent uses them
  prompts.py         # SHRUNK — only `THINKING_INSTRUCTION`, `BROWSE_EXTRACT_PROMPT`,
                     # `EXTRACTOR_PROMPT` (genuinely shared)
  context.py, exa_client.py, extractor.py, browse.py,
    console.py, types.py, trace.py, gemini_client.py,
    openai_adapter.py                                  # UNCHANGED

agents/
  base.py            # EXTENDED — adds `TOOL_SCHEMAS`, `SYSTEM_PROMPT` class slots
  registry.py        # UNCHANGED
  lean_searcher/
    agent.py
    config.yaml
    tools.py         # MOVED FROM core/tools.py — `CANONICAL_TOOLS` (composes EXA_SEARCH_FILTERS)
    prompts.py       # MOVED FROM core/prompts.py — `SEARCHER_PROMPT`, `SEARCHER_PROMPT_NO_SCRATCHPAD`
    harness.py       # MOVED FROM core/harness.py — `SearcherHarness`
  agent_dd/
    agent.py
    config.yaml
    tools.py         # MOVED FROM core/agent_dd_tools.py — `CANONICAL_AGENT_DD_TOOLS` (composes EXA_SEARCH_FILTERS — fixes the bug)
    prompts.py       # MOVED FROM core/agent_dd_prompts.py — `AGENT_DD_SYSTEM_PROMPT`
    harness.py       # MOVED FROM core/agent_dd_harness.py — `AgentDDHarness`
  chroma_agent/
    agent.py
    config.yaml
    tools.py         # MOVED FROM core/chroma_tools.py — `CANONICAL_CHROMA_TOOLS`
    prompts.py       # MOVED FROM core/prompts.py — `CHROMA_SEARCHER_PROMPT`
    harness.py       # MOVED FROM core/chroma_harness.py — `ChromaHarness`
  exa_deep/
    agent.py         # SETS `TOOL_SCHEMAS = None`, `SYSTEM_PROMPT = None` (no harness — single call)
    config.yaml
  _template/
    agent.py, config.yaml, __init__.py
    CHECKLIST.md     # NEW — author contract
```

---

## Phase ordering

Each phase is a single commit. Each phase leaves the codebase
green-on-tests so a halt at any boundary is safe.

| Phase | Goal                                              | Commit message |
|-------|---------------------------------------------------|---|
| A     | Hoist filter fragment + add BaseAgent slots       | `Hoist EXA_SEARCH_FILTERS, add BaseAgent class slots, fix agent_dd filter drop` |
| B0    | Extract shared harness helpers into `core/llm.py` and `core/scratchpad.py` | `Extract shared harness helpers into core/llm.py and core/scratchpad.py` |
| B1    | Migrate agent_dd into `agents/agent_dd/`          | `Migrate agent_dd into agents/agent_dd/` |
| B2    | Migrate chroma_agent into `agents/chroma_agent/`  | `Migrate chroma_agent into agents/chroma_agent/` |
| B3    | Migrate lean_searcher into `agents/lean_searcher/`| `Migrate lean_searcher into agents/lean_searcher/` |
| B4    | Wire exa_deep slots                               | (folded into B3; trivial) |
| C     | Rewrite parity test to iterate over all agents    | `Iterate parity test over all agents via discover_agents` |
| D     | Add CHECKLIST.md, update CLAUDE.md                | `Document new agent architecture (CLAUDE.md, CHECKLIST.md)` |
| E     | Run full test suite + smoke benches               | (no commit — verification only) |

---

## Phase A — Hoist filter fragment + add BaseAgent slots

**Files:** `core/tools.py`, `core/chroma_tools.py`,
`core/agent_dd_tools.py`, `agents/base.py`, every
`agents/<name>/agent.py`.

### A.1 Make `EXA_SEARCH_FILTERS` public

`core/tools.py:17`: rename `_EXA_FILTERS` → `EXA_SEARCH_FILTERS`. Update
`core/chroma_tools.py:10` to import the new name. The leading underscore
was the wrong signal — this is a *public* shared fragment.

### A.2 Compose filters in agent_dd

`core/agent_dd_tools.py:31-52`: replace the hand-rolled `search` tool
with one that composes `EXA_SEARCH_FILTERS`:

```python
from core.tools import CANONICAL_TOOLS as _LEAN_TOOLS, EXA_SEARCH_FILTERS

CANONICAL_AGENT_DD_TOOLS = {
    "search": {
        "description": (
            "Web search via Exa. Returns the top 5 URLs with short "
            "extractive highlight chunks (~200 chars each) ranked against "
            "your query. Every chunk gets a snippet id like S_abc123 that "
            "you MUST cite via the `answer` tool. Optional filters "
            "(category, date range, domain include/exclude) are available."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Be specific. One unknown per search. "
                    "If you need independent sub-queries, emit multiple "
                    "parallel `search` calls in the same turn — all "
                    "parallel calls in one turn cost 1 cycle."
                ),
            },
            **EXA_SEARCH_FILTERS,
        },
        "required": ["query"],
    },
    # browse_page, commit_memory, answer all stay as-is
    ...
}
```

### A.3 Add class slots to `BaseAgent`

`agents/base.py`: extend the class with three optional class attributes
that subclasses populate:

```python
class BaseAgent(ABC):
    """..."""

    # Class-level slots populated by subclasses. Optional — agents
    # without LLM tool calls (e.g. exa_deep) leave them as None.
    TOOL_SCHEMAS: dict | None = None    # canonical name-keyed schema dict
    SYSTEM_PROMPT: str | None = None    # raw template (with {placeholders})

    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        ...
```

### A.4 Populate slots on every agent class (current paths)

Set the slots on each agent class using the *current* import paths.
This makes Phase B file moves a pure relocation — the class attrs
already exist, only the import line changes.

- `agents/lean_searcher/agent.py`: `from core.tools import CANONICAL_TOOLS as _SCHEMAS` and `from core.prompts import SEARCHER_PROMPT as _PROMPT`. Set `TOOL_SCHEMAS = _SCHEMAS; SYSTEM_PROMPT = _PROMPT`.
- `agents/agent_dd/agent.py`: similar with `core.agent_dd_tools` and `core.agent_dd_prompts`.
- `agents/chroma_agent/agent.py`: similar with `core.chroma_tools` and `core.prompts.CHROMA_SEARCHER_PROMPT`.
- `agents/exa_deep/agent.py`: `TOOL_SCHEMAS = None; SYSTEM_PROMPT = None` (explicit, not implicit).

### A.5 Acceptance

```bash
uv run python -c "
from core.tools import EXA_SEARCH_FILTERS
from agents.registry import discover_agents
agents = discover_agents()
for name, cls in agents.items():
    schema = getattr(cls, 'TOOL_SCHEMAS', 'MISSING')
    print(f'{name}: TOOL_SCHEMAS={\"set\" if schema else \"None\"}, SYSTEM_PROMPT={\"set\" if cls.SYSTEM_PROMPT else \"None\"}')
print('agent_dd has filters:', 'category' in agents['agent_dd'].TOOL_SCHEMAS['search']['properties'])
"
# expected: every agent listed; agent_dd has filters: True

uv run pytest tests/ -v
# expected: all pass
```

---

## Phase B0 — Extract shared harness helpers

**Files:** `core/harness.py` (split), new `core/llm.py`, new
`core/scratchpad.py`.

The three harnesses (`SearcherHarness`, `AgentDDHarness`,
`ChromaHarness`) all reach into `core/harness.py` for the same
client/retry/fuzzy helpers. Pull them out so the future per-agent
harness files can import them without circular deps.

### B0.1 Create `core/llm.py`

Extract from `core/harness.py:41-84`:

```python
"""Shared Anthropic-compatible LLM client + retry helpers.

Split out of core/harness.py so per-agent harness files
(agents/<name>/harness.py) can import these without pulling in
SearcherHarness.
"""

from __future__ import annotations

import os
import time

import anthropic

from core.console import console

_RETRY_DELAYS = [15, 30, 45]
_OPENROUTER_BASE = "https://openrouter.ai/api"


def make_client(provider: str, base_url: str = "", api_key_env: str = "") -> anthropic.Anthropic:
    """..."""  # unchanged body


def llm_call(client, **kwargs):
    """..."""  # unchanged body — note rename: _llm_call → llm_call (public)
```

### B0.2 Create `core/scratchpad.py`

Extract `_fuzzy_replace` from `core/harness.py:90-153`:

```python
"""Scratchpad text-edit primitive — fuzzy substring replace.

Used by lean_searcher (commit_memory) and agent_dd (commit_memory).
Three-tier match: exact → whitespace-normalized → rapidfuzz 95% partial.
"""

from __future__ import annotations

import re


def fuzzy_replace(text: str, old: str, new: str) -> tuple[str, bool]:
    """..."""  # unchanged body — rename `_fuzzy_replace` → `fuzzy_replace` (public)
```

### B0.3 Update `core/harness.py` imports

Replace local definitions with `from core.llm import ...`,
`from core.scratchpad import fuzzy_replace`. Keep
`SearcherHarness` and the other lean_searcher-specific helpers in
place — they move out in Phase B3.

### B0.4 Update `core/agent_dd_harness.py` and `core/chroma_harness.py`

Switch their `from core.harness import _fuzzy_replace, _RETRY_DELAYS, _llm_call, make_client`
imports to use `core.llm` and `core.scratchpad`.

Note the rename: `_fuzzy_replace` → `fuzzy_replace`, `_llm_call` →
`llm_call`. Update call sites accordingly.

### B0.5 Acceptance

```bash
uv run pytest tests/test_harness.py tests/test_core_parity.py -v
# expected: all pass — these tests don't touch the renamed symbols directly
# but the rename will have updated their import sites if they did

uv run python -c "
from core.llm import make_client, llm_call
from core.scratchpad import fuzzy_replace
print('helpers exposed correctly')
"
```

---

## Phase B1 — Migrate agent_dd

**Files:** `core/agent_dd_*.py` → `agents/agent_dd/{tools,prompts,harness}.py`.

### B1.1 Create `agents/agent_dd/tools.py`

Move the body of `core/agent_dd_tools.py` verbatim, updating
imports:
- `from core.tools import CANONICAL_TOOLS as _LEAN_TOOLS, EXA_SEARCH_FILTERS`
  stays (those are the genuinely shared bits left in `core/tools.py`).

### B1.2 Create `agents/agent_dd/prompts.py`

Move `core/agent_dd_prompts.py` verbatim. No import changes.

### B1.3 Create `agents/agent_dd/harness.py`

Move `core/agent_dd_harness.py` verbatim, updating imports:
- `from core.agent_dd_prompts import AGENT_DD_SYSTEM_PROMPT`
  → `from agents.agent_dd.prompts import AGENT_DD_SYSTEM_PROMPT`
- `from core.agent_dd_tools import AGENT_DD_ANTHROPIC_TOOLS`
  → `from agents.agent_dd.tools import AGENT_DD_ANTHROPIC_TOOLS`
- `from core.harness import _fuzzy_replace`
  → `from core.scratchpad import fuzzy_replace` (already done in B0)

### B1.4 Update `agents/agent_dd/agent.py`

- `from core.agent_dd_harness import AgentDDHarness`
  → `from agents.agent_dd.harness import AgentDDHarness`
- Replace the Phase A.4 import lines:
  - `from agents.agent_dd.tools import CANONICAL_AGENT_DD_TOOLS`
  - `from agents.agent_dd.prompts import AGENT_DD_SYSTEM_PROMPT`
- `TOOL_SCHEMAS = CANONICAL_AGENT_DD_TOOLS`
- `SYSTEM_PROMPT = AGENT_DD_SYSTEM_PROMPT`

### B1.5 Update `sft/convert.py` consumers

Two import sites:
- `sft/convert.py:464`: `from core.agent_dd_prompts import AGENT_DD_SYSTEM_PROMPT`
  → `from agents.agent_dd.prompts import AGENT_DD_SYSTEM_PROMPT`
- `sft/convert.py:558`: `from core.agent_dd_tools import AGENT_DD_OPENAI_TOOLS`
  → `from agents.agent_dd.tools import AGENT_DD_OPENAI_TOOLS`

### B1.6 Delete the old `core/agent_dd_*.py`

Only after every consumer is updated. Verify with:

```bash
grep -rn "core\.agent_dd_" --include="*.py" .
# expected: empty
rm core/agent_dd_tools.py core/agent_dd_prompts.py core/agent_dd_harness.py
```

### B1.7 Acceptance

```bash
uv run pytest tests/ -v
# expected: all pass

uv run python -c "
from agents.registry import load_agent
agent = load_agent('agent_dd', model='claude_sonnet')
print(agent.TOOL_SCHEMAS.keys())
print('category' in agent.TOOL_SCHEMAS['search']['properties'])
"
# expected: dict_keys(['search', 'browse_page', 'commit_memory', 'answer']); True
```

---

## Phase B2 — Migrate chroma_agent

**Files:** `core/chroma_*.py` → `agents/chroma_agent/{tools,prompts,harness}.py`.

### B2.1 Create `agents/chroma_agent/tools.py`

Move `core/chroma_tools.py` verbatim. Update import:
- `from core.tools import _EXA_FILTERS` (Phase A renamed this)
  → `from core.tools import EXA_SEARCH_FILTERS as _EXA_FILTERS`
  OR rewrite the spread to use the new name throughout.

### B2.2 Create `agents/chroma_agent/prompts.py`

Move ONLY `CHROMA_SEARCHER_PROMPT` from `core/prompts.py:161-208` into
the new file. Leave the other prompts in `core/prompts.py`. The new
file's first line should be a single import:

```python
"""Chroma harness system prompt."""

CHROMA_SEARCHER_PROMPT = """\
..."""
```

### B2.3 Create `agents/chroma_agent/harness.py`

Move `core/chroma_harness.py` verbatim. Update imports:
- `from core.chroma_tools import ANTHROPIC_CHROMA_TOOLS`
  → `from agents.chroma_agent.tools import ANTHROPIC_CHROMA_TOOLS`
- `from core.prompts import CHROMA_SEARCHER_PROMPT, THINKING_INSTRUCTION`
  → `from agents.chroma_agent.prompts import CHROMA_SEARCHER_PROMPT`
  → keep the `from core.prompts import THINKING_INSTRUCTION` (still shared)
- `from core.harness import _RETRY_DELAYS, _llm_call, make_client`
  → `from core.llm import _RETRY_DELAYS, llm_call, make_client`
  (and update call sites for the rename)

### B2.4 Update `agents/chroma_agent/agent.py`

- `from core.chroma_harness import ChromaHarness`
  → `from agents.chroma_agent.harness import ChromaHarness`
- Phase A.4 import lines updated to point at new paths.

### B2.5 No `sft/convert.py` consumers — chroma_agent is bench-only.

### B2.6 Delete `core/chroma_*.py` and remove `CHROMA_SEARCHER_PROMPT` from `core/prompts.py`

```bash
grep -rn "core\.chroma_\|CHROMA_SEARCHER_PROMPT" --include="*.py" .
# expected: empty
rm core/chroma_tools.py core/chroma_harness.py
# manually delete the CHROMA_SEARCHER_PROMPT block from core/prompts.py:161-208
```

### B2.7 Acceptance

```bash
uv run pytest tests/ -v
# expected: all pass
```

---

## Phase B3 — Migrate lean_searcher

The trickiest move because `core/tools.py`, `core/prompts.py`, and
`core/harness.py` all carry a *mix* of lean_searcher-specific and
truly-shared content. Be careful what stays vs. moves.

### B3.1 Split `core/tools.py`

After this phase, `core/tools.py` should contain ONLY:

```python
"""Shared Exa search filter set used by every agent that exposes a
search tool. Hoisted here so no agent re-defines filter parameters
from scratch.

Per-agent tool schemas live in `agents/<name>/tools.py` and compose
**EXA_SEARCH_FILTERS into their `search` tool's properties.
"""

EXA_SEARCH_FILTERS: dict = {
    "category": {...},
    "start_published_date": {...},
    "end_published_date": {...},
    "include_domains": {...},
    "exclude_domains": {...},
}
```

Move everything else (`CANONICAL_TOOLS`, `to_anthropic`, `to_openai`,
`ANTHROPIC_TOOLS`, `OPENAI_TOOLS`, the disabled-prune comment block) to
a new file `agents/lean_searcher/tools.py`.

### B3.2 Split `core/prompts.py`

After this phase, `core/prompts.py` should contain ONLY:

- `THINKING_INSTRUCTION` (used by lean_searcher AND chroma_agent)
- `BROWSE_EXTRACT_PROMPT` (used by `core/browse.py::BrowseExtractor`,
  shared)
- `EXTRACTOR_PROMPT` (used by `core/extractor.py::Extractor`, shared)

Move `SEARCHER_PROMPT` and `SEARCHER_PROMPT_NO_SCRATCHPAD` to
`agents/lean_searcher/prompts.py`.

(The chroma prompt was already moved in B2.)

### B3.3 Split `core/harness.py`

After this phase, `core/harness.py` should be DELETED entirely. Its
remaining contents should be:

- Already moved in B0: `make_client`, `_llm_call` → `core/llm.py`
- Already moved in B0: `_fuzzy_replace` → `core/scratchpad.py`
- `SearcherHarness` class → `agents/lean_searcher/harness.py`

`agents/lean_searcher/harness.py` imports:
- `from core.llm import make_client, llm_call`
- `from core.scratchpad import fuzzy_replace`
- `from core.tools import EXA_SEARCH_FILTERS` (only if needed)
- `from agents.lean_searcher.tools import ANTHROPIC_TOOLS`
- `from agents.lean_searcher.prompts import SEARCHER_PROMPT, SEARCHER_PROMPT_NO_SCRATCHPAD`
- `from core.prompts import THINKING_INSTRUCTION` (still shared)

### B3.4 Update `agents/lean_searcher/agent.py`

- `from core.harness import SearcherHarness`
  → `from agents.lean_searcher.harness import SearcherHarness`
- Phase A.4 import lines updated.

### B3.5 Update `sft/convert.py` consumers

- `sft/convert.py:65`: `from core.prompts import SEARCHER_PROMPT, THINKING_INSTRUCTION`
  → `from agents.lean_searcher.prompts import SEARCHER_PROMPT`
  → `from core.prompts import THINKING_INSTRUCTION` (still core)
- `sft/convert.py:66`: `from core.tools import OPENAI_TOOLS`
  → `from agents.lean_searcher.tools import OPENAI_TOOLS`

### B3.6 Update `tests/test_harness.py`

- `from core.harness import _fuzzy_replace`
  → `from core.scratchpad import fuzzy_replace`
- `from core.harness import ANTHROPIC_TOOLS`
  → `from agents.lean_searcher.tools import ANTHROPIC_TOOLS`
- `from core.harness import SearcherHarness`
  → `from agents.lean_searcher.harness import SearcherHarness`

### B3.7 Delete `core/harness.py`

```bash
grep -rn "from core\.harness\|import core\.harness" --include="*.py" .
# expected: empty
rm core/harness.py
```

### B3.8 Acceptance

```bash
uv run pytest tests/ -v
# expected: all pass

uv run python -c "
from agents.lean_searcher.harness import SearcherHarness
from agents.lean_searcher.tools import CANONICAL_TOOLS, ANTHROPIC_TOOLS, OPENAI_TOOLS
from agents.lean_searcher.prompts import SEARCHER_PROMPT
print('lean_searcher migrated')
"
```

---

## Phase C — Rewrite parity test

**File:** `tests/test_core_parity.py`.

The current parity test was written when only lean_searcher existed.
Many of its `is`-identity checks become moot once schemas live per-agent.
Replace with a parity test that iterates over every discovered agent.

### C.1 New parity test layout

```python
"""Parity tests — every agent must conform to the BaseAgent contract.

Iterates over every agent discovered via agents.registry.discover_agents()
and asserts:
1. TOOL_SCHEMAS is either None or a dict.
2. SYSTEM_PROMPT is either None or a non-empty string.
3. If TOOL_SCHEMAS contains a "search" tool, that tool MUST compose
   EXA_SEARCH_FILTERS — never re-author the filter params.

This is the structural CI guard that prevents the agent_dd filter-drop
regression from recurring on a new agent.
"""

from __future__ import annotations

import pytest

from agents.registry import discover_agents
from core.tools import EXA_SEARCH_FILTERS

_AGENT_CLASSES = discover_agents()
_AGENT_PARAMS = sorted(_AGENT_CLASSES.items())


@pytest.mark.parametrize("name,cls", _AGENT_PARAMS)
def test_agent_class_slots_present(name, cls):
    """Every agent class declares TOOL_SCHEMAS and SYSTEM_PROMPT slots."""
    assert hasattr(cls, "TOOL_SCHEMAS"), (
        f"{name} missing TOOL_SCHEMAS class slot. Set to None for "
        f"non-tool-calling agents (see agents/exa_deep/agent.py)."
    )
    assert hasattr(cls, "SYSTEM_PROMPT")
    if cls.TOOL_SCHEMAS is not None:
        assert isinstance(cls.TOOL_SCHEMAS, dict)
    if cls.SYSTEM_PROMPT is not None:
        assert isinstance(cls.SYSTEM_PROMPT, str)
        assert cls.SYSTEM_PROMPT.strip()


@pytest.mark.parametrize("name,cls", _AGENT_PARAMS)
def test_agent_search_includes_canonical_exa_filters(name, cls):
    """If an agent exposes a `search` tool, it MUST include the canonical
    Exa filter set. Dropping a filter at the schema level propagates
    through synth → SFT → trained model and is unrecoverable without a
    re-run.
    """
    if cls.TOOL_SCHEMAS is None:
        pytest.skip(f"{name} declares no TOOL_SCHEMAS")
    # lean_searcher uses `exa_search`; agent_dd / chroma use `search`.
    # Both names are valid — the rule applies to whichever name an
    # agent uses for Exa search.
    search_keys = [k for k in ("search", "exa_search") if k in cls.TOOL_SCHEMAS]
    if not search_keys:
        pytest.skip(f"{name} exposes no search tool")
    search_props = cls.TOOL_SCHEMAS[search_keys[0]]["properties"]
    missing = [k for k in EXA_SEARCH_FILTERS if k not in search_props]
    assert not missing, (
        f"{name}'s search tool dropped canonical Exa filters: {missing}. "
        f"Compose `**EXA_SEARCH_FILTERS` from core.tools — never re-author."
    )


# Keep the still-applicable shared-symbol tests (these survive)

def test_context_block_builders_are_single_source():
    from core import context as core_context
    from sft import convert as sft_convert
    assert sft_convert.live_state_block is core_context.live_state_block
    assert sft_convert.exa_api_block is core_context.exa_api_block


def test_exa_client_summary_only_contract():
    """..."""  # unchanged from current test


def test_live_state_block_shape(...):
    """..."""  # unchanged from current test
```

The OLD tests that get DELETED:
- `test_prompts_are_single_source` — replaced by class-slot iteration
- `test_openai_tools_are_single_source` — replaced by class-slot iteration
- `test_anthropic_tools_are_single_source` — replaced by class-slot iteration
- `test_tool_name_set_matches_across_formats` — was hardcoded to lean_searcher's three tool names; superseded by per-agent format-pair check (which we add as part of this rewrite)
- `test_tool_required_fields_match` — same; superseded

Add a new parametrized test asserting that for each agent with
`TOOL_SCHEMAS`, the per-agent `to_anthropic()` and `to_openai()` (when
present) cover the same names with matching `required` lists.

### C.2 Acceptance — deliberate-break verification

This is critical. The parity test exists to catch the agent_dd
regression. Verify it does so.

```bash
# 1. All tests green
uv run pytest tests/test_core_parity.py -v
# expected: all parametrized tests pass for all 4 agents

# 2. Deliberate break (REVERT BEFORE COMMITTING):
# Edit agents/agent_dd/tools.py — remove `**EXA_SEARCH_FILTERS` from search
uv run pytest tests/test_core_parity.py::test_agent_search_includes_canonical_exa_filters -v
# expected: FAIL with "agent_dd's search tool dropped canonical Exa filters: [...]"

# 3. Revert the deliberate break.
```

---

## Phase D — CHECKLIST.md and CLAUDE.md

### D.1 New file: `agents/_template/CHECKLIST.md`

```markdown
# Checklist for adding a new agent

Before opening a PR for a new agent under `agents/<your_name>/`:

- [ ] `agent.py` defines exactly one `BaseAgent` subclass
- [ ] Set `TOOL_SCHEMAS` on the class (or `None` if no tool calls)
- [ ] Set `SYSTEM_PROMPT` on the class (or `None` if no system prompt)
- [ ] If your agent exposes a `search` (or `exa_search`) tool, it MUST
      compose `**EXA_SEARCH_FILTERS` from `core.tools` — never
      re-author filter parameters
- [ ] Tools / prompts / harness live in
      `agents/<your_name>/{tools,prompts,harness}.py`. Anything that's
      *truly* shared with another agent goes in `core/` (and probably
      already does — check there first)
- [ ] `uv run pytest tests/test_core_parity.py` passes locally —
      including the parametrized check that picks up your agent
      automatically via `discover_agents()`
- [ ] If your agent emits trajectories with novel fields, document
      the schema and update `sft/convert.py` to handle them — do NOT
      fork the converter

These rules exist because synth → SFT → trained-model is a one-way
pipeline. Schema drops at any layer corrupt all downstream artifacts
and are only fixable by full re-runs. The parity test catches the most
common drop (filter set on `search`); the rest depend on author
discipline.
```

### D.2 Update CLAUDE.md

Update the "Non-negotiable architectural rules" section:

- Rule 1 stays: `core/` is the single source of truth for *truly
  shared* assets (filter fragment, helpers, context builders, exa
  client, types).
- Rule 2 stays: never fork a prompt.
- Rule 3 stays: agents ≠ models.
- Rule 4 stays: scratchpad tool returns status only.
- Rule 5 stays: Exa client uses `summary` mode only.
- Rule 6 stays: source bank is a Trace field.
- **Add Rule 7:** Each agent owns its tool schemas, prompts, and
  harness inside `agents/<name>/`. `BaseAgent.TOOL_SCHEMAS` and
  `BaseAgent.SYSTEM_PROMPT` are class slots; the parity test
  enforces presence and Exa-filter conformance.

Update the "Tech stack" / file layout commentary to reflect the new
layout described above.

---

## Phase E — Final verification

After every phase has landed:

```bash
# 1. Full test suite
uv run pytest tests/ -v
# expected: all green

# 2. Smoke benches (each uses a small handful of API calls)
make bench AGENT=lean_searcher MODEL=claude_sonnet SPLIT=dev
make bench AGENT=agent_dd MODEL=claude_sonnet DATASET=filterbench SPLIT=dev
make bench AGENT=chroma_agent MODEL=claude_sonnet SPLIT=dev
# expected: all complete without import errors or schema rejections

# 3. Synth dry-run (uses load_agent which exercises the new layout end-to-end)
uv run python -m synth.generate_agent_dd \
    --model claude_sonnet --source filterbench --split dev --concurrent 1
# expected: 3 traces written to trajectories/agent_dd/ — bail with Ctrl+C
# after one task completes; we just want to confirm the path resolves.
```

---

## Rollback

Each phase is its own commit. To roll back a single phase:

```bash
git revert <phase-commit-hash>
uv run pytest tests/ -v
# expected: all green at that intermediate state
```

To roll back the entire refactor:

```bash
git revert --no-commit <first-commit>..<last-commit>
git commit -m "Revert agent refactor"
uv run pytest tests/ -v
```

No data files modified. No models retrained. No trajectories regenerated.

---

## After this lands

User has flagged that there is **additional pre-rerun investigation** to
do before regenerating synth data. Do **not** re-run synth or retrain
after this lands — wait for explicit go-ahead.

When the synth re-run does happen, the natural follow-up is:
- Update `agents/agent_dd/prompts.py::AGENT_DD_SYSTEM_PROMPT` with
  filter-usage guidance for the teacher
- Re-generate trajectories via `synth/generate_agent_dd.py`
- Re-convert via `sft/convert.py`
- Retrain LoRA

But none of that happens inside this plan.
