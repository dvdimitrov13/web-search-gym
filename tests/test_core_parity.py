"""Parity tests — every agent must conform to the BaseAgent contract.

The structural-CI guard for the agent_dd filter-drop class of bug:
parametrizes over every discovered agent and asserts that any `search`
or `exa_search` tool composes EXA_SEARCH_FILTERS. Adding a new agent
is automatically picked up — no hand-maintained registry to forget.

Also checks that shared-module imports stay single-sourced (no forking
prompts or context builders) and that lean_searcher's two wire formats
agree.
"""

from __future__ import annotations

import pytest

from agents.registry import discover_agents
from core.tools import EXA_SEARCH_FILTERS

# Discover once at module load. Test parametrize sees the live set.
_AGENT_CLASSES = discover_agents()
_AGENT_PARAMS = sorted(_AGENT_CLASSES.items())


# ── Per-agent class-slot conformance (the structural CI guard) ─────


@pytest.mark.parametrize("name,cls", _AGENT_PARAMS)
def test_agent_class_slots_present(name, cls):
    """Every agent declares TOOL_SCHEMAS and SYSTEM_PROMPT slots."""
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
    """If an agent exposes a `search` (or `exa_search`) tool, it MUST
    include the canonical Exa filter set. Dropping a filter at the
    schema level propagates through synth → SFT → trained model and is
    unrecoverable without a re-run.
    """
    if cls.TOOL_SCHEMAS is None:
        pytest.skip(f"{name} declares no TOOL_SCHEMAS")
    search_keys = [k for k in ("exa_search", "search") if k in cls.TOOL_SCHEMAS]
    if not search_keys:
        pytest.skip(f"{name} exposes no search tool")
    search_props = cls.TOOL_SCHEMAS[search_keys[0]]["properties"]
    missing = [k for k in EXA_SEARCH_FILTERS if k not in search_props]
    assert not missing, (
        f"{name}'s search tool dropped canonical Exa filters: {missing}. "
        f"Compose `**EXA_SEARCH_FILTERS` from core.tools — never re-author."
    )


# ── Shared-module single-source-of-truth checks ────────────────────


def test_prompts_are_single_source():
    """sft/convert.py imports the same SEARCHER_PROMPT object as
    lean_searcher's harness, and the same THINKING_INSTRUCTION as core."""
    from agents.lean_searcher import prompts as ls_prompts
    from core import prompts as core_prompts
    from sft import convert as sft_convert

    assert sft_convert.SEARCHER_PROMPT is ls_prompts.SEARCHER_PROMPT
    assert sft_convert.THINKING_INSTRUCTION is core_prompts.THINKING_INSTRUCTION


def test_openai_tools_are_single_source():
    """sft/convert.py uses the canonical OPENAI_TOOLS from lean_searcher."""
    from agents.lean_searcher import tools as ls_tools
    from sft import convert as sft_convert

    assert sft_convert.OPENAI_TOOLS is ls_tools.OPENAI_TOOLS


def test_anthropic_tools_are_single_source():
    """lean_searcher's harness and tools module reach the same object."""
    from agents.lean_searcher import harness as ls_harness
    from agents.lean_searcher import tools as ls_tools

    assert ls_harness.ANTHROPIC_TOOLS is ls_tools.ANTHROPIC_TOOLS


def test_context_block_builders_are_single_source():
    """live_state_block and exa_api_block are imported once, from core.context."""
    from agents.lean_searcher import harness as ls_harness
    from core import context as core_context
    from sft import convert as sft_convert

    assert ls_harness.live_state_block is core_context.live_state_block
    assert ls_harness.exa_api_block is core_context.exa_api_block
    assert sft_convert.live_state_block is core_context.live_state_block
    assert sft_convert.exa_api_block is core_context.exa_api_block


# ── lean_searcher wire-format agreement ────────────────────────────


def test_tool_name_set_matches_across_formats():
    """The Anthropic and OpenAI renderings cover the same tool names."""
    from agents.lean_searcher.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS

    anthropic_names = {t["name"] for t in ANTHROPIC_TOOLS}
    openai_names = {t["function"]["name"] for t in OPENAI_TOOLS}
    assert anthropic_names == openai_names
    assert anthropic_names == {"exa_search", "commit_memory", "submit"}


def test_tool_required_fields_match():
    """For each tool, required-field sets match between Anthropic and OpenAI formats."""
    from agents.lean_searcher.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS

    by_name_anthropic = {t["name"]: t["input_schema"]["required"] for t in ANTHROPIC_TOOLS}
    by_name_openai = {
        t["function"]["name"]: t["function"]["parameters"]["required"] for t in OPENAI_TOOLS
    }
    assert by_name_anthropic == by_name_openai


# ── Other shared-contract guards ───────────────────────────────────


@pytest.mark.parametrize(
    "max_searches,scratchpad,max_tokens",
    [(5, "", 512), (3, "draft notes", 512), (10, "x" * 500, 128)],
)
def test_live_state_block_shape(max_searches, scratchpad, max_tokens):
    """The block must contain the <budget> and <scratchpad> tags."""
    from core.context import live_state_block

    out = live_state_block(
        search_count=1,
        max_searches=max_searches,
        scratchpad=scratchpad,
        scratchpad_max_tokens=max_tokens,
    )
    assert "<budget>" in out
    assert "</budget>" in out
    assert "<commit_memory>" in out
    assert "</commit_memory>" in out
    assert f"1/{max_searches}" in out


def test_exa_client_summary_only_contract():
    """The Exa client's search signature hasn't regressed into highlights mode.

    Checks for actual API usage patterns (`highlights=` as a kwarg or
    `"highlights":` as a dict key), not the bare word — the file may mention
    highlights in a comment explaining why we don't use it.
    """
    import inspect
    import re

    from core.exa_client import ExaClient

    # Strip comments before checking, so prose mentioning "highlights" doesn't
    # trip the regex.
    src = inspect.getsource(ExaClient.search)
    code_only = re.sub(r"#.*", "", src)
    assert '"summary"' in code_only or "'summary'" in code_only

    bad_patterns = [r"highlights\s*=", r"""["']highlights["']"""]
    for pat in bad_patterns:
        assert not re.search(pat, code_only), (
            "core/exa_client.py must NEVER call Exa with 'highlights'. "
            "Training data uses summary mode; drift here breaks SFT transfer "
            "(Issue 007-5 in the prior repo)."
        )
