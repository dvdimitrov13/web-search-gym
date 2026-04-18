"""CRITICAL: every downstream module must use the SAME prompts, tools, and
context builders. These tests fail loudly if anyone forks the source of truth.

Prevents Issue 007 #1, #2, #3, #5 from ever returning.
"""

from __future__ import annotations

import pytest


def test_prompts_are_single_source():
    """SEARCHER_PROMPT and THINKING_INSTRUCTION are imported from core/prompts,
    not duplicated anywhere else."""
    from core import prompts as core_prompts
    # sft/convert imports these symbols
    from sft import convert as sft_convert

    assert sft_convert.SEARCHER_PROMPT is core_prompts.SEARCHER_PROMPT
    assert sft_convert.THINKING_INSTRUCTION is core_prompts.THINKING_INSTRUCTION


def test_openai_tools_are_single_source():
    """sft/convert.py uses the canonical OPENAI_TOOLS from core.tools."""
    from core import tools as core_tools
    from sft import convert as sft_convert

    assert sft_convert.OPENAI_TOOLS is core_tools.OPENAI_TOOLS


def test_anthropic_tools_are_single_source():
    """core.harness and any agent must use ANTHROPIC_TOOLS from core.tools."""
    from core import harness as core_harness
    from core import tools as core_tools

    assert core_harness.ANTHROPIC_TOOLS is core_tools.ANTHROPIC_TOOLS


def test_context_block_builders_are_single_source():
    """live_state_block and exa_api_block are imported once, from core.context."""
    from core import context as core_context
    from core import harness as core_harness
    from sft import convert as sft_convert

    assert core_harness.live_state_block is core_context.live_state_block
    assert core_harness.exa_api_block is core_context.exa_api_block
    assert sft_convert.live_state_block is core_context.live_state_block
    assert sft_convert.exa_api_block is core_context.exa_api_block


def test_tool_name_set_matches_across_formats():
    """The Anthropic and OpenAI renderings cover the same tool names."""
    from core.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS

    anthropic_names = {t["name"] for t in ANTHROPIC_TOOLS}
    openai_names = {t["function"]["name"] for t in OPENAI_TOOLS}
    assert anthropic_names == openai_names
    assert anthropic_names == {"exa_search", "scratchpad", "submit"}


def test_tool_required_fields_match():
    """For each tool, required-field sets match between Anthropic and OpenAI formats."""
    from core.tools import ANTHROPIC_TOOLS, OPENAI_TOOLS

    by_name_anthropic = {t["name"]: t["input_schema"]["required"] for t in ANTHROPIC_TOOLS}
    by_name_openai = {
        t["function"]["name"]: t["function"]["parameters"]["required"] for t in OPENAI_TOOLS
    }
    assert by_name_anthropic == by_name_openai


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
    assert "<scratchpad>" in out
    assert "</scratchpad>" in out
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
