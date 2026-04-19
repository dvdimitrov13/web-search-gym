"""Harness primitive tests — no network or LLM calls."""

from __future__ import annotations


def test_fuzzy_replace_exact():
    from core.harness import _fuzzy_replace

    out, matched = _fuzzy_replace("hello world", "world", "there")
    assert matched is True
    assert out == "hello there"


def test_fuzzy_replace_whitespace_normalized():
    from core.harness import _fuzzy_replace

    out, matched = _fuzzy_replace("line1\n\nline2", "line1 line2", "ok")
    assert matched is True
    assert "ok" in out


def test_fuzzy_replace_missing():
    from core.harness import _fuzzy_replace

    out, matched = _fuzzy_replace("hello world", "nope", "x")
    assert matched is False
    assert out == "hello world"


def test_tool_set_matches_canonical():
    """Harness-exported ANTHROPIC_TOOLS has every canonical tool."""
    from core.harness import ANTHROPIC_TOOLS

    names = {t["name"] for t in ANTHROPIC_TOOLS}
    assert names == {"exa_search", "scratchpad", "submit"}


def test_system_prompt_contains_date_and_budget():
    """_build_system renders SEARCHER_PROMPT placeholders."""
    # Build a harness without triggering API clients — we only need _build_system.
    from core.harness import SearcherHarness

    # Instantiate with dummy creds; Exa client construction is lazy via default.
    import os
    os.environ.setdefault("EXA_API_KEY", "test")
    h = SearcherHarness(
        searcher_model="claude-sonnet-4-5",
        max_searches=7,
        scratchpad_max_tokens=256,
    )
    sys_text = h._build_system()
    assert "Today's date is" in sys_text
    assert "7" in sys_text  # max_searches
    assert "<exa_api>" in sys_text
