"""Dynamic context blocks injected into prompts at every turn.

Three blocks, shared between runtime (harness) and training-data conversion
(sft/convert.py). If either caller drifts the format, SFT inference will see
prompts it wasn't trained on. Never fork — always call these helpers.

- `exa_api_block()` — static (except for Exa SDK introspection). Appended to
  the system prompt.
- `live_state_block()` — per-turn. Contains <budget> + <scratchpad>. Lives in
  the last user message (keeps system prompt cacheable).
- `estimate_tokens()` — crude char/4 estimate, good enough for budget math.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Consistent across train/infer."""
    return len(text) // 4


# Default Exa filter signature rendered into the <exa_api> block. The harness
# can override this by passing a custom signature (e.g. one introspected live
# from the Exa SDK). The default string matches what SFT data is trained on.
DEFAULT_EXA_SIGNATURE = """\
  include_domains: list[str]
  exclude_domains: list[str]
  start_crawl_date: str
  end_crawl_date: str
  start_published_date: str
  end_published_date: str
  include_text: list[str]
  exclude_text: list[str]
  category: one of ['company', 'research paper', 'news', 'pdf', 'personal site', 'financial report', 'people']"""


def exa_api_block(signature: str = DEFAULT_EXA_SIGNATURE) -> str:
    """Static block appended to the system prompt describing Exa filters."""
    return (
        "<exa_api>\n"
        "exa_search optional parameters (pass as flat keys alongside query):\n"
        f"{signature}\n"
        "\nDate format: YYYY-MM-DDTHH:MM:SS.000Z (e.g. 2024-01-01T00:00:00.000Z)\n"
        "These filters are optional — only use them when they clearly help narrow "
        "results.\n"
        "</exa_api>"
    )


def live_state_block(
    search_count: int,
    max_searches: int,
    scratchpad: str,
    scratchpad_max_tokens: int,
    label: str = "searches",
    searches_issued: int | None = None,
) -> str:
    """Per-turn <budget> + <scratchpad> block.

    Appended to the last user message (NOT the system prompt) so the system
    prompt stays cacheable across turns and the live state sits right next to
    the model's generation point.

    `label` lets callers swap the budget unit — e.g. "search cycles" for a
    harness that allows parallel searches within a turn. `searches_issued`,
    if set, is shown alongside cycles so the model sees both counters.
    """
    scratchpad_tokens = estimate_tokens(scratchpad)
    remaining = max_searches - search_count
    scratchpad_remaining = max(0, scratchpad_max_tokens - scratchpad_tokens)
    extra = (
        f" · {searches_issued} search call(s) issued so far"
        if searches_issued is not None
        else ""
    )
    return (
        "<budget>\n"
        f"{label}: {search_count}/{max_searches} used "
        f"({remaining} remaining){extra}\n"
        f"commit_memory: ~{scratchpad_tokens}/{scratchpad_max_tokens} tokens used "
        f"(~{scratchpad_remaining} remaining)\n"
        "</budget>\n\n"
        f"<commit_memory>\n{scratchpad}\n</commit_memory>"
    )
