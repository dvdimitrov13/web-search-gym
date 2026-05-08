"""Shared tool-schema fragments.

Every per-agent tool schema (in `agents/<name>/tools.py`) that exposes
a `search` or `exa_search` tool spreads `**EXA_SEARCH_FILTERS` into its
`properties`. Never re-author filter parameters from scratch. The
parity test asserts conformance.

`COMMIT_MEMORY_SPEC` is a full per-tool spec (not a fragment) for the
edit-in-place commit_memory tool. lean_searcher and agent_dd both
expose it, with identical schema so the training contract stays
identical across them.

Per-agent tool schemas are rendered to Anthropic / OpenAI wire formats
inside `agents/<name>/tools.py` itself.
"""

# Keep this list in sync with the <exa_api> block rendered by core/context.py.
EXA_SEARCH_FILTERS: dict = {
    "category": {
        "type": "string",
        "description": "Filter by content category",
        "enum": [
            "company", "research paper", "news",
            "personal site", "financial report", "people", "pdf",
        ],
    },
    "start_published_date": {
        "type": "string",
        "description": (
            "Filter results published after this date "
            "(YYYY-MM-DDTHH:MM:SS.000Z)"
        ),
    },
    "end_published_date": {
        "type": "string",
        "description": (
            "Filter results published before this date "
            "(YYYY-MM-DDTHH:MM:SS.000Z)"
        ),
    },
    "include_domains": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Only return results from these domains",
    },
    "exclude_domains": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Exclude results from these domains",
    },
}


# The commit_memory tool schema. Shared verbatim between lean_searcher and
# agent_dd so a student trained on one's traces can be evaluated on the
# other without a tool-name / argument shift.
COMMIT_MEMORY_SPEC: dict = {
    "description": (
        "A persistent working memory you can edit in place. Commit to it "
        "after each search to lock in what you've resolved and what's "
        "still open. "
        "To create or overwrite: provide only new_text. "
        "To edit in place: provide old_text (substring to find) and "
        "new_text (replacement). "
        "To delete a section: provide old_text and set new_text to empty string."
    ),
    "properties": {
        "old_text": {
            "type": "string",
            "description": (
                "Substring to find in the current memory. "
                "Omit to overwrite the entire memory."
            ),
        },
        "new_text": {
            "type": "string",
            "description": (
                "Replacement text (or full content if old_text is omitted)."
            ),
        },
    },
    "required": ["new_text"],
}
