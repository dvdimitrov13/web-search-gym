"""Canonical tool schemas. Single source of truth for exa_search / scratchpad / submit.

Two wire formats are needed in this project:

- **Anthropic messages API** (`name`, `description`, `input_schema`) — used at
  inference time by `core/harness.py`.
- **OpenAI function-calling** (`type: "function"`, nested `function` object with
  `parameters`) — used by `sft/convert.py` when building Qwen3 training examples
  via `tokenizer.apply_chat_template(..., tools=...)`.

The canonical definition lives in CANONICAL_TOOLS. `to_anthropic()` / `to_openai()`
convert on demand. Never duplicate a schema; always call the converters.
"""

# The Exa filters the searcher is allowed to pass alongside `query`.
# Keep this list in sync with the <exa_api> block rendered by core/context.py.
_EXA_FILTERS: dict = {
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

# Canonical schemas, name-keyed. Each value is a dict with the fields needed
# to render into both wire formats.
CANONICAL_TOOLS = {
    "exa_search": {
        "description": (
            "Search the web using Exa's neural search engine. "
            "Returns relevant web pages with titles, URLs, and summaries. "
            "Pass 'query' (required) plus any optional Exa filters as a flat "
            "JSON object."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Be specific and detailed.",
            },
            **_EXA_FILTERS,
        },
        "required": ["query"],
    },
    "scratchpad": {
        "description": (
            "A persistent scratchpad document you can edit in place. "
            "Use it to write your search plan, track findings, and note "
            "coverage gaps. "
            "To create or overwrite: provide only new_text. "
            "To edit in place: provide old_text (substring to find) and "
            "new_text (replacement). "
            "To delete a section: provide old_text and set new_text to empty string."
        ),
        "properties": {
            "old_text": {
                "type": "string",
                "description": (
                    "Substring to find in the current scratchpad. "
                    "Omit to overwrite the entire scratchpad."
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
    },
    # Disabled — kept here (not in CANONICAL_TOOLS) so we can re-enable easily.
    # The companion handler in core/harness.py is also disabled. Sonnet declined
    # to use this tool on 3 noisy smoke tasks; re-evaluate when a prompt
    # instruction or a weaker model makes adoption more likely.
    # "prune": {
    #     "description": (
    #         "Remove URLs from the source bank. Use this to drop irrelevant "
    #         "results that shouldn't feed the downstream extractor — noisy "
    #         "pages, off-topic hits, duplicates, wrong entities with the same "
    #         "name. Pruned URLs are permanently removed from this run's source "
    #         "bank, even if a later search re-surfaces them."
    #     ),
    #     "properties": {
    #         "urls": {
    #             "type": "array",
    #             "description": "URLs to remove from the source bank.",
    #             "items": {"type": "string"},
    #         },
    #         "reason": {
    #             "type": "string",
    #             "description": (
    #                 "One short sentence on why these URLs are being pruned "
    #                 "(e.g., 'off-topic product listings', 'different person "
    #                 "with same name')."
    #             ),
    #         },
    #     },
    #     "required": ["urls"],
    # },
    "submit": {
        "description": (
            "Submit your final ranked list of relevant URLs. "
            "Call this when you have finished searching and are ready to hand "
            "off to the downstream stage."
        ),
        "properties": {
            "urls": {
                "type": "array",
                "description": "Ranked list of relevant URLs (most important first).",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The page URL.",
                        },
                        "score": {
                            "type": "number",
                            "description": "Relevance score from 0 to 1.",
                        },
                    },
                    "required": ["url", "score"],
                },
            },
        },
        "required": ["urls"],
    },
}


def to_anthropic() -> list[dict]:
    """Render CANONICAL_TOOLS into Anthropic messages API format."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
                # exa_search allows flat filter passthrough.
                **({"additionalProperties": True} if name == "exa_search" else {}),
            },
        }
        for name, spec in CANONICAL_TOOLS.items()
    ]


def to_openai() -> list[dict]:
    """Render CANONICAL_TOOLS into OpenAI function-calling format (Qwen3-compatible)."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                },
            },
        }
        for name, spec in CANONICAL_TOOLS.items()
    ]


# Pre-rendered, cached for convenience. Importers can use either the callables
# above (for one-off renders) or these module-level constants.
ANTHROPIC_TOOLS = to_anthropic()
OPENAI_TOOLS = to_openai()
