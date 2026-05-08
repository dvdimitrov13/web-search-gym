"""Tool schemas for lean_searcher — exa_search / commit_memory / submit.

Two wire formats:
- **Anthropic messages API** (`name`, `description`, `input_schema`) —
  used at inference time by the lean_searcher harness.
- **OpenAI function-calling** (`type: "function"`, nested `function`
  object with `parameters`) — used by `sft/convert.py` when building
  Qwen3 training examples via `tokenizer.apply_chat_template(..., tools=...)`.

The canonical definition lives in CANONICAL_TOOLS. `to_anthropic()` /
`to_openai()` convert on demand. Never duplicate a schema — always call
the converters.
"""

from __future__ import annotations

from core.tools import COMMIT_MEMORY_SPEC, EXA_SEARCH_FILTERS

# Canonical schemas, name-keyed. Each value is a dict with the fields needed
# to render into both wire formats.
CANONICAL_TOOLS = {
    "exa_search": {
        "description": (
            "Search the web using Exa's neural search engine. "
            "Returns relevant web pages with titles, URLs, and summaries. "
            "Two query fields control retrieval and summarization separately: "
            "`query` picks which URLs come back; `summary_query` (optional) "
            "shapes the per-URL summary. If summary_query is omitted the "
            "search query is used for both. Use a distinct summary_query when "
            "you want to retrieve BROAD (big candidate pool) but extract "
            "NARROW (specific fact, person, date, quantity). Example: "
            "query='Manoj Bajpayee filmography', "
            "summary_query='son gets a plaster cast'."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Retrieval query. Controls which URLs Exa returns. "
                    "Be specific and detailed."
                ),
            },
            "summary_query": {
                "type": "string",
                "description": (
                    "Optional. Controls what each per-URL summary focuses on "
                    "(not which URLs are returned). Default: same as `query`. "
                    "Use a distinct summary_query when the search needs to cast "
                    "a wide net but the summary should extract one specific "
                    "detail."
                ),
            },
            **EXA_SEARCH_FILTERS,
        },
        "required": ["query"],
    },
    "commit_memory": COMMIT_MEMORY_SPEC,
    # Disabled — kept here (not in CANONICAL_TOOLS) so we can re-enable easily.
    # Sonnet declined to use this tool on 3 noisy smoke tasks; re-evaluate when
    # a prompt instruction or a weaker model makes adoption more likely.
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
