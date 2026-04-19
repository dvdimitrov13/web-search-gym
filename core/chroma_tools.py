"""Tool schemas for chroma_agent — search / grep / prune / submit.

Separate from core/tools.py so the lean_searcher training contract stays
pinned (that pipeline must not see these tools). The chroma_agent harness is
eval-only for now.
"""

from __future__ import annotations

from core.tools import _EXA_FILTERS


CANONICAL_CHROMA_TOOLS = {
    "search": {
        "description": (
            "Search the web with Exa and return ranked highlight chunks per "
            "result. Each chunk is a short, query-reranked excerpt from one "
            "page — cheap to read, focused on the query. Use this to discover "
            "new sources. Pass optional filters (category, date range, "
            "domains) as a flat JSON object."
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
    "grep": {
        "description": (
            "Regex (Python syntax) search across the chunks you've surfaced "
            "so far in this run. Returns up to 5 matching chunks with their "
            "source URLs. Use this to pattern-match across prior search "
            "results without spending a new search call — especially useful "
            "for cross-referencing names, numbers, or dates that appeared in "
            "different searches."
        ),
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Python regex pattern. Case-insensitive.",
            },
        },
        "required": ["pattern"],
    },
    "prune": {
        "description": (
            "Drop URLs from your working context. Removed chunks no longer "
            "appear in grep results and are marked as pruned in the past "
            "tool_result history, freeing token budget for more exploration. "
            "Call this when chunks are off-topic, wrong-entity, or no longer "
            "relevant to remaining hops."
        ),
        "properties": {
            "urls": {
                "type": "array",
                "description": "URLs whose chunks to drop entirely.",
                "items": {"type": "string"},
            },
            "reason": {
                "type": "string",
                "description": (
                    "One short sentence: why these URLs are being pruned "
                    "(e.g., 'different person with same name')."
                ),
            },
        },
        "required": ["urls"],
    },
    "submit": {
        "description": (
            "Submit your final ranked list of relevant URLs. Ends the search."
        ),
        "properties": {
            "urls": {
                "type": "array",
                "description": "Ranked list of relevant URLs (most important first).",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "score": {"type": "number", "description": "Relevance 0-1."},
                    },
                    "required": ["url", "score"],
                },
            },
        },
        "required": ["urls"],
    },
}


def to_anthropic() -> list[dict]:
    """Render canonical chroma tools into Anthropic messages API format."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
                **({"additionalProperties": True} if name == "search" else {}),
            },
        }
        for name, spec in CANONICAL_CHROMA_TOOLS.items()
    ]


ANTHROPIC_CHROMA_TOOLS = to_anthropic()
