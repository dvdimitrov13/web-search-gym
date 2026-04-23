"""Tool schemas for the agent_dd agent.

Four tools, matching the DR Tulu pattern adapted for our infra:

- `search`         — Exa highlights mode (maxCharacters=200, per-URL chunked
                     extractive snippets). Page text is cached on the side
                     for browse_page. Returns tagged snippets `[S_...]`.
- `browse_page`    — Read a URL's cached full-page text and return a dense
                     Haiku-extracted summary (~256 tokens) focused on a
                     question. Returns a tagged blob `[B_...]`.
- `commit_memory`  — Persistent edit-in-place scratchpad (shared schema
                     with lean_searcher). Free — does NOT consume a cycle.
                     Use it as a constraints/state tracker across cycles.
- `answer`         — Final answer with explicit citation ids. Ends the
                     rollout. Replaces the searcher+extractor split.

Schemas here are Anthropic-native only (we aren't training a Qwen3 model
on this path yet).
"""

from __future__ import annotations

from core.tools import CANONICAL_TOOLS as _LEAN_TOOLS

# Reuse the lean_searcher commit_memory schema verbatim so the training
# contract stays identical across both agents. If lean changes the schema,
# agent_dd picks it up automatically.
_COMMIT_MEMORY_SPEC = _LEAN_TOOLS["commit_memory"]


CANONICAL_AGENT_DD_TOOLS = {
    "search": {
        "description": (
            "Web search. Returns the top 5 URLs with short extractive "
            "highlight chunks (~200 chars each) ranked against your query. "
            "Every chunk gets a snippet id like S_abc123 that you MUST "
            "cite via the `answer` tool. Use this to breadth-scan the web "
            "for candidate URLs before drilling in with `browse_page`."
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
        },
        "required": ["query"],
    },
    "browse_page": {
        "description": (
            "Return an information-dense extractive summary (~256 tokens) "
            "of a page you ALREADY retrieved via `search`. The full page "
            "text was cached at search time, so browse_page has no network "
            "fetch — it only runs the extractor against the cached text "
            "focused on your `question`. Use this when a `search` "
            "highlight showed a URL was on-topic but didn't surface the "
            "exact fact you need. The result gets a snippet id like "
            "B_xyz789 that you cite via `answer`. "
            "IMPORTANT: the URL MUST come from a prior `search` result in "
            "this rollout — browse_page will error on any URL that wasn't "
            "cached by a previous search."
        ),
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "URL to browse — MUST be one surfaced by a prior "
                    "`search` call in this rollout. Typos or invented "
                    "URLs will error."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "What to extract from the page. Be specific about the "
                    "fact/entity/list/detail you want — the extractor "
                    "focuses on exactly this. Vague questions yield vague "
                    "extractions."
                ),
            },
        },
        "required": ["url", "question"],
    },
    "commit_memory": _COMMIT_MEMORY_SPEC,
    "answer": {
        "description": (
            "Submit the final answer. Call this exactly once, when you "
            "have sufficient cited evidence. Ends the rollout. The "
            "`final_answer` is what gets graded — keep it concise and "
            "directly responsive to the original question."
        ),
        "properties": {
            "final_answer": {
                "type": "string",
                "description": (
                    "The concise final answer to the original question. "
                    "Single entity, short phrase, comma-separated list, "
                    "number, date — whatever the question asked for. "
                    "This is what the grader reads."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "Brief reasoning showing how the cited evidence "
                    "supports the answer. Reference snippet ids inline "
                    "like [S_abc] or [B_xyz]."
                ),
            },
            "citations": {
                "type": "array",
                "description": (
                    "Snippet ids from prior tool results (S_* / B_*) that "
                    "directly support the final answer. Do not invent ids."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["final_answer"],
    },
}


def to_anthropic_tools() -> list[dict]:
    """Render to Anthropic messages-API tool format."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
            },
        }
        for name, spec in CANONICAL_AGENT_DD_TOOLS.items()
    ]


def to_openai_tools() -> list[dict]:
    """Render to OpenAI function-calling format — Qwen3 / tokenizer-compatible.

    Used by `sft/convert.py` when building per-turn training examples.
    """
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
        for name, spec in CANONICAL_AGENT_DD_TOOLS.items()
    ]


AGENT_DD_ANTHROPIC_TOOLS = to_anthropic_tools()
AGENT_DD_OPENAI_TOOLS = to_openai_tools()
