"""Shared Anthropic-compatible LLM client + retry helpers.

Split out of core/harness.py so per-agent harness files
(agents/<name>/harness.py) can import these without pulling in
SearcherHarness or any of the lean_searcher-specific machinery.
"""

from __future__ import annotations

import os
import time

import anthropic

from core.console import console

_RETRY_DELAYS = [15, 30, 45]
_OPENROUTER_BASE = "https://openrouter.ai/api"


def make_client(
    provider: str,
    base_url: str = "",
    api_key_env: str = "",
) -> anthropic.Anthropic:
    """Create an Anthropic-compatible client.

    - `anthropic` (default): native Anthropic.
    - `openrouter`: routes to OpenRouter with the Anthropic-compatible schema.
    - Custom `base_url` (e.g. a vLLM proxy): uses a fake API key if the env
      var isn't set.
    """
    if provider == "openrouter":
        return anthropic.Anthropic(
            base_url=_OPENROUTER_BASE,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if base_url:
        return anthropic.Anthropic(
            base_url=os.path.expandvars(base_url),
            api_key=os.environ.get(api_key_env, "none") if api_key_env else "none",
        )
    return anthropic.Anthropic()


def llm_call(client, **kwargs):
    """Sync call with backoff on rate-limit / connection errors."""
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            return client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            console.print(
                f"  [yellow]Retry {attempt + 1}/{len(_RETRY_DELAYS)}: "
                f"{type(e).__name__}, waiting {delay}s…[/yellow]"
            )
            time.sleep(delay)
    return client.messages.create(**kwargs)
