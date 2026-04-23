"""Browse tool: Jina Reader fetch + Haiku extractive cleanup.

Given a URL and a research question, fetches the page via Jina Reader and
runs Haiku with a detail-preservation prompt to produce an information-dense
~256-token extract focused on the question.

Mirrors DR Tulu's `ChainedTool([JinaBrowseTool, WebPageReaderAgentV2])`:
raw fetch + LLM cleanup. Our cleaner is Haiku instead of Qwen3-8B — no
extra GPU, cheap per-call, steerable via prompt.

Jina Reader endpoint: `https://r.jina.ai/{url}`. Works unauthenticated at
a lower rate-limit tier; set `JINA_API_KEY` in .env for higher throughput.
"""

from __future__ import annotations

import os

import anthropic
import requests

_HAIKU_EXTRACT_PROMPT = """You are a research assistant extracting evidence from a webpage.

Research question: {question}

Webpage title: {title}
URL: {url}

<page>
{content}
</page>

Your job: extract every specific fact on this page relevant to the research \
question — named entities, exact numbers, dates, quantities, list items, \
step details. Enumerate VERBATIM. Do NOT paraphrase or abstract. If the page \
contains a list, reproduce it. If steps, reproduce each step. If a table of \
values, reproduce the values.

Constraints:
- If no relevant facts are present, output exactly: "No relevant facts found."
- Prefer dense bullet points and structured enumeration over prose.
- Information-dense language — every token should carry factual weight.
- Target ~256 tokens; do not exceed that materially.

Extracted facts:"""


def fetch_webpage_jina(
    url: str,
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """Fetch readable webpage text via Jina Reader. Returns {url, title, content}."""
    headers = {"Accept": "application/json"}
    key = api_key or os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Jina's JSON shape: {"code": 200, "data": {"title": ..., "content": ..., "url": ...}}
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    return {
        "url": inner.get("url") or url,
        "title": inner.get("title") or "(untitled)",
        "content": inner.get("content") or "",
    }


class HaikuBrowseExtractor:
    """Haiku-backed page-content cleaner. One call per URL.

    Exposes both `extract` (sync) and `extract_async` (async). The async
    form lets the agent_dd harness fan out multiple browse_page calls in one
    turn via `asyncio.gather` without spinning up a thread per call.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 320,  # ~256 tokens + small overhead margin
        max_page_chars: int = 128000,  # ~32k input tokens (4 chars/tok heuristic)
    ):
        self.client = anthropic.Anthropic()
        # Lazy — only created when `extract_async` is first called, so sync-
        # only callers don't pay for two client instances.
        self._async_client: anthropic.AsyncAnthropic | None = None
        self.model = model
        self.max_tokens = max_tokens
        self.max_page_chars = max_page_chars

    @property
    def async_client(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic()
        return self._async_client

    def _build_prompt(self, *, url: str, title: str, question: str, content: str) -> str:
        if len(content) > self.max_page_chars:
            content = content[: self.max_page_chars] + "\n[...truncated]"
        return _HAIKU_EXTRACT_PROMPT.format(
            question=question, title=title, url=url, content=content,
        )

    @staticmethod
    def _extract_text(resp) -> str:
        parts = [
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        ]
        return "".join(parts).strip()

    def extract(self, *, url: str, title: str, question: str, content: str) -> str:
        prompt = self._build_prompt(url=url, title=title, question=question, content=content)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(resp)

    async def extract_async(
        self, *, url: str, title: str, question: str, content: str,
    ) -> str:
        prompt = self._build_prompt(url=url, title=title, question=question, content=content)
        resp = await self.async_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(resp)


def browse_and_extract(
    url: str,
    question: str,
    extractor: HaikuBrowseExtractor,
    jina_api_key: str | None = None,
) -> dict:
    """Full browse pipeline: Jina Reader fetch → Haiku extractive cleanup.

    Returns {url, title, text} where `text` is the Haiku extract.
    """
    fetched = fetch_webpage_jina(url, api_key=jina_api_key)
    extracted = extractor.extract(
        url=fetched["url"],
        title=fetched["title"],
        question=question,
        content=fetched["content"],
    )
    return {
        "url": fetched["url"],
        "title": fetched["title"],
        "text": extracted,
    }
