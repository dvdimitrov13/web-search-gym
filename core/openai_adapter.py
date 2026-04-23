"""Adapter exposing an `anthropic.Anthropic`-shaped client backed by OpenAI.

Why this exists: our harnesses call `client.messages.create(...)` with
Anthropic's Messages API shape (system + messages-with-tool-use-blocks +
tools schema). vLLM and most self-hosted inference stacks only speak the
OpenAI Chat Completions API. Rather than fork the harness for every backend,
we wrap an OpenAI client so it looks like an Anthropic one.

Supported providers (via `is_openai_compat`):
  - "vllm"  — self-hosted vLLM (gemma, qwen, etc.)
  - "openai" / "openai_compatible" — any OAI-shaped endpoint

Limitations (acceptable for baselines; revisit if we need more):
  - `cache_control` is silently dropped (OAI has no prompt caching yet).
  - `thinking` blocks are not translated (we don't enable them for these models).
  - `usage` only carries input/output tokens; cache_* fields are always 0.
  - Tool-call `arguments` must parse as JSON; malformed args get a stub dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import openai


_OPENAI_COMPAT_PROVIDERS = {"vllm", "openai", "openai_compatible"}


def is_openai_compat(provider: str) -> bool:
    return provider in _OPENAI_COMPAT_PROVIDERS


# ── Anthropic-shaped response shims ────────────────────────────────

@dataclass
class _UsageLike:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _BlockLike:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        if self.type == "text":
            return {"type": "text", "text": self.text}
        if self.type == "tool_use":
            return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}
        return {"type": self.type}


@dataclass
class _ResponseLike:
    content: list[_BlockLike]
    usage: _UsageLike
    stop_reason: str = ""


# ── Anthropic → OpenAI translation ─────────────────────────────────

def _system_text(system) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    # List of content blocks (text blocks with optional cache_control)
    parts = []
    for b in system:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n\n".join(parts)


def _read_block_field(b, key: str, default=None):
    if isinstance(b, dict):
        return b.get(key, default)
    return getattr(b, key, default)


def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Fan Anthropic messages (with content blocks) into OpenAI chat messages.

    Anthropic packs everything for one turn into one message. OpenAI splits
    tool_result content out into separate `role=tool` messages ordered
    alongside the assistant turn they answer. We preserve that ordering.
    """
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        content = m["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for b in content:
                btype = _read_block_field(b, "type")
                if btype == "text":
                    text_parts.append(_read_block_field(b, "text", "") or "")
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": _read_block_field(b, "id", "") or "",
                        "type": "function",
                        "function": {
                            "name": _read_block_field(b, "name", "") or "",
                            "arguments": json.dumps(_read_block_field(b, "input", {}) or {}),
                        },
                    })
                # thinking blocks ignored — callers don't set thinking_budget for these models
            msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)

        elif role == "user":
            text_parts: list[str] = []
            tool_msgs: list[dict] = []
            for b in content:
                btype = _read_block_field(b, "type")
                if btype == "text":
                    text_parts.append(_read_block_field(b, "text", "") or "")
                elif btype == "tool_result":
                    rc = _read_block_field(b, "content", "")
                    if isinstance(rc, list):
                        rc_text = "".join(_read_block_field(bb, "text", "") or "" for bb in rc)
                    else:
                        rc_text = str(rc) if rc is not None else ""
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": _read_block_field(b, "tool_use_id", "") or "",
                        "content": rc_text,
                    })
            # OpenAI convention: tool results come FIRST (they respond to the
            # immediately preceding assistant turn's tool_calls), then any
            # additional user text becomes a separate user message.
            out.extend(tool_msgs)
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})

        else:
            # system or other — pass string-ified
            out.append({"role": role, "content": content if isinstance(content, str) else str(content)})

    return out


def _tools_to_openai(tools) -> list[dict]:
    if not tools:
        return []
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _build_kwargs(**anthropic_kwargs) -> dict:
    messages = anthropic_kwargs.get("messages", []) or []
    system = anthropic_kwargs.get("system")
    tools = anthropic_kwargs.get("tools")

    openai_messages = _messages_to_openai(messages)
    sys_text = _system_text(system)
    if sys_text:
        openai_messages = [{"role": "system", "content": sys_text}] + openai_messages

    kwargs = {
        "model": anthropic_kwargs["model"],
        "messages": openai_messages,
        "max_tokens": anthropic_kwargs.get("max_tokens", 4096),
        "temperature": anthropic_kwargs.get("temperature", 0.2),
    }
    if tools:
        kwargs["tools"] = _tools_to_openai(tools)
        kwargs["tool_choice"] = "auto"
    # Pass through OpenAI-only extensions (e.g. vLLM chat_template_kwargs for
    # per-request Gemma4 enable_thinking toggle). Callers opt in explicitly.
    extra_body = anthropic_kwargs.get("extra_body")
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


# ── OpenAI → Anthropic translation ─────────────────────────────────

_FINISH_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _response_to_anthropic(resp) -> _ResponseLike:
    choice = resp.choices[0]
    msg = choice.message

    blocks: list[_BlockLike] = []

    text = getattr(msg, "content", None) or ""
    if text:
        blocks.append(_BlockLike(type="text", text=text))

    for tc in getattr(msg, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        args_str = getattr(fn, "arguments", "") or "" if fn is not None else ""
        try:
            args = json.loads(args_str) if args_str else {}
        except Exception:
            args = {"__unparsed_arguments": args_str}
        blocks.append(_BlockLike(
            type="tool_use",
            id=getattr(tc, "id", "") or "",
            name=(getattr(fn, "name", "") if fn is not None else "") or "",
            input=args,
        ))

    u = getattr(resp, "usage", None)
    usage = _UsageLike(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
    )

    finish = getattr(choice, "finish_reason", "") or ""
    return _ResponseLike(
        content=blocks, usage=usage, stop_reason=_FINISH_MAP.get(finish, finish),
    )


# ── Public client shims ────────────────────────────────────────────

class _SyncMessages:
    def __init__(self, client: openai.OpenAI):
        self._c = client

    def create(self, **kwargs) -> _ResponseLike:
        resp = self._c.chat.completions.create(**_build_kwargs(**kwargs))
        return _response_to_anthropic(resp)


class _AsyncMessages:
    def __init__(self, client: openai.AsyncOpenAI):
        self._c = client

    async def create(self, **kwargs) -> _ResponseLike:
        resp = await self._c.chat.completions.create(**_build_kwargs(**kwargs))
        return _response_to_anthropic(resp)


class AnthropicShim:
    """Sync. Instantiate as drop-in for `anthropic.Anthropic(...)` when the
    backend speaks OpenAI Chat Completions."""

    def __init__(self, base_url: str, api_key: str = "none"):
        self._c = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.messages = _SyncMessages(self._c)


class AsyncAnthropicShim:
    """Async counterpart. Drop-in for `anthropic.AsyncAnthropic(...)`."""

    def __init__(self, base_url: str, api_key: str = "none"):
        self._c = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.messages = _AsyncMessages(self._c)
