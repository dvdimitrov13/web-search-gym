"""Convert `core.trace.Trace` files into Qwen3 SFT JSONL.

Two modes, each paired with a specific loss regime in `sft/train.py`:

  --mode whole       One example per trace. Thinking stripped from every
                     assistant turn. Trainer runs `assistant_only_loss=True`
                     → loss fires on every assistant token.

  --mode per-turn    One example per assistant turn, emitted as Qwen's
                     recommended prompt/completion split. Prior assistant
                     turns sit in `prompt` with thinking stripped; the
                     final turn (with thinking) is `completion`. Trainer
                     runs `completion_only_loss=True` → loss fires only
                     on the final turn. Matches jklj077's guidance in
                     QwenLM/Qwen3#1398.

This module is the *only* place that converts between runtime messages and
training format. It imports everything structural from `core/`:
  - `core.tools.OPENAI_TOOLS` — the tool schemas Qwen3 was trained to see
  - `core.prompts.SEARCHER_PROMPT` / `THINKING_INSTRUCTION` — the system text
  - `core.context` — `exa_api_block`, `live_state_block`, `estimate_tokens`

Any drift between training and inference comes from here. Keep it thin.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.context import (
    estimate_tokens,
    exa_api_block,
    live_state_block,
)
from core.prompts import SEARCHER_PROMPT, THINKING_INSTRUCTION
from core.tools import OPENAI_TOOLS
from core.trace import Trace


# Harness defaults. Override if the trace was produced with different budgets.
DEFAULT_MAX_SEARCHES = 5
DEFAULT_SCRATCHPAD_MAX_TOKENS = 512


# ── Content-block → Qwen3 message mapping ──────────────────────────


def _convert_assistant(blocks: list[dict], include_thinking: bool) -> dict:
    """One Anthropic-style assistant turn → Qwen3 assistant message."""
    thinking = ""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for b in blocks:
        btype = b.get("type")
        if btype == "thinking":
            thinking = b.get("thinking", "")
        elif btype == "text":
            text_parts.append(b.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": b.get("input", {}),
                    },
                }
            )
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if thinking and include_thinking:
        msg["reasoning_content"] = thinking
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tool_results(blocks: list[dict], scratchpad_tokens: int) -> list[dict]:
    """User-role tool_result blocks → Qwen3 tool-role messages.

    Scratchpad tool results carry only a status message in inference (the
    content lives in the `<scratchpad>` system block). Reproduce that here
    so training matches inference exactly (Issue 007-4).
    """
    out: list[dict] = []
    for b in blocks:
        if b.get("type") != "tool_result":
            continue
        content = b.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        content = str(content)
        # Identify scratchpad results: they start with "Scratchpad" in the
        # harness's canonical wording. Rewrite to a compact status message.
        if content.startswith("Scratchpad "):
            # Keep whatever verb the harness wrote ("written" / "edited" / …)
            # but drop any inline content.
            head = content.split("\n", 1)[0].rstrip(".")
            content = (
                f"{head} (~{scratchpad_tokens}/{DEFAULT_SCRATCHPAD_MAX_TOKENS} "
                "tokens used)."
            )
        out.append({"role": "tool", "content": content})
    return out


# ── Per-turn state simulation ──────────────────────────────────────


def _simulate_states(messages: list[dict]) -> list[dict]:
    """Compute (search_count, scratchpad) state going INTO each assistant turn.

    Walks the trace messages, updating state after each tool_result block.
    Returns one state dict per assistant turn, in order.
    """
    states = []
    search_count = 0
    scratchpad = ""
    for msg in messages:
        if msg["role"] == "assistant":
            states.append(
                {"search_count": search_count, "scratchpad": scratchpad}
            )
            continue
        if msg["role"] != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") != "tool_result":
                continue
            text = b.get("content", "")
            if isinstance(text, list):
                text = "\n".join(
                    c.get("text", "") for c in text if c.get("type") == "text"
                )
            text = str(text)
            # Search-result detection: harness formats each hit with "Title:".
            if "Title:" in text and "URL:" in text:
                search_count += 1
            # Scratchpad update: we can't reconstruct the content from a
            # status-only message; fall back to the raw text written above
            # if the harness didn't sanitize yet (old traces / tests).
            # Production traces will only carry status messages here, and
            # we replay the scratchpad from the model's own tool_use args
            # instead — see _scratchpad_from_turn below.
    return states


def _scratchpad_from_trace(messages: list[dict]) -> list[str]:
    """Replay the scratchpad per assistant turn from tool_use inputs.

    Walks the model's own `scratchpad` tool calls (input: old_text, new_text)
    and applies them to a running string, returning the value that was live
    going INTO each assistant turn.
    """
    running = ""
    states_in: list[str] = []
    for msg in messages:
        if msg["role"] == "assistant":
            states_in.append(running)
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for b in content:
                if b.get("type") == "tool_use" and b.get("name") in ("commit_memory", "scratchpad"):
                    inp = b.get("input", {}) or {}
                    old = inp.get("old_text")
                    new = inp.get("new_text", "")
                    if old is None:
                        running = new
                    else:
                        # Best-effort replace (exact first, then whitespace-normalized).
                        if old in running:
                            running = running.replace(old, new, 1)
    return states_in


# ── System prompt + initial user message ──────────────────────────


def _build_system_prompt(max_searches: int = DEFAULT_MAX_SEARCHES) -> str:
    from datetime import date as _date
    system = SEARCHER_PROMPT.format(
        date=_date.today().isoformat(),
        max_searches=max_searches,
    )
    system = f"{system}\n\n{THINKING_INSTRUCTION}"
    system += f"\n\n{exa_api_block()}"
    return system


def _initial_user_message(
    question: str,
    search_count: int,
    scratchpad: str,
    max_searches: int,
    scratchpad_max_tokens: int,
) -> dict:
    live = live_state_block(
        search_count=search_count,
        max_searches=max_searches,
        scratchpad=scratchpad,
        scratchpad_max_tokens=scratchpad_max_tokens,
    )
    return {
        "role": "user",
        "content": f"Research task:\n\n{question}\n\n{live}",
    }


def _live_user_message(
    search_count: int,
    scratchpad: str,
    max_searches: int,
    scratchpad_max_tokens: int,
) -> dict:
    return {
        "role": "user",
        "content": live_state_block(
            search_count=search_count,
            max_searches=max_searches,
            scratchpad=scratchpad,
            scratchpad_max_tokens=scratchpad_max_tokens,
        ),
    }


# ── Trace → training examples ──────────────────────────────────────


def _iter_assistant_tool_pairs(messages: list[dict]):
    """Yield (assistant_msg_raw, following_tool_results_raw_or_None) pairs."""
    i = 0
    n = len(messages)
    while i < n:
        if messages[i]["role"] == "assistant":
            nxt = None
            if i + 1 < n and messages[i + 1]["role"] == "user":
                nxt_content = messages[i + 1].get("content", "")
                if isinstance(nxt_content, list) and any(
                    b.get("type") == "tool_result" for b in nxt_content
                ):
                    nxt = messages[i + 1]
            yield messages[i], nxt
            i += 2 if nxt else 1
        else:
            i += 1


def convert_whole(
    trace: Trace,
    *,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    scratchpad_max_tokens: int = DEFAULT_SCRATCHPAD_MAX_TOKENS,
) -> list[dict]:
    """Whole-trajectory mode. One example per trace, thinking stripped everywhere.

    Every assistant turn contributes loss (trainer uses `assistant_only_loss`).
    """
    scratchpads_in = _scratchpad_from_trace(trace.messages)
    search_counts = [s["search_count"] for s in _simulate_states(trace.messages)]
    initial_sc = search_counts[0] if search_counts else 0
    initial_sp = scratchpads_in[0] if scratchpads_in else ""

    out: list[dict] = [
        {"role": "system", "content": _build_system_prompt(max_searches)},
        _initial_user_message(
            trace.question,
            initial_sc,
            initial_sp,
            max_searches,
            scratchpad_max_tokens,
        ),
    ]

    assistant_idx = 0
    for a_msg, t_msg in _iter_assistant_tool_pairs(trace.messages):
        blocks = a_msg.get("content", [])
        out.append(_convert_assistant(blocks, include_thinking=False))
        assistant_idx += 1
        if t_msg is None:
            continue
        # Next state (going into the following assistant turn).
        next_sc = (
            search_counts[assistant_idx]
            if assistant_idx < len(search_counts)
            else search_counts[-1]
        )
        next_sp = (
            scratchpads_in[assistant_idx]
            if assistant_idx < len(scratchpads_in)
            else scratchpads_in[-1]
        )
        out.extend(_convert_tool_results(t_msg["content"], estimate_tokens(next_sp)))
        # Only inject live state if there IS a following assistant turn.
        if assistant_idx < len(search_counts):
            out.append(
                _live_user_message(
                    next_sc, next_sp, max_searches, scratchpad_max_tokens
                )
            )

    return [{
        "task_idx": trace.task_idx,
        "messages": out,
        "tools": OPENAI_TOOLS,
    }]


def convert_per_turn(
    trace: Trace,
    *,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    scratchpad_max_tokens: int = DEFAULT_SCRATCHPAD_MAX_TOKENS,
) -> list[dict]:
    """Per-turn mode. One prompt/completion example per assistant turn.

    Prior assistant turns are placed in `prompt` with thinking stripped; the
    current turn (with thinking) is the `completion`. Trainer runs
    `completion_only_loss=True` so only the final turn contributes loss — the
    paradigm jklj077 recommends in QwenLM/Qwen3#1398.
    """
    scratchpads_in = _scratchpad_from_trace(trace.messages)
    search_counts = [s["search_count"] for s in _simulate_states(trace.messages)]
    initial_sc = search_counts[0] if search_counts else 0
    initial_sp = scratchpads_in[0] if scratchpads_in else ""

    parsed: list[tuple[str, list[dict]]] = []
    for a_msg, t_msg in _iter_assistant_tool_pairs(trace.messages):
        parsed.append(("assistant", a_msg.get("content", [])))
        if t_msg is not None:
            parsed.append(("tool", t_msg.get("content", [])))

    examples: list[dict] = []
    assistant_turn_idx = 0
    for i, (role, payload) in enumerate(parsed):
        if role != "assistant":
            continue

        prompt: list[dict] = [
            {"role": "system", "content": _build_system_prompt(max_searches)},
            _initial_user_message(
                trace.question,
                initial_sc,
                initial_sp,
                max_searches,
                scratchpad_max_tokens,
            ),
        ]

        # Replay all PRIOR turns with thinking stripped → prompt context only.
        walk_a = 0
        for j in range(i):
            prev_role, prev_payload = parsed[j]
            if prev_role == "assistant":
                prompt.append(_convert_assistant(prev_payload, include_thinking=False))
                walk_a += 1
            else:
                sp_tokens = estimate_tokens(
                    scratchpads_in[walk_a]
                    if walk_a < len(scratchpads_in)
                    else ""
                )
                prompt.extend(_convert_tool_results(prev_payload, sp_tokens))
                if walk_a < len(search_counts):
                    prompt.append(
                        _live_user_message(
                            search_counts[walk_a],
                            scratchpads_in[walk_a] if walk_a < len(scratchpads_in) else "",
                            max_searches,
                            scratchpad_max_tokens,
                        )
                    )

        # Current turn WITH thinking is the sole loss target.
        completion = [_convert_assistant(payload, include_thinking=True)]

        examples.append(
            {
                "task_idx": trace.task_idx,
                "turn_idx": assistant_turn_idx,
                "prompt": deepcopy(prompt),
                "completion": deepcopy(completion),
                "tools": OPENAI_TOOLS,
            }
        )
        assistant_turn_idx += 1

    return examples


# ── File-level driver + CLI ────────────────────────────────────────


def convert_dir(
    input_dir: Path,
    output_path: Path,
    mode: str,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    scratchpad_max_tokens: int = DEFAULT_SCRATCHPAD_MAX_TOKENS,
) -> int:
    """Convert all Trace JSON files under `input_dir` into a single JSONL."""
    records: list[dict] = []
    for path in sorted(input_dir.glob("idx-*.json")):
        trace = Trace.load(path)
        if mode == "whole":
            records.extend(
                convert_whole(
                    trace,
                    max_searches=max_searches,
                    scratchpad_max_tokens=scratchpad_max_tokens,
                )
            )
        elif mode == "per-turn":
            records.extend(
                convert_per_turn(
                    trace,
                    max_searches=max_searches,
                    scratchpad_max_tokens=scratchpad_max_tokens,
                )
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main():
    p = argparse.ArgumentParser(description="Trace → Qwen3 SFT JSONL")
    p.add_argument("--input", type=Path, required=True, help="Trace dir OR single JSONL with concatenated traces")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["whole", "per-turn"], default="per-turn")
    p.add_argument("--max-searches", type=int, default=DEFAULT_MAX_SEARCHES)
    p.add_argument(
        "--scratchpad-max-tokens",
        type=int,
        default=DEFAULT_SCRATCHPAD_MAX_TOKENS,
    )
    args = p.parse_args()

    if args.input.is_dir():
        n = convert_dir(
            args.input,
            args.output,
            args.mode,
            max_searches=args.max_searches,
            scratchpad_max_tokens=args.scratchpad_max_tokens,
        )
    else:
        raise SystemExit(
            "Single-JSONL input mode not implemented; pass a Trace directory."
        )
    print(f"Converted {n} examples ({args.mode}) → {args.output}")


if __name__ == "__main__":
    main()
