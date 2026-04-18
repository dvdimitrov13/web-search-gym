"""Retroactively inject synthetic thinking on assistant turns that have none.

Problem: Claude Sonnet with concise-thinking + modest budget emits no thinking
on tool-use continuations (turns 1+). The model learns to think only on turn 0.
At inference the resulting agent routinely skips thinking mid-task, which
hurts decision quality.

Fix: Use Claude to generate 1–2 sentences of targeted reasoning per turn,
grounded in the prior tool result + the upcoming action. Inject into the
trace's assistant turn as a `thinking` content block.

This is an offline, one-shot pass on a directory of Trace JSON files. Output
is either in-place (new suffix) or to a sibling dir.

Not wired into the default SFT pipeline — enable explicitly when preparing
data for a training run.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import anthropic

from core.trace import Trace


AUGMENT_PROMPT = """\
You are augmenting a search agent's trajectory by inserting a brief inner \
monologue ("thinking") before its next action.

Given the preceding tool result and the action the agent is about to take, \
produce 1-2 sentences that explain WHY this action makes sense given what was \
just seen. Be concrete — reference specific observations.

Output only the thinking text. No preamble, no formatting, no quotes."""


def _prior_tool_result_text(messages: list[dict], i: int) -> str:
    """Return the last tool_result text before assistant turn at index `i`."""
    for j in range(i - 1, -1, -1):
        msg = messages[j]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for b in reversed(content):
            if b.get("type") == "tool_result":
                text = b.get("content", "")
                if isinstance(text, list):
                    text = "\n".join(
                        c.get("text", "") for c in text if c.get("type") == "text"
                    )
                return str(text)
    return ""


def _next_action_text(blocks: list[dict]) -> str:
    for b in blocks:
        if b.get("type") == "tool_use":
            return f"Calling {b.get('name', '')} with input {json.dumps(b.get('input', {}))}"
        if b.get("type") == "text":
            return f"Saying: {b.get('text', '')[:200]}"
    return "(no action)"


def _has_thinking(blocks: list[dict]) -> bool:
    return any(
        b.get("type") == "thinking" and b.get("thinking", "").strip()
        for b in blocks
    )


def augment_trace(trace: Trace, client: anthropic.Anthropic, model: str) -> Trace:
    new_messages = deepcopy(trace.messages)
    for i, msg in enumerate(new_messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list) or _has_thinking(content):
            continue

        prior = _prior_tool_result_text(new_messages, i)[:4000]
        action = _next_action_text(content)
        prompt = (
            f"Prior tool result:\n{prior}\n\n"
            f"Upcoming action: {action}\n\n"
            "Produce the thinking text."
        )
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.7,
            system=AUGMENT_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if text:
            content.insert(0, {"type": "thinking", "thinking": text})

    trace.messages = new_messages
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(args.input_dir.glob("idx-*.json")):
        trace = Trace.load(path)
        augmented = augment_trace(trace, client, args.model)
        augmented.save(args.output_dir / path.name)
        n += 1
        print(f"augmented {path.name}")
    print(f"\n{n} traces augmented → {args.output_dir}")


if __name__ == "__main__":
    main()
