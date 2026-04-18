"""Trace → Qwen3 conversion tests on a hand-built minimal trace."""

from __future__ import annotations

from core.trace import SubmittedUrl, Trace, TraceMetadata


def _minimal_trace() -> Trace:
    """Build the smallest plausible two-turn trace by hand."""
    return Trace(
        task_idx=42,
        question="What year did the FIA introduce active aero to F1?",
        messages=[
            {"role": "user", "content": "Research task:\n\nWhat year…"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to find FIA active aero intro year."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "exa_search",
                        "input": {"query": "FIA active aero F1 introduction year"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": (
                            "Title: FIA regulations\nURL: https://fia.com/rule\n"
                            "Published: 2024\nSummary: Active aero comes in 2026."
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Enough — submit."},
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "submit",
                        "input": {"urls": [{"url": "https://fia.com/rule", "score": 0.9}]},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": "URLs submitted. Research complete.",
                    }
                ],
            },
        ],
        source_bank={
            "https://fia.com/rule": {
                "url": "https://fia.com/rule",
                "title": "FIA regulations",
                "summary": "Active aero comes in 2026.",
                "published": "2024",
            }
        },
        submitted=[SubmittedUrl(url="https://fia.com/rule", score=0.9)],
        metadata=TraceMetadata(search_count=1, nudge_count=0, stop_reason="submit"),
    )


def test_convert_whole_produces_one_example():
    from sft.convert import convert_whole

    out = convert_whole(_minimal_trace())
    assert len(out) == 1
    ex = out[0]
    assert ex["task_idx"] == 42
    roles = [m["role"] for m in ex["messages"]]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assert ex["tools"] and len(ex["tools"]) == 3


def test_convert_whole_strips_thinking_everywhere():
    """Whole mode trains on all assistant turns; thinking must be dropped
    so the model isn't taught to emit thinking-free completions."""
    from sft.convert import convert_whole

    ex = convert_whole(_minimal_trace())[0]
    for m in ex["messages"]:
        if m["role"] == "assistant":
            assert "reasoning_content" not in m


def test_convert_per_turn_emits_prompt_completion():
    """Per-turn mode uses prompt/completion so only the final turn is the loss
    target (TRL's completion_only_loss path)."""
    from sft.convert import convert_per_turn

    out = convert_per_turn(_minimal_trace())
    assert len(out) == 2
    assert [e["turn_idx"] for e in out] == [0, 1]

    for ex in out:
        assert "prompt" in ex and "completion" in ex
        assert "messages" not in ex

        # Completion is exactly one assistant turn WITH thinking.
        assert len(ex["completion"]) == 1
        final = ex["completion"][0]
        assert final["role"] == "assistant"
        assert "reasoning_content" in final

        # Prior assistant turns (if any) live in the prompt with thinking stripped.
        prior_assistants = [m for m in ex["prompt"] if m["role"] == "assistant"]
        for m in prior_assistants:
            assert "reasoning_content" not in m


def test_convert_preserves_task_question_in_initial_user():
    from sft.convert import convert_whole

    out = convert_whole(_minimal_trace())
    initial_user = out[0]["messages"][1]
    assert initial_user["role"] == "user"
    assert "FIA" in initial_user["content"]
    assert "<budget>" in initial_user["content"]
