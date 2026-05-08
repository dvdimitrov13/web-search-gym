"""Scratchpad text-edit primitive — fuzzy substring replace.

Used by lean_searcher (commit_memory) and agent_dd (commit_memory).
Three-tier match: exact → whitespace-normalized → rapidfuzz 95% partial.
"""

from __future__ import annotations

import re


def fuzzy_replace(text: str, old: str, new: str) -> tuple[str, bool]:
    """Replace `old` in `text` with relaxed whitespace matching.

    Three tiers, in order:
    1. Exact substring match.
    2. Whitespace-normalized match, mapped back to original text.
    3. rapidfuzz partial-ratio alignment with 95% similarity floor.

    Returns (new_text, matched).
    """
    if old in text:
        return text.replace(old, new, 1), True

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    norm_old = normalize(old)
    norm_text = normalize(text)
    idx = norm_text.find(norm_old)

    if idx != -1:
        # Map normalized index back to original positions.
        char_map: list[int] = []
        in_ws = False
        norm_pos = 0
        for oi, c in enumerate(text):
            if c in " \t\n\r":
                if not in_ws and norm_pos > 0:
                    char_map.append(oi)
                    norm_pos += 1
                in_ws = True
            else:
                in_ws = False
                char_map.append(oi)
                norm_pos += 1

        if idx < len(char_map):
            orig_start = char_map[idx]
            end_idx = idx + len(norm_old)
            if end_idx <= len(char_map):
                orig_end = char_map[end_idx - 1] + 1
            elif char_map:
                orig_end = char_map[-1] + 1
            else:
                orig_end = len(text)
            return text[:orig_start] + new + text[orig_end:], True

    # Tier 3: rapidfuzz
    try:
        from rapidfuzz import fuzz
        alignment = fuzz.partial_ratio_alignment(old, text, score_cutoff=95)
        if (
            alignment is not None
            and alignment.score >= 95
            and alignment.dest_end > alignment.dest_start
        ):
            return (
                text[: alignment.dest_start] + new + text[alignment.dest_end :],
                True,
            )
    except ImportError:
        pass

    return text, False
