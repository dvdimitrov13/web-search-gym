"""DeepSearchQA loader + official Gemini-2.5-flash grader.

Dataset: `google/deepsearchqa` on Hugging Face. 900 hand-crafted multi-step
info-seeking tasks across 17 domains. Answers are either a single entity
('Single Answer') or an unordered set ('Set Answer', ~65% of the dataset).

Grading follows the official Kaggle starter notebook verbatim (prompt + JSON
parsing + precision/recall/F1 aggregation). The autorater is pinned to
`gemini-2.5-flash`; the paper warns that any deviation from prompt or model
meaningfully shifts numbers.

Splits (inlined here, not in bench/splits.yaml — DSQA indices are its own
namespace into the HF `eval` split):

    dev     — 10 tasks, smoke tests
    smoke   — 50 tasks, rough comparisons
    domain2 — 34 tasks, 2 per problem_category (17 domains), seeded
    full    — all 900 tasks

Reference: https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf
Starter:   https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code
"""

from __future__ import annotations

import json
import logging
import random
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from core.gemini_client import GeminiClient
from core.types import Task

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _REPO_ROOT / "bench" / ".cache" / "deepsearchqa"

# Pinned autorater — changing this breaks comparability with the published
# DeepSearchQA numbers.
AUTORATER_MODEL = "gemini-2.5-flash"


# ── Dataset loading ─────────────────────────────────────────────────


DSQA_CSV_URL = "https://huggingface.co/datasets/google/deepsearchqa/resolve/main/DSQA-full.csv"


def _load_rows() -> list[dict]:
    """Fetch the DSQA eval split from the HF CSV, cached locally."""
    cache_path = CACHE_DIR / "DSQA-full.csv"
    if not cache_path.exists():
        import pandas as pd

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(DSQA_CSV_URL)
        df.to_csv(cache_path, index=False)
    else:
        import pandas as pd

        df = pd.read_csv(cache_path)
    # Normalize NaN → empty string for downstream `.strip()` safety.
    df = df.fillna("")
    return df.to_dict(orient="records")


def _domain2_indices(rows: list[dict], seed: int = 42) -> list[int]:
    """2 questions per problem_category, sorted by index for reproducibility."""
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_cat[r.get("problem_category", "Unknown")].append(i)
    picked: list[int] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        picked.extend(rng.sample(pool, k=min(2, len(pool))))
    return sorted(picked)


def _split_indices(split: str, rows: list[dict]) -> list[int] | None:
    if split == "full":
        return None
    if split == "dev":
        # First 10 indices across early rows for fast iteration.
        return list(range(10))
    if split == "smoke":
        # Evenly-spaced 50-index stride across the 900.
        return list(range(0, len(rows), max(1, len(rows) // 50)))[:50]
    if split == "domain2":
        return _domain2_indices(rows)
    raise ValueError(f"Unknown DeepSearchQA split: {split}")


def load_tasks(
    split: str = "domain2",
    indices: list[int] | None = None,
) -> list[Task]:
    rows = _load_rows()
    idx_list = indices if indices is not None else _split_indices(split, rows)
    if idx_list is None:
        idx_list = list(range(len(rows)))

    tasks: list[Task] = []
    for i in idx_list:
        if i < 0 or i >= len(rows):
            continue
        r = rows[i]
        meta = {
            "problem_category": r.get("problem_category", ""),
            "answer_type": r.get("answer_type", ""),
        }
        tasks.append(
            Task(
                idx=i,
                question=str(r.get("problem", "")).strip(),
                answer=str(r.get("answer", "") or "").strip(),
                metadata=meta,
            )
        )
    return tasks


# ── Grader: verbatim prompt + parsing from the Kaggle starter notebook ──


_DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

_GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


def _parse_json_response(raw: str) -> dict | None:
    """Lift the JSON payload out of ```json ...``` fences if present."""
    try:
        s = (raw or "").strip()
        start = s.find("```json")
        if start != -1:
            s = s[start + len("```json"):].strip()
            end = s.rfind("```")
            if end != -1:
                s = s[:end].strip()
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _correctness_details(payload: dict) -> dict[str, bool] | None:
    try:
        details = payload["Answer Correctness"]["Correctness Details"]
    except (KeyError, TypeError):
        return None
    if not isinstance(details, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, bool) for k, v in details.items()):
        return None
    return details


def _excessive_answers(payload: dict) -> list[str] | None:
    """None = malformed; [] = none provided (valid)."""
    try:
        items = payload["Answer Correctness"]["Excessive Answers"]
    except (KeyError, TypeError):
        return []
    if not isinstance(items, list):
        return None
    if not all(isinstance(x, str) for x in items):
        return None
    return items


@dataclass
class DSQAGradeResult:
    is_correct: bool
    raw_judge_output: str
    metrics: dict = field(default_factory=dict)


class DSQAGrader:
    """Official DeepSearchQA autorater. Model pinned to gemini-2.5-flash."""

    model = AUTORATER_MODEL
    # The runner reads this: DSQA grades a prose response, not a single-line
    # extracted answer. Agents that produce a natural_text get it routed here
    # verbatim instead of the BrowseComp-formatted wrapper.
    prefers_natural_response = True

    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient(default_model=AUTORATER_MODEL)

    def _build_prompt(self, question: str, answer: str, response: str, prompt_type: str) -> str:
        return _DEEPSEARCH_QA_PROMPT + _GRADER_RATING_OUTPUT_EXAMPLE.format(
            prompt=question.strip(),
            prompt_type=prompt_type.strip(),
            answer=answer.strip(),
            response=response.strip(),
        )

    def grade(self, task: Task, response_text: str) -> DSQAGradeResult:
        prompt_type = task.metadata.get("answer_type", "Single Answer")
        gold = task.answer or ""
        rating_prompt = self._build_prompt(task.question, gold, response_text, prompt_type)

        try:
            raw = self.client.generate(rating_prompt, model=self.model)
        except Exception as e:
            logging.exception("DSQA grader LLM call failed for task %s", task.idx)
            return DSQAGradeResult(
                is_correct=False,
                raw_judge_output="",
                metrics={"error": f"grader_llm_failed: {e}", "status": "empty_auto_rater"},
            )

        payload = _parse_json_response(raw)
        if not payload:
            return DSQAGradeResult(
                is_correct=False,
                raw_judge_output=raw,
                metrics={"status": "invalid_auto_rater"},
            )

        details = _correctness_details(payload)
        if details is None:
            return DSQAGradeResult(
                is_correct=False,
                raw_judge_output=raw,
                metrics={"status": "invalid_auto_rater"},
            )
        excessive = _excessive_answers(payload)
        if excessive is None:
            return DSQAGradeResult(
                is_correct=False,
                raw_judge_output=raw,
                metrics={"status": "invalid_auto_rater"},
            )

        tp = sum(1 for v in details.values() if v)
        fn = sum(1 for v in details.values() if not v)
        fp = len(excessive)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        has_expected = bool(details)
        all_expected_correct = has_expected and (tp == len(details))
        has_excessive = bool(excessive)
        # Mirrors the Kaggle `is_all_correct` definition: all expected answers
        # correct (or no expected answers) AND no excessive answers.
        is_correct = (all_expected_correct or not has_expected) and not has_excessive

        return DSQAGradeResult(
            is_correct=is_correct,
            raw_judge_output=raw,
            metrics={
                "status": "ok",
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "correctness_details": details,
                "excessive_answers": excessive,
                "explanation": (payload.get("Answer Correctness") or {}).get("Explanation", ""),
                "category": task.metadata.get("problem_category", ""),
                "answer_type": task.metadata.get("answer_type", ""),
            },
        )

    # ── Aggregation ────────────────────────────────────────────────

    def aggregate(self, records: list[dict]) -> dict:
        """Dataset-level metrics. `records` is a list of TaskRecord.to_dict()."""
        n = len(records)
        precisions, recalls, f1s = [], [], []
        all_correct = 0
        invalid = 0
        per_category_all_correct: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in records:
            m = r.get("metrics") or {}
            if m.get("status") != "ok":
                invalid += 1
                continue
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            cat = m.get("category") or "Unknown"
            per_category_all_correct[cat][1] += 1
            if r.get("is_correct"):
                all_correct += 1
                per_category_all_correct[cat][0] += 1

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "num_valid": len(precisions),
            "num_invalid_auto_rater": invalid,
            "precision": _mean(precisions),
            "recall": _mean(recalls),
            "f1": _mean(f1s),
            "pct_all_correct": all_correct / n if n else 0.0,
            "per_category_all_correct": {
                cat: {"all_correct": c[0], "evaluated": c[1]}
                for cat, c in sorted(per_category_all_correct.items())
            },
        }
