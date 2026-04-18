"""BrowseComp loader and LLM-judge grader.

Dataset: https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
Each row has base64+XOR-encrypted `problem` and `answer`, decrypted with the
per-row `canary` password (derive_key → SHA256).

Judge: OpenAI-compatible chat model (default gpt-4o-mini). The judge prompt is
ported verbatim from openai/simple-evals so our numbers are comparable.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.types import Answer, Task

DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)

# Local cache path — avoids re-downloading on every CLI invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = _REPO_ROOT / "bench" / ".cache" / "browse_comp_test_set.csv"


# ── Decryption ──────────────────────────────────────────────────────


def derive_key(password: str, length: int) -> bytes:
    """SHA256-derived fixed-length key (port of simple-evals)."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """base64 decode → XOR with derived key → UTF-8 decode."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


# ── Dataset loading ─────────────────────────────────────────────────


def _load_dataframe() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    df = pd.read_csv(DATASET_URL)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def load_tasks(indices: list[int] | None = None) -> list[Task]:
    """Load and decrypt BrowseComp tasks.

    If `indices` is provided, return only those row indices (0-based) from the CSV.
    Otherwise returns all 1,266 tasks.
    """
    df = _load_dataframe()
    rows = df.to_dict(orient="records")
    if indices is not None:
        rows = [rows[i] for i in indices if 0 <= i < len(rows)]
        idx_seq = [i for i in indices if 0 <= i < len(df)]
    else:
        idx_seq = list(range(len(rows)))

    tasks = []
    for idx, row in zip(idx_seq, rows):
        canary = row.get("canary", "")
        try:
            question = decrypt(row["problem"], canary)
            answer = decrypt(row["answer"], canary)
        except Exception as e:
            raise RuntimeError(f"BrowseComp row {idx}: decrypt failed: {e}") from e
        meta = {k: row[k] for k in row if k not in {"problem", "answer", "canary"}}
        tasks.append(Task(idx=idx, question=question, answer=answer, metadata=meta))
    return tasks


# ── Grader ──────────────────────────────────────────────────────────


_GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on \
the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to extract \
from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based \
on [correct_answer], focusing only on if there are meaningful differences \
between [correct_answer] and the extracted_final_answer. Do not comment on any \
background to the problem, do not attempt to solve the problem, do not argue \
for any answer different than [correct_answer], focus only on whether the \
answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, \
non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. \
Put 100 if there is no confidence score available."""


_CORRECT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)


@dataclass
class GradeResult:
    is_correct: bool
    raw_judge_output: str


def answer_to_browsecomp_text(answer: Answer) -> str:
    """Render an Answer into the canonical BrowseComp response format."""
    return (
        f"Explanation: {answer.explanation}\n"
        f"Exact Answer: {answer.exact_answer}\n"
        f"Confidence: {answer.confidence}"
    )


class Grader:
    """OpenAI-compatible LLM judge for BrowseComp."""

    def __init__(
        self,
        model: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI  # type: ignore

        self.model = model or os.environ.get("BROWSECOMP_JUDGE_MODEL", "gpt-4o-mini")
        api_key_env = api_key_env or os.environ.get(
            "BROWSECOMP_JUDGE_API_KEY_ENV", "OPENAI_API_KEY"
        )
        base_url = base_url or os.environ.get("BROWSECOMP_JUDGE_BASE_URL")
        api_key = os.environ.get(api_key_env)
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def grade(self, question: str, correct_answer: str, response: str) -> GradeResult:
        prompt = _GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        m = _CORRECT_RE.search(text)
        verdict = m.group(1).lower() if m else "no"
        return GradeResult(is_correct=(verdict == "yes"), raw_judge_output=text)
