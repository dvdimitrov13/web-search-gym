"""Shared prompts.

Per-agent system prompts (lean_searcher's SEARCHER_PROMPT, agent_dd's
AGENT_DD_SYSTEM_PROMPT, chroma_agent's CHROMA_SEARCHER_PROMPT) live in
`agents/<name>/prompts.py`. The three below are genuinely shared:

- THINKING_INSTRUCTION: appended onto any agent's system prompt when
  extended thinking is enabled.
- BROWSE_EXTRACT_PROMPT: page-to-bullets extraction used by
  core.browse.BrowseExtractor.
- EXTRACTOR_PROMPT: single-shot short-answer extraction for
  BrowseComp's required Explanation/Exact Answer/Confidence format.
"""

BROWSE_EXTRACT_PROMPT = """You are a research assistant extracting evidence from a webpage.

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

THINKING_INSTRUCTION = """\
Think concisely. For each decision, state what you need and why in 1-2 sentences, \
then act. Do not deliberate at length or enumerate options you won't pursue."""


# Used by core/extractor.py to emit BrowseComp's required answer format.
# The BrowseComp grader extracts `Exact Answer:` via regex, so the format is
# load-bearing — keep it exact.
EXTRACTOR_PROMPT = """\
You will receive a research question and a set of sources the searcher found.
Answer the question using only information present in the sources.

Your output MUST follow this exact format (each field on its own block):

Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer -- a few words, NOT a sentence}
Confidence: {your confidence score as a whole-number percent 0-100}

Rules:
- The Exact Answer must be short: a name, number, date, or short phrase. Never a full sentence.
- If the sources don't contain enough information, your best single guess still \
belongs on the Exact Answer line. Set Confidence low.
- Do not add any text before "Explanation:" or after the Confidence line."""
