"""DR Tulu-style single-agent harness.

Unlike lean_searcher (searcher → extractor two-stage), this agent drives
retrieval AND synthesis in ONE rollout. Tools: search (highlights) /
browse_page (Jina + Haiku extract) / answer (final, cited). No downstream
extractor — the `answer` tool's `final_answer` field is the graded output.
"""

from __future__ import annotations

import time
from pathlib import Path

from agents.base import BaseAgent
from core.console import console
from core.agent_dd_harness import AgentDDHarness
from core.types import Answer, Task

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORIES_DIR = _REPO_ROOT / "trajectories"


class AgentDD(BaseAgent):
    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        super().__init__(config_path, model_configs)

        if "searcher" not in model_configs:
            raise ValueError(
                "agent_dd requires a searcher model via --model or "
                "--searcher-model."
            )

        s_model = model_configs["searcher"]
        budget = self.config.get("budget", {})
        searcher_cfg = self.config.get("searcher", {})
        browse_cfg = self.config.get("browse", {})

        self.harness = AgentDDHarness(
            searcher_model=s_model["model_name"],
            provider=s_model.get("provider", "anthropic"),
            base_url=s_model.get("base_url", ""),
            api_key_env=s_model.get("api_key_env", ""),
            temperature=s_model.get("temperature", searcher_cfg.get("temperature", 0.2)),
            thinking_budget=s_model.get(
                "thinking_budget", searcher_cfg.get("thinking_budget")
            ),
            thinking_passthrough=searcher_cfg.get("thinking_passthrough", False),
            max_cycles=budget.get("max_cycles", 5),
            results_per_query=budget.get("results_per_query", 5),
            highlight_max_chars=budget.get("highlight_max_chars", 200),
            highlights_per_url=budget.get("highlights_per_url", 3),
            exa_search_type=searcher_cfg.get("exa_search_type", "instant"),
            browse_extractor_model=browse_cfg.get(
                "extractor_model", "claude-haiku-4-5-20251001"
            ),
            browse_extractor_max_tokens=browse_cfg.get("extractor_max_tokens", 320),
            scratchpad_max_tokens=searcher_cfg.get("scratchpad_max_tokens", 1024),
            max_shrink_attempts=searcher_cfg.get("max_shrink_attempts", 2),
            max_nudges=searcher_cfg.get("max_nudges", 3),
            verbose=self.config.get("verbose", False),
        )

        self.trajectories_dir = TRAJECTORIES_DIR / self.name

    def answer(self, task: Task) -> Answer:
        t0 = time.time()
        console.rule(f"[bold]Task idx-{task.idx}[/bold]")

        trace, final_answer = self.harness.run(task)
        trace.agent = self.name
        trace.save(self.trajectories_dir / f"idx-{task.idx}.json")

        elapsed = time.time() - t0
        console.print(
            f"  [green]Done[/green] — {len(trace.source_bank)} snippets, "
            f"exact='{final_answer.exact_answer[:60]}', "
            f"{elapsed:.1f}s"
        )
        return final_answer
