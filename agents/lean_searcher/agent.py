"""Lean searcher — the one harness that ships.

Two-stage pipeline:
1. `SearcherHarness` runs the Exa tool loop until the model calls `submit`.
2. `Extractor` turns the source bank into BrowseComp's Explanation / Exact
   Answer / Confidence format.

Trace is written to `trajectories/lean_searcher/idx-{N}.json` so downstream
tools (SFT conversion, RL reward, viz replay) can consume it without re-running.
"""

from __future__ import annotations

import time
from pathlib import Path

from agents.base import BaseAgent
from core.console import console
from core.extractor import Extractor
from core.harness import SearcherHarness
from core.types import Answer, Task

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORIES_DIR = _REPO_ROOT / "trajectories"


class LeanSearcherAgent(BaseAgent):
    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        super().__init__(config_path, model_configs)

        if "searcher" not in model_configs:
            raise ValueError(
                "lean_searcher requires a searcher model. "
                "Pass --model <name> or --searcher-model <name>."
            )
        if "extractor" not in model_configs:
            raise ValueError(
                "lean_searcher requires an extractor model. "
                "Pass --model <name> or --extractor-model <name>."
            )

        s_model = model_configs["searcher"]
        e_model = model_configs["extractor"]

        budget = self.config.get("budget", {})
        searcher_cfg = self.config.get("searcher", {})

        self.harness = SearcherHarness(
            searcher_model=s_model["model_name"],
            provider=s_model.get("provider", "anthropic"),
            base_url=s_model.get("base_url", ""),
            api_key_env=s_model.get("api_key_env", ""),
            temperature=s_model.get("temperature", searcher_cfg.get("temperature", 0.2)),
            thinking_budget=s_model.get(
                "thinking_budget", searcher_cfg.get("thinking_budget")
            ),
            thinking_instruction=searcher_cfg.get("thinking_instruction", True),
            thinking_passthrough=searcher_cfg.get("thinking_passthrough", True),
            force_tools=s_model.get("force_tools", searcher_cfg.get("force_tools", False)),
            max_searches=budget.get("max_searches", 5),
            scratchpad_max_tokens=budget.get("scratchpad_max_tokens", 512),
            results_per_query=budget.get("results_per_query", 5),
            max_nudges=searcher_cfg.get("max_nudges", 5),
            max_shrink_attempts=searcher_cfg.get("max_shrink_attempts", 2),
            verbose=self.config.get("verbose", False),
        )

        ext_cfg = self.config.get("extractor", {})
        # Extractor precedence: agent yaml wins over model yaml so the extractor
        # stays deterministic even when the shared model config enables sampling
        # for its searcher role (e.g. claude_sonnet uses temperature=1.0 for
        # extended thinking).
        self.extractor = Extractor(
            model_name=e_model["model_name"],
            provider=e_model.get("provider", "anthropic"),
            base_url=e_model.get("base_url", ""),
            api_key_env=e_model.get("api_key_env", ""),
            temperature=ext_cfg.get("temperature", e_model.get("temperature", 0.0)),
            max_tokens=ext_cfg.get("max_tokens", 1024),
        )

        self.trajectories_dir = TRAJECTORIES_DIR / self.name

    def answer(self, task: Task) -> Answer:
        t0 = time.time()
        console.rule(f"[bold]Task idx-{task.idx}[/bold]")

        trace = self.harness.run(task)
        trace.agent = self.name
        trace.save(self.trajectories_dir / f"idx-{task.idx}.json")

        sources = trace.sources_in_order()
        if not sources:
            console.print("  [red]No sources — extractor will see nothing[/red]")

        answer = self.extractor.extract(task.question, sources)
        elapsed = time.time() - t0
        console.print(
            f"  [green]Done[/green] — {len(sources)} sources, "
            f"exact='{answer.exact_answer[:60]}', "
            f"conf={answer.confidence}, {elapsed:.1f}s"
        )
        return answer
