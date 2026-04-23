"""Exa Deep agent — a single `/search` call at a chosen deep tier.

Not a harness: the "searcher" is one Exa call (deep-lite | deep | deep-reasoning).
Exa runs its own multi-step query plan internally; we hand the returned sources
to the same Sonnet extractor used by lean_searcher so BrowseComp comparisons are
apples-to-apples against the harness.

The tier comes from `--model <exa_deep_lite | exa_deep | exa_deep_reasoning>`.
"""

from __future__ import annotations

import time
from pathlib import Path

from agents.base import BaseAgent
from core.console import console
from core.exa_client import ExaClient
from core.extractor import Extractor
from core.trace import SubmittedUrl, Trace, TraceMetadata
from core.types import Answer, Task


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORIES_DIR = _REPO_ROOT / "trajectories"


class ExaDeepAgent(BaseAgent):
    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        super().__init__(config_path, model_configs)

        if "searcher" not in model_configs:
            raise ValueError(
                "exa_deep requires --model <exa_deep_lite | exa_deep | "
                "exa_deep_reasoning>."
            )
        if "extractor" not in model_configs:
            raise ValueError(
                "exa_deep requires an extractor model (default: claude_sonnet)."
            )

        s = model_configs["searcher"]
        if s.get("provider") != "exa":
            raise ValueError(
                f"exa_deep expected an Exa tier model, got "
                f"provider={s.get('provider')!r}. Use one of: "
                "exa_deep_lite, exa_deep, exa_deep_reasoning."
            )
        self.tier: str = s["model_name"]  # deep-lite | deep | deep-reasoning
        self.verbose: bool = bool(self.config.get("verbose", False))
        self.exa = ExaClient()

        e = model_configs["extractor"]
        ext_cfg = self.config.get("extractor", {})
        # Agent yaml wins over model yaml for extractor temperature so shared
        # model configs (e.g. claude_sonnet with thinking temperature=1.0) don't
        # leak sampling into the deterministic extractor.
        self.extractor = Extractor(
            model_name=e["model_name"],
            provider=e.get("provider", "anthropic"),
            base_url=e.get("base_url", ""),
            api_key_env=e.get("api_key_env", ""),
            temperature=ext_cfg.get("temperature", e.get("temperature", 0.0)),
            max_tokens=ext_cfg.get("max_tokens", 1024),
        )

        self.trajectories_dir = TRAJECTORIES_DIR / self.name

    def answer(self, task: Task) -> Answer:
        t0 = time.time()
        console.rule(f"[bold]Task idx-{task.idx}[/bold]")

        if self.verbose:
            console.print(
                f"  [cyan]Exa {self.tier}:[/cyan] [dim]{task.question[:90]}[/dim]"
            )

        results = self.exa.deep_search(task.question, tier=self.tier)

        if self.verbose:
            console.print(
                f"  [green]Sources:[/green] {len(results.sources)} · "
                f"[green]synthesis:[/green] {len(results.synthesis)} chars"
            )
            if results.synthesis:
                preview = results.synthesis.replace("\n", " ")[:160]
                console.print(f"  [dim]↳ {preview}…[/dim]")

        # Write a Trace for downstream parity (SFT, RL reward, replay). Deep
        # search doesn't produce a tool-call transcript, so `messages` is empty;
        # the multi-hop reasoning lives in `synthesis`.
        trace = Trace(
            task_idx=task.idx,
            question=task.question,
            messages=[],
            source_bank={s.url: s.to_dict() for s in results.sources},
            submitted=[SubmittedUrl(url=s.url, score=1.0) for s in results.sources],
            metadata=TraceMetadata(
                search_count=1,
                elapsed_seconds=round(time.time() - t0, 2),
                stop_reason="exa_deep",
            ),
            agent=self.name,
            searcher_model=self.tier,
            synthesis=results.synthesis,
        )
        trace.save(self.trajectories_dir / f"idx-{task.idx}.json")

        if not results.synthesis and not results.sources:
            console.print("  [red]No synthesis and no sources — extractor will see nothing[/red]")

        # DeepSearchQA (and any grader that wants prose) reads a full natural
        # response. Skip the extractor — its single-line `Exact Answer:` field
        # collapses Set Answer enumerations. We still stash the synthesis on
        # all three legacy fields so short-answer tooling keeps working.
        if task.metadata.get("dataset") == "deepsearchqa" and results.synthesis:
            elapsed = time.time() - t0
            console.print(
                f"  [green]Done (no-extract)[/green] — {len(results.sources)} sources, "
                f"synthesis={len(results.synthesis)} chars, {elapsed:.1f}s"
            )
            return Answer(
                explanation=results.synthesis,
                exact_answer=results.synthesis,
                confidence=80,
                natural_text=results.synthesis,
            )

        # When we have Exa's synthesis, feed only that to the extractor — the
        # per-URL source_bank has no content (we skipped per-page summaries).
        answer = self.extractor.extract(
            task.question,
            sources=[] if results.synthesis else results.sources,
            synthesis=results.synthesis,
        )
        answer.natural_text = results.synthesis
        elapsed = time.time() - t0
        console.print(
            f"  [green]Done[/green] — {len(results.sources)} sources, "
            f"exact='{answer.exact_answer[:60]}', "
            f"conf={answer.confidence}, {elapsed:.1f}s"
        )
        return answer
