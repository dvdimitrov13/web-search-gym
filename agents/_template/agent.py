"""Template harness — copy this directory to create a new agent.

Steps:
1. cp -r agents/_template agents/my_harness
2. Edit config.yaml: rename `template` → `my_harness`, set budgets
3. Replace the `answer` body below with your harness logic
4. Run: make bench AGENT=my_harness MODEL=claude_sonnet SPLIT=dev
"""

from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent
from core.types import Answer, Task


class TemplateAgent(BaseAgent):
    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        super().__init__(config_path, model_configs)
        # TODO: instantiate harness, extractor, etc. from core.*

    def answer(self, task: Task) -> Answer:
        # TODO: replace with real implementation
        raise NotImplementedError("Implement answer() in your agent")
