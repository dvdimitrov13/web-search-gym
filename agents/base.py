"""BaseAgent protocol.

An agent is a *harness shape*, not a model. The agent receives two config
blocks at construction: its own harness config (budgets, timeouts) and a set
of model configs (one per stage). It instantiates the underlying `core/`
components with those configs.

The contract is simple:

    agent.answer(task) -> Answer

Everything else (searching, extracting, trace writing) is internal detail.
The benchmark runner only needs `answer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from core.types import Answer, Task


class BaseAgent(ABC):
    """Abstract base class for all harness shapes.

    Create a new agent by:
    1. cp -r agents/_template agents/my_harness
    2. Edit config.yaml with its harness settings
    3. Implement `answer(task) -> Answer`
    4. Registry auto-discovers it; invoke via
         make bench AGENT=my_harness MODEL=<any_model>
    """

    def __init__(self, config_path: Path, model_configs: dict[str, dict]):
        """
        Args:
            config_path: path to agent's config.yaml.
            model_configs: dict mapping stage name ("searcher", "extractor")
                to a resolved model config dict loaded from `models/*.yaml`.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.name = self.config["agent"]["name"]
        self.display_name = self.config["agent"].get("display_name", self.name)
        self.model_configs = model_configs

    @abstractmethod
    def answer(self, task: Task) -> Answer:
        """Produce an Answer for a task. Must be implemented by subclasses."""
        ...

    def setup(self) -> None:
        """Optional one-time setup before the first task."""
        pass

    def teardown(self) -> None:
        """Optional cleanup after all tasks."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
