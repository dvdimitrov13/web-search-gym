"""Auto-discovery of agents and loading of model configs.

Agent directory conventions:
- agents/<name>/agent.py contains exactly one BaseAgent subclass
- agents/<name>/config.yaml holds its harness config
- directories starting with "_" are ignored (e.g. _template)

Model directory conventions:
- models/<name>.yaml is a model config dict, loaded by name
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml

from agents.base import BaseAgent

_REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = _REPO_ROOT / "agents"
MODELS_DIR = _REPO_ROOT / "models"


def discover_agents() -> dict[str, type[BaseAgent]]:
    """Scan agents/*/agent.py for BaseAgent subclasses, keyed by directory name."""
    found: dict[str, type[BaseAgent]] = {}
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        if not (agent_dir / "agent.py").exists():
            continue
        module = importlib.import_module(f"agents.{agent_dir.name}.agent")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                if (agent_dir / "config.yaml").exists():
                    found[agent_dir.name] = obj
                break
    return found


def list_model_configs() -> list[str]:
    """Return the names of all model YAMLs (without extension)."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))


def load_model_config(name: str) -> dict:
    """Load models/<name>.yaml."""
    path = MODELS_DIR / f"{name}.yaml"
    if not path.exists():
        avail = ", ".join(list_model_configs()) or "(none)"
        raise ValueError(f"Model config {name!r} not found. Available: {avail}")
    with open(path) as f:
        return yaml.safe_load(f)


# Simplifying assumption: the short-answer extractor is always Sonnet unless
# the caller explicitly overrides with --extractor-model. This keeps the
# extractor stage identical across agents so benchmark deltas reflect the
# searcher, not the formatter.
DEFAULT_EXTRACTOR = "claude_sonnet"


def load_agent(
    name: str,
    *,
    model: str | None = None,
    searcher_model: str | None = None,
    extractor_model: str | None = None,
) -> BaseAgent:
    """Instantiate an agent by directory name with resolved model configs.

    `--model` (shorthand) only targets the searcher. The extractor is pinned to
    DEFAULT_EXTRACTOR unless `--extractor-model` is passed explicitly.

    The returned agent has `self.model_configs` pre-populated with a dict:
        {"searcher": <loaded model dict>, "extractor": <loaded model dict>}
    for each stage resolved at load time. The agent class decides which
    stages it cares about.
    """
    agents = discover_agents()
    if name not in agents:
        available = ", ".join(sorted(agents.keys())) or "(none)"
        raise ValueError(f"Agent {name!r} not found. Available: {available}")

    agent_cls = agents[name]
    config_path = AGENTS_DIR / name / "config.yaml"

    resolved: dict[str, dict] = {}
    searcher = searcher_model or model
    extractor = extractor_model or DEFAULT_EXTRACTOR
    if searcher:
        resolved["searcher"] = load_model_config(searcher)
    resolved["extractor"] = load_model_config(extractor)

    return agent_cls(config_path, resolved)
