"""CLI for the gym. Thin Click wrapper over agents/ + bench/."""

from __future__ import annotations

import time
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()

_REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_YAML = _REPO_ROOT / "bench" / "splits.yaml"


def _load_split_indices(name: str) -> list[int] | None:
    if not SPLITS_YAML.exists():
        raise click.UsageError(f"Missing {SPLITS_YAML}")
    with open(SPLITS_YAML) as f:
        data = yaml.safe_load(f) or {}
    if name not in data:
        raise click.UsageError(
            f"Unknown split '{name}'. Known: {', '.join(sorted(data))}"
        )
    return data[name].get("indices")


@click.group()
def main():
    """web-search-gym: Train and benchmark web-search agents on BrowseComp."""


@main.command("list-agents")
def list_agents():
    """Show all discovered agent harnesses."""
    from agents.registry import discover_agents

    found = discover_agents()
    if not found:
        console.print("[yellow]No agents found. Create one from agents/_template/[/yellow]")
        return
    table = Table(title="Available Agents")
    table.add_column("Name", style="bold")
    table.add_column("Class")
    for name, cls in sorted(found.items()):
        table.add_row(name, cls.__name__)
    console.print(table)


@main.command("list-models")
def list_models():
    """Show all model configs in models/."""
    from agents.registry import list_model_configs, load_model_config

    names = list_model_configs()
    if not names:
        console.print("[yellow]No models/*.yaml found.[/yellow]")
        return
    table = Table(title="Available Models")
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("Model ID")
    for n in names:
        try:
            cfg = load_model_config(n)
        except Exception as e:
            table.add_row(n, "[red]error[/red]", str(e))
            continue
        table.add_row(n, cfg.get("provider", "?"), cfg.get("model_name", "?"))
    console.print(table)


@main.command("run")
@click.option("--agent", "-a", default="lean_searcher", help="Agent name (default: lean_searcher)")
@click.option("--model", "-m", default=None, help="Model name applied to all stages")
@click.option("--searcher-model", default=None, help="Override searcher-stage model")
@click.option("--extractor-model", default=None, help="Override extractor-stage model")
@click.option("--split", "-s", default="dev", help="Split: dev | smoke | full (default: dev)")
@click.option("--indices", default=None, help="Comma-separated row indices (overrides --split)")
@click.option("--concurrent", "-c", default=1, help="Max concurrent tasks")
@click.option("--run-id", default=None, help="Run ID (default: timestamp)")
def run_cmd(agent, model, searcher_model, extractor_model, split, indices, concurrent, run_id):
    """Run an agent on BrowseComp tasks and grade it."""
    from agents.registry import load_agent
    from bench.browsecomp import load_tasks
    from bench.runner import run as run_bench

    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    if indices:
        idx_list = [int(x.strip()) for x in indices.split(",") if x.strip()]
    else:
        idx_list = _load_split_indices(split)

    agent_inst = load_agent(
        agent,
        model=model,
        searcher_model=searcher_model,
        extractor_model=extractor_model,
    )

    tasks = load_tasks(indices=idx_list)
    if not tasks:
        console.print("[yellow]No tasks to run.[/yellow]")
        return

    run_bench(agent_inst, tasks, run_id=run_id, max_concurrent=concurrent)


@main.command("leaderboard")
def leaderboard():
    """Show accuracy across all scored runs."""
    from bench.compare import print_leaderboard

    print_leaderboard()


if __name__ == "__main__":
    main()
