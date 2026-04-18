.PHONY: help setup bench synth convert train serve list-agents new-agent lint test clean

PYTHON := uv run python

AGENT ?= lean_searcher
MODEL ?=
SEARCHER_MODEL ?=
EXTRACTOR_MODEL ?=
SPLIT ?= dev
RUN_ID ?= $(shell date +%Y%m%d_%H%M%S)
CONCURRENT ?= 1
MODE ?= per-turn
CONFIG ?=
QUESTIONS ?=

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === Environment ===

setup: ## uv sync + .env
	uv sync --all-extras
	@cp -n .env.example .env 2>/dev/null || true
	@echo ""
	@echo "Done. Edit .env with your API keys (ANTHROPIC, EXA, OPENAI)."

# === Benchmark ===

bench: ## Run bench (usage: make bench AGENT=lean_searcher MODEL=claude_sonnet SPLIT=dev)
	$(PYTHON) -m bench.cli run \
		--agent $(AGENT) \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(SEARCHER_MODEL),--searcher-model $(SEARCHER_MODEL),) \
		$(if $(EXTRACTOR_MODEL),--extractor-model $(EXTRACTOR_MODEL),) \
		--split $(SPLIT) \
		--concurrent $(CONCURRENT) \
		--run-id $(RUN_ID)

leaderboard: ## Show leaderboard across all agent × model runs
	$(PYTHON) -m bench.cli leaderboard

list-agents: ## List discovered agents
	$(PYTHON) -m bench.cli list-agents

list-models: ## List available model configs
	$(PYTHON) -m bench.cli list-models

# === Agent scaffolding ===

new-agent: ## Scaffold a new agent harness (make new-agent NAME=my_harness)
	@test -n "$(NAME)" || (echo "ERROR: specify NAME=<agent_name>" && exit 1)
	@cp -r agents/_template agents/$(NAME)
	@sed -i '' 's/template/$(NAME)/g' agents/$(NAME)/config.yaml 2>/dev/null || \
		sed -i 's/template/$(NAME)/g' agents/$(NAME)/config.yaml
	@echo "Created agents/$(NAME)/ -- edit agent.py and config.yaml"

# === Synth (data generation) ===

synth: ## Generate trajectories (make synth MODEL=claude_sonnet QUESTIONS=path/to/q.jsonl)
	@test -n "$(QUESTIONS)" || (echo "ERROR: specify QUESTIONS=<path>" && exit 1)
	$(PYTHON) -m synth.generate \
		--questions $(QUESTIONS) \
		$(if $(MODEL),--model $(MODEL),)

# === SFT ===

convert: ## Convert trajectories to Qwen3 SFT format (make convert MODE=per-turn INPUT=... OUTPUT=...)
	@test -n "$(INPUT)" || (echo "ERROR: specify INPUT=<path>" && exit 1)
	@test -n "$(OUTPUT)" || (echo "ERROR: specify OUTPUT=<path>" && exit 1)
	$(PYTHON) -m sft.convert --input $(INPUT) --output $(OUTPUT) --mode $(MODE)

train: ## Train SFT model (make train CONFIG=sft/configs/config.yaml)
	@test -n "$(CONFIG)" || (echo "ERROR: specify CONFIG=<path>" && exit 1)
	$(PYTHON) -m sft.train --config $(CONFIG)

serve: ## Serve a trained SFT checkpoint via vLLM
	@test -n "$(CONFIG)" || (echo "ERROR: specify CONFIG=<path>" && exit 1)
	$(PYTHON) -m sft.serve --config $(CONFIG)

# === Development ===

lint: ## Run ruff
	uv run ruff check core/ agents/ bench/ synth/ sft/ tests/

test: ## Run pytest
	uv run pytest tests/ -v

clean: ## Remove generated reports, trajectories, results
	rm -rf reports/* trajectories/* results/*
