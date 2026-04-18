"""Serve a trained Qwen3 SFT checkpoint via vLLM's OpenAI-compatible endpoint.

Thin wrapper around `vllm.entrypoints.openai.api_server`. The served endpoint
is wired to agents via a model config YAML with:

    provider: "custom"
    base_url: "http://localhost:8000"
    api_key_env: ""
    model_name: "Qwen/Qwen3-8B"   # or your checkpoint path / alias

After starting this, run:

    make bench AGENT=lean_searcher MODEL=<your_served_model_yaml> SPLIT=dev
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="YAML with serving args")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    serve = cfg.get("serve", cfg)  # accept {serve: {...}} or flat
    cmd = [
        "vllm",
        "serve",
        serve["model_path"],
        "--host", serve.get("host", "0.0.0.0"),
        "--port", str(serve.get("port", 8000)),
    ]
    if served_name := serve.get("served_name"):
        cmd.extend(["--served-model-name", served_name])
    if dtype := serve.get("dtype"):
        cmd.extend(["--dtype", dtype])
    for flag in ("enable-lora", "trust-remote-code"):
        if serve.get(flag):
            cmd.append(f"--{flag}")
    if gpu_mem := serve.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(gpu_mem)])

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
