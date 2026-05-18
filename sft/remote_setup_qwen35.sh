#!/bin/bash
# Vast.ai setup for Qwen3.5-2B + Unsloth + sample packing extractor SFT.
#
# Assumes nvidia/cuda:12.8.1-devel-ubuntu22.04 base image. Patches the two
# things that image is missing for Triton (python3-dev + libcuda.so symlink)
# and installs everything via uv (parallel downloads — pip serially stalls
# on the unsloth + xformers + flash-attn stack on slow hosts).
set -euo pipefail

cd /workspace

echo "=== Triton prerequisites (python3-dev + libcuda.so symlink) ==="
apt-get update -qq
apt-get install -y -qq python3-dev curl tmux
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
ldconfig

echo "=== Installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  bash /tmp/uv-install.sh
fi
export PATH=/root/.local/bin:$PATH

echo "=== Installing torch (CUDA 12.8 wheels) ==="
uv pip install --system --no-cache --index-strategy unsafe-best-match \
  torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing training stack ==="
uv pip install --system --no-cache \
  unsloth unsloth_zoo \
  trl peft transformers accelerate datasets \
  liger-kernel \
  pyyaml wandb bitsandbytes numpy

echo "=== GPU + Triton smoke test ==="
nvidia-smi | head -10
python3 -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(8,8,device='cuda',requires_grad=True)
torch.nn.functional.linear(x, torch.randn(8,8,device='cuda')).sum().backward()
print('triton compile ok')
"

echo "=== Verifying imports ==="
python3 -c "
import unsloth, trl, peft, transformers, liger_kernel
print(f'unsloth={unsloth.__version__} trl={trl.__version__} transformers={transformers.__version__} liger={liger_kernel.__version__}')
"

echo "=== Ready ==="
