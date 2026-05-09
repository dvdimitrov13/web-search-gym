#!/bin/bash
# Runs on the vast.ai instance after files are uploaded.
# Assumes: /workspace/sft/{train.py, configs/cloud.yaml, data/agent_dd_mixed.norm.jsonl.gz}
#          ~/.cache/huggingface/token (for Gemma 4 gated access)
set -euo pipefail

cd /workspace

# Decompress data
if [ ! -f sft/data/agent_dd_mixed.norm.jsonl ]; then
  echo "=== Decompressing training data ==="
  gunzip -k sft/data/agent_dd_mixed.norm.jsonl.gz
  ls -lh sft/data/agent_dd_mixed.norm.jsonl
fi

# Install Python deps. Base image has torch+cuda already.
echo "=== Installing training deps ==="
pip install -q --no-cache-dir \
  'transformers>=4.46' \
  'trl==1.2.0' \
  'peft>=0.13' \
  'datasets>=3.0' \
  'accelerate>=1.0' \
  'liger-kernel>=0.5' \
  'pyyaml' \
  'huggingface_hub[cli]'

echo "=== GPU check ==="
nvidia-smi | head -20
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== HF login check ==="
python -c "from huggingface_hub import whoami; print(whoami().get('name'))" || {
  echo "ERROR: HF token not working. Make sure ~/.cache/huggingface/token was uploaded."
  exit 1
}

echo "=== Ready to train ==="
