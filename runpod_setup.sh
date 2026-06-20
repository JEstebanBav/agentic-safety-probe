#!/bin/bash
# =============================================================
# RunPod Setup Script
# Run this after SSH-ing into your RunPod instance.
#
# Recommended pod:
#   - GPU: RTX 4090 (24GB) or A5000 (24GB)
#   - Template: RunPod PyTorch 2.1+ (CUDA 12.x)
#   - Disk: 50GB (model ~14GB + dependencies)
#   - Budget: ~$0.39/hr → $10 gives ~25 hours
#
# Usage:
#   chmod +x runpod_setup.sh && ./runpod_setup.sh
# =============================================================

set -e

echo "=== RunPod Setup for Agentic Safety Probe ==="

# 1. Clone or copy your repo (adjust URL as needed)
# git clone https://github.com/YOUR_USER/agentic-safety-probe.git
# cd agentic-safety-probe

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Login to Hugging Face (needed for gated models like Mistral)
echo ""
echo "=== Hugging Face Login ==="
echo "You need a HF token for Mistral-7B (it's a gated model)."
echo "Get your token at: https://huggingface.co/settings/tokens"
echo "Then accept the model license at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
echo ""
huggingface-cli login

# 4. Quick GPU check
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 5. Test model loading
python -c "
from src.model_loader import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer(quantize=False)
print('Model loaded successfully in FP16!')
"

echo ""
echo "=== Setup Complete ==="
echo "Run the experiment with:"
echo "  python run_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3 --no-quantize"
echo ""
echo "Or for a quick test:"
echo "  python run_experiment.py --no-quantize --n-samples 3 --layers 10,14,18,22"
