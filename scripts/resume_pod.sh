#!/bin/bash
# =============================================================
# Restore a RunPod pod after a stop/start.
#
# WHY THIS IS NEEDED
#   RunPod wipes the CONTAINER disk when a pod is stopped, so every pip
#   install is lost on restart. The VOLUME disk (/workspace) survives, which
#   means the code and the downloaded model weights are still there — only the
#   Python environment has to be rebuilt.
#
#   Rebuilding it by hand is where mistakes creep in: forgetting to upgrade
#   torchvision alongside torch, or leaving the memory budget at a value tuned
#   for a different model. This script does the whole sequence the same way
#   every time.
#
# SAFE TO RE-RUN. Every step checks before acting.
#
# Usage:
#   bash /workspace/resume_pod.sh
# =============================================================

set -uo pipefail

PROJECT="${PROJECT:-/workspace/agentic-safety-probe}"
# Fraction of each GPU's VRAM HELD BACK from accelerate. See build_max_memory
# in src/model_loader.py: the remaining budget is a ceiling that shapes the
# placement plan, and it must exceed the model's total size or accelerate
# spills to disk and the load fails. 0.12 leaves 164 GiB of a 186 GiB pod,
# which fits Llama-3.3-70B in FP16 (132 GiB) with room for activations.
export PROBE_GPU_RESERVE="${PROBE_GPU_RESERVE:-0.12}"

echo "============================================================"
echo "RESUMING POD"
echo "============================================================"

# ---------- 1. Environment ----------
echo
echo "--- Environment ---"
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME=/workspace/hf-cache
    echo "  HF_HOME was unset -> exported ${HF_HOME}"
    echo "  Add it to the pod template so it survives the next restart."
else
    echo "  HF_HOME=${HF_HOME}"
fi
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

if [ -z "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN is NOT set — required for gated models (Llama, Mistral)."
else
    echo "  HF_TOKEN: set"
fi

# ---------- 2. Project ----------
echo
echo "--- Project ---"
if [ ! -f "${PROJECT}/run_experiment.py" ]; then
    echo "  ERROR: ${PROJECT}/run_experiment.py not found."
    echo "  The volume should have survived the stop. Check the mount."
    exit 1
fi
cd "${PROJECT}" || exit 1
echo "  ${PROJECT}: OK"

# ---------- 3. System packages ----------
echo
echo "--- System packages ---"
if ! command -v tmux >/dev/null 2>&1; then
    echo "  installing tmux..."
    apt-get update -qq && apt-get install -y -qq tmux
fi
echo "  tmux: $(tmux -V 2>/dev/null || echo MISSING)"

# ---------- 4. Python packages ----------
# torch, torchvision and torchaudio are upgraded in ONE command: they ship
# compiled extensions built against a specific torch, and bumping torch alone
# leaves them unable to register their operators. The symptom is a misleading
# "operator torchvision::nms does not exist" followed by a bogus
# "Could not import module 'Qwen3MoeForCausalLM'".
echo
echo "--- Python packages ---"
NEED_INSTALL=0
python -c "import transformers, accelerate" 2>/dev/null || NEED_INSTALL=1
python -c "
import torch, sys
sys.exit(0 if tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 5) else 1)
" 2>/dev/null || NEED_INSTALL=1

if [ "${NEED_INSTALL}" -eq 1 ]; then
    echo "  rebuilding the environment (container disk was wiped)..."
    pip install -q -r requirements_runpod.txt
    pip install -q --upgrade torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "  already present, skipping install"
fi

python -c "
import torch, transformers, accelerate
print(f'  torch       : {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  accelerate  : {accelerate.__version__}')
print(f'  GPUs        : {torch.cuda.device_count()}')
total = sum(torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())) / 1024**3
print(f'  VRAM total  : {total:.0f} GiB')
"

# ---------- 5. Memory budget ----------
# Read from PROBE_GPU_RESERVE by build_max_memory, so no source editing is
# needed. If that function is absent, this pod is running an older copy of the
# file than the repo and should be refreshed before anything else.
echo
echo "--- Memory budget ---"
if python -c "from src.model_loader import build_max_memory" 2>/dev/null; then
    echo "  PROBE_GPU_RESERVE=${PROBE_GPU_RESERVE}"
    python -c "
from src.model_loader import build_max_memory
print('  budget:', build_max_memory())
"
else
    echo "  WARNING: src/model_loader.py has no build_max_memory()."
    echo "  This pod is running an outdated copy. Re-upload src/model_loader.py"
    echo "  from the repo before running the experiment."
fi

# ---------- 6. Cached models ----------
echo
echo "--- Cached models ---"
du -sh "${HF_HOME}"/hub/models--* 2>/dev/null || echo "  (none yet)"
df -h /workspace | tail -1

echo
echo "============================================================"
echo "READY"
echo "============================================================"
cat <<'NEXT'

If the model is not cached yet:
    tmux new -s dl
    hf download meta-llama/Llama-3.3-70B-Instruct

Dry run (16 forward passes, validates the whole path):
    python run_experiment.py --model meta-llama/Llama-3.3-70B-Instruct \
      --no-quantize --limit-pairs 4 --skip-intervention \
      --n-repeats 1 --n-label-permutations 0 --output-dir /workspace/dryrun

Real run:
    mkdir -p /workspace/results_llama33
    python run_experiment.py --model meta-llama/Llama-3.3-70B-Instruct \
      --no-quantize --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 \
      --n-label-permutations 10 --n-bootstrap 10000 --decompose \
      --output-dir /workspace/results_llama33 \
      2>&1 | tee /workspace/results_llama33/run.log

NEXT
