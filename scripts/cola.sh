#!/bin/bash
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Each run goes in the next free version of its model, never on top of an
# existing one (results/README.md). run_modelo.sh already uses dataset_v2.

# Espera a que termine lo que esté corriendo ahora
while pgrep -u "$USER" -f run_experiment.py > /dev/null; do
    echo "[$(date +%H:%M:%S)] esperando a la corrida en curso..."
    sleep 60
done

echo "=== [1/2] Qwen3-8B FP16  $(date +%H:%M:%S) ==="
./scripts/run_modelo.sh Qwen/Qwen3-8B ./results/corridas/qwen3_8b/v3 --no-quantize \
  || echo "!!! Qwen3-8B FALLO, sigo con el siguiente"

echo "=== [2/2] Mistral-7B FP16  $(date +%H:%M:%S) ==="
./scripts/run_modelo.sh mistralai/Mistral-7B-Instruct-v0.3 ./results/corridas/mistral7b/v4 \
  --no-quantize || echo "!!! Mistral FP16 FALLO"

echo "=== COLA TERMINADA $(date) ==="
