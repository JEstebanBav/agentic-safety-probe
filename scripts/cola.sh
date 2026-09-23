#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Espera a que termine lo que esté corriendo ahora (Mistral 4-bit)
while pgrep -u "$USER" -f run_experiment.py > /dev/null; do
    echo "[$(date +%H:%M:%S)] esperando a la corrida en curso..."
    sleep 60
done

echo "=== [1/2] Qwen3-8B FP16  $(date +%H:%M:%S) ==="
./scripts/run_modelo.sh Qwen/Qwen3-8B ./results/qwen8b --no-quantize \
  || echo "!!! Qwen3-8B FALLO, sigo con el siguiente"

echo "=== [2/2] Mistral-7B FP16 con intervencion arreglada  $(date +%H:%M:%S) ==="
./scripts/run_modelo.sh mistralai/Mistral-7B-Instruct-v0.3 ./results/mistral7b_fp16 \
  --no-quantize || echo "!!! Mistral FP16 FALLO"

echo "=== COLA TERMINADA $(date) ==="
