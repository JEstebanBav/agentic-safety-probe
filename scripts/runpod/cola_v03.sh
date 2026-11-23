#!/bin/bash
# Las cuatro corridas sobre el corpus v0.3, una tras otra, en un pod con una
# H200 (141 GB). Cada una usa los mismos flags que su corrida anterior
# publicada, para que los resultados sean comparables:
#   Mistral-7B (v4), Llama-3.3-70B (v4: 20 capas, test causal en 76,56,28),
#   Qwen3-8B (v3) y Qwen3-32B (v3).
# Orden: de la mas corta a la mas larga. Si algo falla en la primera se ve
# pronto, y si el presupuesto se acaba quedan hechas las tres primeras.
# Si una corrida falla, se anota y la cola sigue con la siguiente.
#
# Uso, dentro de tmux y desde la raiz del repositorio:
#   bash scripts/runpod/cola_v03.sh                  # salida en /workspace/out/corridas
#   bash scripts/runpod/cola_v03.sh /otra/carpeta
cd "$(dirname "$0")/../.."
BASE="${1:-/workspace/out/corridas}"

# nombre  carpeta  reserva_gpu  modelo  [flags extra]
run() {
  local name=$1 out=$2 reserve=$3 model=$4
  shift 4
  if [ -f "$out/results.json" ]; then
    echo "=== $name: ya existe $out/results.json, se omite ==="
    return
  fi
  mkdir -p "$out"
  echo "=== $name  inicio $(date '+%F %T') ==="
  if PROBE_GPU_RESERVE="$reserve" bash scripts/runpod/run_modelo_v03.sh \
       "$model" "$out" --no-quantize "$@" > "$out/run.log" 2>&1; then
    echo "    terminado $(date '+%F %T')"
  else
    echo "!!! $name FALLO $(date '+%F %T'): ver $out/run.log. Sigo con el siguiente."
  fi
}

run "Mistral-7B"    "$BASE/mistral7b/v5"   0.12 mistralai/Mistral-7B-Instruct-v0.3
# 131 GiB de pesos en ~140 GiB de tarjeta: con la reserva por defecto (12 %)
# el presupuesto no alcanza y la carga falla. 0.04 deja 134 GiB, lo mismo que
# registro el log de la corrida v4.
run "Llama-3.3-70B" "$BASE/llama33_70b/v5" 0.04 meta-llama/Llama-3.3-70B-Instruct \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76 \
  --intervention-layers 76,56,28
run "Qwen3-8B"      "$BASE/qwen3_8b/v4"    0.12 Qwen/Qwen3-8B
run "Qwen3-32B"     "$BASE/qwen3_32b/v4"   0.12 Qwen/Qwen3-32B

echo "=== COLA TERMINADA $(date '+%F %T') ==="
