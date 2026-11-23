#!/bin/bash
# Corridas del corpus v0.3, por maquina. Ver docs/runpod/guia_corrida_v03.md.
#
#   bash scripts/runpod/cola_v03.sh <maquina> <modo> [carpeta-base]
#
#   maquina: h200  1x H200: Llama-3.3-70B, Qwen3-32B y los 5 modelos nuevos
#            l40s  1x L40S: Mistral-7B y Qwen3-8B
#            glm   4x H200: GLM-5.3-Flash, solo extraccion de activaciones
#   modo:    prueba  4 pares por modelo, todas las etapas (obligatoria antes)
#            real    corridas completas
#
# Todos los modelos usan los 96 pares en todas las etapas, test causal
# incluido. Las re-corridas (Mistral, Qwen3-8B, Qwen3-32B, Llama) usan los
# flags de su corrida publicada y FP16, como antes; los modelos nuevos van en
# BF16, su precision nativa.
# Comun a todos: hasta PENSAR tokens de razonamiento (4096; con 1024 el 31-33 %
# de las respuestas agenticas de Qwen3-8B no terminaba) y 3 direcciones
# aleatorias en el control causal (las 2 extra en las dosis x1 y x4). Los
# modelos que no razonan ignoran el limite de razonamiento.
# Si una corrida falla, se anota y la cola sigue. Una corrida que ya tiene
# results.json se omite, asi que relanzar la cola continua donde quedo.
set -u
cd "$(dirname "$0")/../.."
MAQUINA="${1:?uso: cola_v03.sh <h200|l40s|glm> <prueba|real> [carpeta-base]}"
MODO="${2:?falta el modo: prueba o real}"
NUEVOS="--dtype bfloat16"
COMUN="--thinking-budget ${PENSAR:-4096} --n-random-controls 3 --extra-control-doses 1,4"

case "$MODO" in
  prueba) BASE="${3:-/workspace/out/pruebas}"
          EXTRA="--limit-pairs 4 --n-repeats 1 --n-label-permutations 0 --n-bootstrap 1000 --n-selection-splits 20" ;;
  real)   BASE="${3:-/workspace/out/corridas}"; EXTRA="" ;;
  *) echo "modo desconocido: $MODO"; exit 1 ;;
esac

# nombre  carpeta  reserva_gpu  modelo  [flags]
run() {
  local name=$1 out=$2 reserve=$3 model=$4
  shift 4
  if [ -f "$out/results.json" ]; then
    echo "=== $name: ya existe $out/results.json, se omite ==="
    return
  fi
  mkdir -p "$out"
  echo "=== $name  inicio $(date '+%F %T') ==="
  # $EXTRA va al final: en modo prueba prevalece sobre los flags del modelo.
  # shellcheck disable=SC2086
  if PROBE_GPU_RESERVE="$reserve" PYTHONUNBUFFERED=1 bash scripts/runpod/run_modelo_v03.sh \
       "$model" "$out" --no-quantize $COMUN "$@" $EXTRA > "$out/run.log" 2>&1; then
    echo "    terminado $(date '+%F %T')  $(grep -h 'Generation stages' "$out/run.log" | tail -1)"
  else
    echo "!!! $name FALLO $(date '+%F %T'): ver $out/run.log. Sigo con el siguiente."
  fi
}

LLAMA_CAPAS="0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76"

case "$MAQUINA" in
  h200)
    # 131 GiB de pesos en ~140 GiB de tarjeta: con la reserva por defecto
    # (12 %) el presupuesto no alcanza. 0.04 deja 134 GiB, como en la v4.
    run "Llama-3.3-70B" "$BASE/llama33_70b/v5" 0.04 meta-llama/Llama-3.3-70B-Instruct \
        --layers "$LLAMA_CAPAS" --intervention-layers 76,56,28
    run "Qwen3-32B"        "$BASE/qwen3_32b/v4"        0.12 Qwen/Qwen3-32B
    # shellcheck disable=SC2086
    {
    run "Qwen3.8-27B"      "$BASE/qwen38_27b/v1"       0.12 Qwen/Qwen3.8-27B $NUEVOS
    run "Gemma-4-31B-it"   "$BASE/gemma4_31b/v1"       0.12 google/gemma-4-31B-it $NUEVOS
    run "Muse-Glimmer-30B" "$BASE/muse_glimmer_30b/v1" 0.12 meta-models/Muse-Glimmer-30B $NUEVOS
    run "gpt-oss-20b"      "$BASE/gpt_oss_20b/v1"      0.12 openai/gpt-oss-20b $NUEVOS
    run "gpt-oss-120b"     "$BASE/gpt_oss_120b/v1"     0.12 openai/gpt-oss-120b $NUEVOS
    }
    ;;
  l40s)
    run "Mistral-7B" "$BASE/mistral7b/v5" 0.12 mistralai/Mistral-7B-Instruct-v0.3
    run "Qwen3-8B"   "$BASE/qwen3_8b/v4"  0.12 Qwen/Qwen3-8B
    ;;
  glm)
    # Solo extraccion: sin generacion, y el analisis pesado (sondas,
    # cross-fitting) se rehace despues en el portatil con --load-activations.
    run "GLM-5.3-Flash" "$BASE/glm53_flash/v1" 0.12 zai-org/GLM-5.3-Flash \
        --dtype bfloat16 --skip-intervention --skip-probes --skip-robust \
        --n-bootstrap 1000 --n-selection-splits 0
    ;;
  *) echo "maquina desconocida: $MAQUINA"; exit 1 ;;
esac

echo "=== COLA $MAQUINA/$MODO TERMINADA $(date '+%F %T') ==="
