#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL="${1:?uso: ./run_modelo.sh <modelo> <carpeta-salida> [flags extra...]}"
OUT="${2:?falta la carpeta de salida}"
shift 2
mkdir -p "$OUT"
echo "modelo: $MODEL"
echo "salida: $OUT"
echo "extra : $*"
python run_experiment.py \
  --model "$MODEL" \
  --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 \
  --n-label-permutations 10 --n-bootstrap 10000 --decompose \
  --output-dir "$OUT" "$@"
