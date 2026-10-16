#!/bin/bash
set -e
cd "$(dirname "$0")/.."
# Local machines use a .venv; a RunPod pod installs into the system Python
# (docs/guia_runpod_llama33.md). Under `set -e` an unconditional source would
# abort the pod run before it started.
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
MODEL="${1:?uso: ./run_modelo.sh <modelo> <carpeta-salida> [flags extra...]}"
OUT="${2:?falta la carpeta de salida}"
shift 2
mkdir -p "$OUT"
echo "modelo: $MODEL"
echo "salida: $OUT"
echo "extra : $*"
# --factorial replaces --decompose: the corpus has no role_only /
# role_plus_tools variants, so --decompose only renamed the agent arm and then
# skipped its own analysis. --factorial derives the two intermediate
# conditions (system message without tools, tools without it) on the fly.
# --dataset-path comes before "$@" so a --dataset-path passed as an extra flag
# overrides it (argparse keeps the last value). Without it the run would use
# the script default, the 42-pair dataset_full.jsonl.
python run_experiment.py \
  --model "$MODEL" \
  --dataset-path data/dataset_v2.jsonl \
  --n-folds 5 --n-repeats 10 --mlp-n-repeats 1 \
  --n-label-permutations 10 --n-bootstrap 10000 --factorial \
  --output-dir "$OUT" "$@"
