#!/bin/bash
# Corrida sobre el corpus v0.3 (data/v03/). Igual que run_modelo.sh, pero con
# los tres insumos que este corpus exige:
#   --dataset-path     el JSONL v0.3 (96 pares, 384 variantes)
#   --theme-groups     los 67 grupos tematicos: los folds no separan variantes
#                      cercanas entre entrenamiento y evaluacion
#   --tool-catalog-dir el catalogo neutral; solo lo leen --toolbox,
#                      --agency-level y --include-history. Sin el, la caja de
#                      herramientas reintroduce nombres como query_onion_search.
# Los flags extra van al final y prevalecen (argparse conserva el ultimo valor).
set -e
cd "$(dirname "$0")/../.."
MODEL="${1:?uso: bash scripts/runpod/run_modelo_v03.sh <modelo> <carpeta-salida> [flags extra...]}"
OUT="${2:?falta la carpeta de salida}"
shift 2
exec bash scripts/runpod/run_modelo.sh "$MODEL" "$OUT" \
  --dataset-path data/v03/dataset_v03_rerun.jsonl \
  --theme-groups data/v03/theme_groups_67.csv \
  --tool-catalog-dir data/v03/catalogos_neutrales \
  "$@"
