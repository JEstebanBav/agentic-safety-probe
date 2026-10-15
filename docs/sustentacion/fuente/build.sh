#!/usr/bin/env bash
# Construcción completa: datos -> borrador -> ecuaciones -> PowerPoint (pptx, pdf, png) -> guion PDF.
# Requiere Windows con PowerPoint (COM) y Microsoft Edge; no ejecuta modelos.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/c/Users/esteb/miniconda3/python.exe}
export NODE_PATH="$PWD/node_modules"
$PY datos.py
node construir.js
$PY postproceso.py build/borrador.pptx build/con_ecuaciones.pptx
$PY respaldo_ecuaciones.py build/con_ecuaciones.pptx build/con_respaldo.pptx
OUT=..; NOMBRE=Sustentacion_Espitia_Leiva
$PY exportar.py build/con_respaldo.pptx "$OUT/$NOMBRE.pptx" --pdf "$OUT/$NOMBRE.pdf" --png build/png "$@"
$PY guion_pdf.py build/guion.json build/png "$OUT/${NOMBRE}_guion.pdf"
