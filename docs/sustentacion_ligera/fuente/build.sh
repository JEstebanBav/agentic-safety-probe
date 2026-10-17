#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY datos.py
node construir.js
$PY postproceso.py build/borrador.pptx build/con_ecuaciones.pptx
$PY ../../sustentacion/fuente/respaldo_ecuaciones.py build/con_ecuaciones.pptx build/con_respaldo.pptx
NOMBRE=Sustentacion_Espitia_Leiva_ligera
$PY ../../sustentacion/fuente/exportar.py build/con_respaldo.pptx "../$NOMBRE.pptx" --pdf "../$NOMBRE.pdf"
$PY miniaturas.py "../$NOMBRE.pdf" build/png
$PY guion_pdf.py build/guion.json build/png "../${NOMBRE}_guion.pdf"
