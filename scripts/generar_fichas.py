# -*- coding: utf-8 -*-
"""
Regenera la FICHA.md de cada versión y el índice results/README.md.

Cada carpeta con resultados bajo results/corridas/<modelo>/<versión>/ o
results/pruebas/<modelo>/<versión>/ recibe una ficha con el modelo, la
precisión, el corpus, los parámetros, las etapas ejecutadas y los resultados
clave (ver src/ficha.py). Lo que ningún archivo registra —el identificador
exacto del modelo en corridas antiguas, de dónde vino la carpeta, qué
complementa, dónde se cita— sale de results/registro.json.

Uso, desde la raíz del repositorio:
  python scripts/generar_fichas.py              # todas
  python scripts/generar_fichas.py --solo results/corridas/llama33_70b/v4
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ficha import collect, load_registro, render, render_index  # noqa: E402

RESULTS = ROOT / "results"

INTRO = """# Resultados

Cada versión de cada modelo vive en su propia carpeta y lleva una
**`FICHA.md`**: qué modelo era, cómo se cargó, con qué corpus y parámetros,
qué etapas corrieron y qué salió. Empieza por la ficha, no por los JSON.

```
results/
├── corridas/<modelo>/<versión>/   resultados reales
├── pruebas/<modelo>/<versión>/    corridas de prueba (4 pares): solo
│                                  comprueban que el pipeline corre; sus
│                                  estadísticas no significan nada
├── comparativa/                   figuras y tablas entre modelos
└── registro.json                  lo que ningún archivo registra
```

**Versiones.** `v1`, `v2`, `v3`… son las versiones que cita la tesis. Una
carpeta `vN_<variante>` es una corrida complementaria hecha después de la
versión `vN` (otra capa causal, una verificación, el token inicial antiguo);
su ficha dice qué complementa. El número de pares no va en el nombre: está
en la ficha y en esta tabla, porque una misma versión no siempre usó el mismo
corpus en todos los modelos.

**Una corrida nueva** va en la siguiente versión libre del modelo
(`results/corridas/<modelo>/vN/`). `run_experiment.py` escribe su ficha al
terminar; después añade su entrada en `registro.json` (etiqueta, origen,
relación, uso) y corre `python scripts/generar_fichas.py`.

`activations.npz` no se sube a git (pesa cientos de MB): existe solo en la
máquina donde se descargó. La ficha marca si está presente.
"""


def run_dirs():
    for tipo in ("corridas", "pruebas"):
        base = RESULTS / tipo
        if not base.exists():
            continue
        for modelo in sorted(p for p in base.iterdir() if p.is_dir()):
            for version in sorted(p for p in modelo.iterdir() if p.is_dir()):
                if any(version.glob("*.json")) or any(version.glob("*.log")):
                    yield version


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solo", default=None, help="Regenerar solo esta carpeta")
    args = ap.parse_args()

    registro = load_registro()
    dirs = [Path(args.solo)] if args.solo else list(run_dirs())
    rows = []
    for d in dirs:
        data = collect(d, registro)
        (d / "FICHA.md").write_text(render(data), encoding="utf-8")
        rows.append(data)
        print("ficha: results/%s" % data["carpeta"])

    sin_registro = [r["carpeta"] for r in rows if not r["registro"]]
    if not args.solo:
        (RESULTS / "README.md").write_text(render_index(rows, INTRO), encoding="utf-8")
        print("índice: results/README.md (%d versiones)" % len(rows))
        huerfanas = sorted(set(registro) - {r["carpeta"] for r in rows})
        if huerfanas:
            print("AVISO: entradas del registro sin carpeta:", ", ".join(huerfanas))
    if sin_registro:
        print("AVISO: carpetas sin entrada en results/registro.json:", ", ".join(sin_registro))


if __name__ == "__main__":
    main()
