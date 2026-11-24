# -*- coding: utf-8 -*-
"""
Rehace el análisis robusto de una corrida desde su activations.npz, sin GPU.

Las ocho corridas del corpus v0.3 perdieron la selección de capa en muestra
dividida: con los grupos temáticos registrados, la mitad de inferencia (48
pares) se particionaba contra el mapa de 96 y la etapa abortaba con un aviso.
El error está corregido en src/robust_estimation.py (pair_clusters_subset).

Este script vuelve a correr run_robust_analysis con los parámetros de la
corrida original y solo AÑADE la selección de capa (robust.layer_selection)
a results.json y robust_results.json: lo que el pod calculó se queda como
está. Antes de escribir comprueba que el recálculo reproduce lo del pod:
todo lo que no usa regresión logística debe coincidir (diferencia relativa
< 1e-6); lo que la usa (w_agent, cosenos logísticos) puede diferir un poco
entre máquinas por las versiones de numpy/scikit-learn, y solo se acota.
La selección de capa usa diferencias de medias y permutaciones, no regresión
logística. Si algo no cuadra, no toca la carpeta. La provenance queda en
results.json["robust_recomputed"] y la salida en rehacer_robusto.log.

Uso, desde la raíz del repositorio:
  python scripts/analisis/rehacer_analisis_robusto.py results/corridas/gemma4_31b/v1 [...]
"""

import argparse
import contextlib
import datetime
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
from data.loader import load_theme_groups  # noqa: E402

MOTIVO = ("La selección de capa en muestra dividida falló en el pod (mapa de "
          "grupos temáticos de 96 pares aplicado a la mitad de inferencia de 48). "
          "Se recalculó desde activations.npz con el error corregido y solo se "
          "añadió robust.layer_selection; per_layer y rotation son los del pod.")

# per_layer fields that come from a logistic regression (w_agent and the
# monitor built on it). Their solver output shifts slightly across
# numpy/scikit-learn builds; everything else is exact arithmetic.
LOGISTIC = {"auroc_w_agent_in_sample", "auroc_w_agent_oof", "auroc_w_agent_oof_sd",
            "split_half_reliability", "attenuation_factor", "monitor_f1_in_sample",
            "monitor_f1_oof", "monitor_precision_oof", "monitor_recall_oof",
            "monitor_fpr_oof"}
TOL_LOGISTIC = 0.05


def _is_logistic(path: str) -> bool:
    parts = path.split(".")
    return "logistic" in parts or parts[-1] in LOGISTIC


def _diffs(a, b, path, out):
    """Collect (relative difference, path) for every number; raises if the shapes differ."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            raise ValueError(f"{path}: las claves no coinciden")
        for k in a:
            _diffs(a[k], b[k], f"{path}.{k}", out)
        return out
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            raise ValueError(f"{path}: las longitudes no coinciden")
        for i, (x, y) in enumerate(zip(a, b)):
            _diffs(x, y, f"{path}[{i}]", out)
        return out
    numeric = (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
    if numeric(a) and numeric(b):
        if math.isnan(a) or math.isnan(b):
            d = 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
        else:
            d = abs(a - b) / max(abs(a), abs(b), 1e-12)
        out.append((d, path))
    elif a != b and not _is_logistic(path):
        raise ValueError(f"{path}: {a!r} != {b!r}")
    return out


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def rehacer(run_dir: Path, tol: float) -> None:
    res_path = run_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))
    cfg = results["config"]
    old = results.get("robust") or {}
    if "layer_selection" in old:
        print(f"{run_dir}: ya tiene la selección de capa; no se toca.")
        return

    activations, pair_order, meta = rx.load_activations(str(run_dir / "activations.npz"))
    layers = [int(l) for l in meta["layers"]]
    if layers != sorted(int(l) for l in old.get("per_layer", {})):
        raise SystemExit(f"{run_dir}: las capas de activations.npz no son las del análisis original.")

    groups = None
    if cfg.get("theme_groups"):
        groups = load_theme_groups(str(ROOT / cfg["theme_groups"]))
        missing = set(next(iter(pair_order.values()))) - set(groups)
        if missing:
            raise SystemExit(f"{run_dir}: {len(missing)} pares sin grupo temático.")
    partition = rx._register_theme_groups(groups, pair_order)

    t0 = time.time()
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(buf):
            new = rx.run_robust_analysis(
                activations=activations, layers=layers, output_dir=tmp,
                n_folds=cfg["n_folds"], seed=cfg["seed"],
                n_selection_splits=cfg["n_selection_splits"])
        log = buf.getvalue()
        if "layer selection failed" in log or "layer_selection" not in new:
            raise SystemExit(f"{run_dir}: la selección de capa sigue fallando; no se escribe nada.")
        new_json = rx._jsonify(new)
        diffs = []
        for k in old:
            _diffs(old[k], new_json[k], k, diffs)
        exact = max((d for d, p in diffs if not _is_logistic(p)), default=0.0)
        logit = max((d for d, p in diffs if _is_logistic(p)), default=0.0)
        if exact > tol or logit > TOL_LOGISTIC:
            worst = max(diffs)
            raise SystemExit(f"{run_dir}: el recálculo no reproduce lo del pod (exacto "
                             f"{exact:.1e}, logística {logit:.1e}; peor: {worst[1]}); "
                             "no se escribe nada.")
        robust_file = json.loads((Path(tmp) / "robust_results.json").read_text(encoding="utf-8"))

    sel = new_json["layer_selection"]
    results["robust"]["layer_selection"] = sel
    results["robust_recomputed"] = {
        "fecha": datetime.date.today().isoformat(),
        "codigo": _commit(),
        "motivo": MOTIVO,
        "particion": partition,
        "reproduccion": {
            "diferencia_relativa_maxima_sin_logistica": exact,
            "diferencia_relativa_maxima_logistica": logit,
            "nota": "per_layer y rotation recalculados aquí y comparados con los del "
                    "pod; se conservan los del pod",
        },
        "capa_seleccionada": sel["selected_layer"],
        "capa_de_referencia_max_delta_p": results.get("best_layer"),
    }
    pod_file_path = run_dir / "robust_results.json"
    pod_file = json.loads(pod_file_path.read_text(encoding="utf-8"))
    pod_file["layer_selection"] = robust_file["layer_selection"]
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(pod_file_path, "w") as f:
        json.dump(pod_file, f, indent=2, default=str)
    (run_dir / "rehacer_robusto.log").write_text(log, encoding="utf-8")

    causal = json.loads((run_dir / "causal_results.json").read_text(encoding="utf-8"))
    causal_layers = sorted(int(l) for l in causal.get("causal", {}))
    print(f"{run_dir}: capa seleccionada {sel['selected_layer']} | máximo ΔP "
          f"{results.get('best_layer')} | test causal en {causal_layers} | "
          f"reproducción: exacto {exact:.0e}, logística {logit:.0e} | "
          f"{time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("carpetas", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Diferencia relativa máxima aceptada al reproducir lo que ya estaba")
    args = ap.parse_args()
    for d in args.carpetas:
        rehacer(d, args.tol)


if __name__ == "__main__":
    main()
