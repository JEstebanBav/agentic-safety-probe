# -*- coding: utf-8 -*-
"""
Comprueba el presupuesto de tokens ANTES de gastar GPU.

La rutina de extracción aborta la corrida si algún estímulo alcanza el
presupuesto (4096 tokens por defecto), en lugar de recortarlo en silencio.
Es la decisión correcta —sobre un texto recortado el último token dejaría de
ser el final de la petición y la medición perdería sentido— pero el aborto
ocurre DESPUÉS de cargar el modelo, que en 70 mil millones de parámetros
cuesta varios minutos y dos GPU.

Este script renderiza exactamente los mismos estímulos que la extracción, con
el tokenizador real del modelo, y sin descargar ni un peso. Si pasa, la corrida
no va a morir por longitud.

Uso:
  python check_token_budget.py --model meta-llama/Llama-3.3-70B-Instruct
  python check_token_budget.py --model Qwen/Qwen3-8B \\
      --dataset-path data/dataset_expanded.jsonl --agency-level autonomous --toolbox 12

Sin --agency-level ni --toolbox comprueba TODAS las combinaciones, que es lo
razonable antes de una tanda de corridas. Devuelve código de salida 1 si alguna
combinación no cabe.
"""

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agency import AGENCY_LEVELS, expand_toolbox, format_agency_prompt  # noqa: E402
from src.model_loader import format_agent_prompt, format_chat_prompt  # noqa: E402


def cargar_pares(path):
    pares = {}
    for linea in io.open(path, encoding="utf-8"):
        if linea.strip():
            r = json.loads(linea)
            pares.setdefault(r["pair_id"], {})[r["variant"]] = r
    return pares


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Identificador del modelo en HuggingFace")
    ap.add_argument("--dataset-path", default=None,
                    help="Corpus a comprobar. Por defecto, todos los que existan en data/")
    ap.add_argument("--agency-level", default=None, choices=list(AGENCY_LEVELS),
                    help="Un nivel concreto. Por defecto, todos")
    ap.add_argument("--toolbox", type=int, default=None,
                    help="Un tamaño de caja concreto. Por defecto, 0/12/16/26")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Presupuesto de la rutina de extracción")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print("No se pudo cargar el tokenizador de %s" % args.model)
        print("  %s: %s" % (type(e).__name__, str(e).splitlines()[0]))
        print("  Si el repositorio está restringido, acepta las condiciones en HuggingFace")
        print("  y autentica la sesión (huggingface-cli login) antes de repetir.")
        return 2

    if args.dataset_path:
        corpus = [args.dataset_path]
    else:
        corpus = [str(p) for p in sorted((ROOT / "data").glob("dataset_*.jsonl"))]
    niveles = [args.agency_level] if args.agency_level else list(AGENCY_LEVELS)
    cajas = [args.toolbox] if args.toolbox is not None else [0, 12, 16, 26]

    def medir(texto):
        return len(tok(texto, add_special_tokens=False)["input_ids"])

    print("modelo      : %s" % args.model)
    print("presupuesto : %d tokens" % args.max_length)
    print()
    filas, peor_global, fallos = [], (0, ""), 0

    for c in corpus:
        pares = cargar_pares(c)
        nombre = Path(c).name

        peor = max(
            medir(format_chat_prompt(tok, v["chat_" + arm]["prompt"],
                                     system_message=v["chat_" + arm]["system_prompt"]))
            for v in pares.values() for arm in ("harmful", "benign")
        )
        filas.append((nombre, "conversacional", "-", peor))

        for caja in cajas:
            for nivel in niveles:
                peor_comb = 0
                for pid, v in pares.items():
                    for arm in ("harmful", "benign"):
                        e = v["agent_" + arm]
                        tools = (expand_toolbox(pid, e["tool_definitions"], caja)
                                 if caja else e["tool_definitions"])
                        n = medir(format_agency_prompt(tok, e["prompt"], tools=tools, level=nivel))
                        if n > peor_comb:
                            peor_comb = n
                            if n > peor_global[0]:
                                peor_global = (n, "%s / %s / caja %d / %s" % (nombre, nivel, caja, pid))
                filas.append((nombre, nivel, str(caja), peor_comb))

        # La condición original, sin ninguna bandera: el camino por defecto.
        peor_def = 0
        for v in pares.values():
            for arm in ("harmful", "benign"):
                e = v["agent_" + arm]
                peor_def = max(peor_def, medir(format_agent_prompt(
                    tok, e["prompt"], tools=e["tool_definitions"] or None,
                    system_message=e["system_prompt"] or None)))
        filas.append((nombre, "(por defecto)", "-", peor_def))

    ancho = max(len(f[0]) for f in filas)
    print("%-*s  %-14s %-6s %8s  %s" % (ancho, "corpus", "nivel", "caja", "peor", "veredicto"))
    print("-" * (ancho + 46))
    for nombre, nivel, caja, peor in filas:
        cabe = peor < args.max_length
        fallos += not cabe
        print("%-*s  %-14s %-6s %8d  %s" % (
            ancho, nombre, nivel, caja, peor,
            "cabe (margen %d)" % (args.max_length - peor) if cabe else "NO CABE"))

    print()
    if fallos:
        print("%d combinaciones superan el presupuesto. La corrida abortaría en ellas." % fallos)
        print("Opciones: reducir --toolbox, evitar el nivel trajectory, o subir max_length")
        print("en la rutina de extracción (el modelo admite mucho más que 4096).")
        return 1
    print("Todas las combinaciones caben. Peor caso: %d tokens (%s), margen %d."
          % (peor_global[0], peor_global[1], args.max_length - peor_global[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
