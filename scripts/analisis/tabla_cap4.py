# -*- coding: utf-8 -*-
"""
Imprime las tablas 4.2 (línea base) y 4.3 (desplazamiento y control benigno)
del capítulo 4 a partir de docs/paper/evidence/metrics.json, y compara cada
celda con la cifra impresa en la tesis (docs/paper/evidence/table_transcription.json).

Uso, desde la raíz del repositorio y después de regenerar las métricas:
  python scripts/paper/build_paper.py      # lee results/ultimos_resultados, no carga modelos
  python scripts/analisis/tabla_cap4.py
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "docs" / "paper" / "evidence"
M = json.loads((EV / "metrics.json").read_text(encoding="utf-8-sig"))
T = json.loads((EV / "table_transcription.json").read_text(encoding="utf-8-sig"))


def num(x, n):
    return f"{x:.{n}f}"


def con_ic(x, ic, n):
    return f"{x:.{n}f} [{ic[0]:.{n}f}; {ic[1]:.{n}f}]"


def imprimir(titulo, cabecera, filas, clave, anchos):
    tesis = {r[0]: r[1:] for r in T[clave]["rows"]}
    print(titulo)
    print("".join(h.ljust(w) for h, w in zip(cabecera, anchos)) + "Tesis")
    total = iguales = 0
    for modelo, celdas in filas:
        esperado = tesis[modelo][-len(celdas):]
        ok = [c == e for c, e in zip(celdas, esperado)]
        total += len(ok)
        iguales += sum(ok)
        marca = "coincide" if all(ok) else "DIFIERE: " + str([e for e, o in zip(esperado, ok) if not o])
        print(modelo.ljust(anchos[0]) + "".join(c.ljust(w) for c, w in zip(celdas, anchos[1:])) + marca)
    print(f"{iguales}/{total} celdas iguales a la tesis\n")


filas = []
for m in M.values():
    d = m["deteccion"]
    filas.append((m["modelo"], [str(m["capa_referencia"]),
                                con_ic(d["auroc_chat"], d["auroc_chat_ic"], 4),
                                con_ic(d["auroc_agente"], d["auroc_agente_ic"], 4),
                                con_ic(d["brecha"], d["brecha_ic"], 4)]))
imprimir("Tabla 4.2  AUROC fuera de muestra de la dirección conversacional (5 folds por pares, 96 pares)",
         ["Modelo", "Capa", "AUROC chat", "AUROC agente", "Diferencia"], filas, "tab:linea_base", [15, 6, 26, 26, 26])

filas = []
for m in M.values():
    s = m["desplazamiento"]
    filas.append((m["modelo"], [con_ic(s["dp_h"], s["dp_h_ic"], 2),
                                con_ic(s["dp_b"], s["dp_b_ic"], 2),
                                con_ic(s["interaccion"], s["interaccion_ic"], 2),
                                f"{s['pares_interaccion_positiva']}/{m['n_pares']}"]))
imprimir("Tabla 4.3  Desplazamiento fuera de muestra en la capa de referencia (IC bootstrap 95 %)",
         ["Modelo", "Δp_h (dañino)", "Δp_b (benigno)", "Δ²p (interacción)", "Pares Δ²p>0"], filas, "tab:gap", [15, 26, 24, 26, 13])
