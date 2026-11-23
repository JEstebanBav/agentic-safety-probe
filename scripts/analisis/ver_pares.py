# -*- coding: utf-8 -*-
"""
Muestra pares del corpus (data/dataset_v2.jsonl) con sus cuatro variantes.

Uso, desde la raíz del repositorio:
  python scripts/analisis/ver_pares.py                       # resumen del corpus
  python scripts/analisis/ver_pares.py drugs_01 fraud_02 disinfo_08
"""
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
filas = [json.loads(l) for l in (ROOT / "data" / "dataset_v2.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
pares = {}
for r in filas:
    pares.setdefault(r["pair_id"], {})[r["variant"]] = r

ESTRATO = {"explicit": "explícito", "contextual": "contextual", "framed": "apariencia legítima"}

print(f"{len(filas)} entradas = {len(pares)} pares x 4 variantes")
print("Estratos:", dict(Counter(ESTRATO[v['chat_harmful']['subtlety']] for v in pares.values())))
print("Fuentes:", dict(Counter(v['chat_harmful']['source'] for v in pares.values())))
mismo_texto = sum(v["chat_harmful"]["prompt"] == v["agent_harmful"]["prompt"]
                  and v["chat_benign"]["prompt"] == v["agent_benign"]["prompt"] for v in pares.values())
mismas_herr = sum(v["agent_harmful"]["tools"] == v["agent_benign"]["tools"] for v in pares.values())
print(f"Texto idéntico chat/agente: {mismo_texto}/{len(pares)} · mismas herramientas dañino/benigno: {mismas_herr}/{len(pares)}")

for pid in sys.argv[1:]:
    v = pares.get(pid)
    if v is None:
        print(f"\n{pid}: no existe")
        continue
    h, b, ah = v["chat_harmful"], v["chat_benign"], v["agent_harmful"]
    print(f"\n=== {pid} · {h['category']} · {ESTRATO[h['subtlety']]} · Jaccard {h['lexical_jaccard']}")
    print(f"DAÑINO : {h['prompt']}")
    print(f"BENIGNO: {b['prompt']}")
    print(f"Agente : mismas dos frases + sistema de agente + herramientas {ah['tools']}")
