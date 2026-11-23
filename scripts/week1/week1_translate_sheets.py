"""Attach the reviewed Spanish translations and prepare separate Excel sheets.

python scripts/week1/week1_translate_sheets.py
Existing CSV labels/notes are preserved. Existing XLSX workbooks are never
overwritten, because they may already contain human decisions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import xlsxwriter

from week1_audit import write_csv
from week1_excel import HEADERS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/week1"
FIELDS = ["annotation_id", "prompt", "prompt_es", "stratum", "intent_clear", "evidence", "notes"]
DISPLAY = {value: key for key, value in HEADERS.items()}


def write_workbook(path, rows, version):
    workbook = xlsxwriter.Workbook(str(path), {"strings_to_urls": False, "strings_to_formulas": False})
    workbook.set_properties({"title": "Solicitudes bilingües para anotación independiente",
                             "comments": f"Traducción {version}; no contiene etiquetas previas ni sugeridas."})
    sheet = workbook.add_worksheet("Anotacion")
    guide = workbook.add_worksheet("Guía")
    header = workbook.add_format({"bold": True, "bg_color": "#244861", "font_color": "white",
                                  "text_wrap": True, "valign": "vcenter", "border": 1})
    original = workbook.add_format({"text_wrap": True, "valign": "top", "bg_color": "#F2F5F7", "locked": True})
    translation = workbook.add_format({"text_wrap": True, "valign": "top", "locked": True})
    answer = workbook.add_format({"text_wrap": True, "valign": "top", "bg_color": "#FFF4CC", "locked": False})
    identifier = workbook.add_format({"bold": True, "valign": "top", "locked": True})
    sheet.freeze_panes(1, 1)
    sheet.set_zoom(85)
    sheet.set_row(0, 42)
    for index, field in enumerate(FIELDS):
        sheet.write_string(0, index, DISPLAY[field], header)
    sheet.set_column(0, 0, 9)
    sheet.set_column(1, 2, 62)
    sheet.set_column(3, 3, 16)
    sheet.set_column(4, 4, 19)
    sheet.set_column(5, 6, 35)
    for index, row in enumerate(rows, 1):
        line_count = max(sum(max(1, math.ceil(len(line) / 58)) for line in text.split("\n"))
                         for text in (row["prompt"], row["prompt_es"]))
        sheet.set_row(index, min(409, max(120, line_count * 16 + 12)))
        for column, field in enumerate(FIELDS):
            fmt = identifier if column == 0 else original if column == 1 else translation if column == 2 else answer
            sheet.write_string(index, column, row.get(field, ""), fmt)
    sheet.autofilter(0, 0, len(rows), len(FIELDS) - 1)
    sheet.data_validation(1, 3, len(rows), 3, {
        "validate": "list", "source": ["explicit", "contextual", "framed"],
        "input_title": "Tu decisión independiente", "input_message": "Aplica la guía acordada; no compares con otra persona todavía.",
        "error_title": "Etiqueta no válida", "error_message": "Elige explicit, contextual o framed.", "error_type": "stop"})
    sheet.data_validation(1, 4, len(rows), 4, {
        "validate": "list", "source": ["yes", "no", "uncertain"],
        "input_title": "Claridad de la intención", "input_message": "yes: sí; no: no; uncertain: falta contexto.",
        "error_title": "Valor no válido", "error_message": "Elige yes, no o uncertain.", "error_type": "stop"})
    sheet.write_comment(0, 3, "explicit = explícito; contextual = contextual; framed = con encuadre legitimador. Revisa la pestaña Guía antes de empezar.")
    sheet.write_comment(0, 5, "Copia una frase breve que respalde tu decisión. Puede estar en inglés o español.")
    sheet.protect("", {"autofilter": True, "select_locked_cells": True, "select_unlocked_cells": True})
    guide.set_column(0, 0, 105)
    guide_style = workbook.add_format({"text_wrap": True, "valign": "top", "font_size": 12})
    instructions = [
        "Cómo completar esta hoja",
        "Cada persona completa su propio archivo A o B. Los dos archivos contienen las mismas 96 solicitudes, en distinto orden. No se repartan las filas: cada persona anota las 96.",
        "Lee las reglas de docs/week1/codebook.md y acuerden la guía antes de comenzar. Después trabajen de forma independiente, sin consultar las etiquetas anteriores, el informe de auditoría ni la hoja de la otra persona.",
        "Las columnas blancas y grises conservan el original y la traducción. Completa las cuatro columnas amarillas. Los textos están protegidos sin contraseña para evitar cambios accidentales.",
        "Estrato: elige explicit, contextual o framed. Son los códigos que usa el cálculo de Kappa; no los traduzcas al escribirlos.",
        "Regla de prioridad propuesta en la guía: framed si una justificación legitimadora presenta la acción como parte de un fin profesional, educativo, investigador o de ayuda; si no hay ese encuadre, explicit cuando el abuso se expresa directamente; en otro caso, contextual cuando se infiere del contexto. Acordar esta regla antes de anotar.",
        "¿El daño está claro?: yes = sí; no = no; uncertain = falta información para decidir. Esta columna es distinta del estrato. Una actividad presentada como autorizada no se vuelve dañina por llevar una etiqueta de encuadre.",
        "Fragmento de evidencia: copia la frase que respalda tu decisión, en inglés o español. Notas: explica ambigüedades o dudas. Si no puedes asignar estrato, déjalo vacío y registra la duda; la anotación seguirá pendiente.",
        "La traducción es una ayuda de lectura asistida por IA, no una clasificación. Conserva las ambigüedades de permisos, veracidad e intención del original. No modifica el dataset experimental. Si detectas una discrepancia, registra la duda y acuerden una corrección de traducción para ambos antes de cerrar la ronda.",
        "Se conservaron nombres, direcciones, correos, enlaces, rutas, cantidades y las fechas tal como aparecen. 8/21 y 8/23 se mantienen en su formato original. Los títulos de archivos u obras y los identificadores técnicos pueden seguir en inglés.",
        "Glosario de lectura: abstract = resumen de un artículo; benchmark = prueba comparativa; credential dumping = volcado de credenciales; endpoint = punto de conexión o destino de un servicio; hashes = valores hash; scraping = extracción automatizada de datos; phishing = suplantación mediante mensajes; dark web = parte de la web accesible mediante redes específicas. Estas aclaraciones no adjudican la intención de un caso.",
        "Al terminar, guarda este mismo archivo. Calculamos Kappa con las decisiones iniciales de A y B antes de discutir desacuerdos. Luego registramos el consenso en un archivo separado. Objetivo del mentor: Kappa ≥ 0,6.",
        f"Versión de traducción: {version}. Usa la misma versión para ambos anotadores. La decisión de cada fila está vinculada a su ID, aunque los órdenes sean distintos.",
    ]
    for index, text in enumerate(instructions):
        guide.write_string(index, 0, text, header if index == 0 else guide_style)
        guide.set_row(index, 32 if index == 0 else 78)
    workbook.close()


def main():
    corpus_hash = hashlib.sha256((ROOT / "data/dataset_v2.jsonl").read_bytes()).hexdigest()
    document = json.loads((DATA / "strata_translations_es.json").read_text(encoding="utf-8"))
    if document["source_corpus_sha256"] != corpus_hash:
        raise ValueError("Translations belong to a different corpus version")
    translations = document["translations"]
    plans = []
    for annotator in ("A", "B"):
        path = DATA / f"strata_annotator_{annotator}_BLIND.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            rows = list(reader)
        if len(rows) != 96 or len({r["annotation_id"] for r in rows}) != 96 or {r["annotation_id"] for r in rows} != set(translations):
            raise ValueError(f"{path.name}: translations do not cover exactly the annotation IDs")
        extra = [field for field in existing_fields if field not in FIELDS]
        enriched = [dict(row, prompt_es=translations[row["annotation_id"]]) for row in rows]
        plans.append((annotator, path, enriched, FIELDS + extra))
    texts_a = {r["annotation_id"]: r["prompt"] for r in plans[0][2]}
    texts_b = {r["annotation_id"]: r["prompt"] for r in plans[1][2]}
    if texts_a != texts_b:
        raise ValueError("Annotators do not have identical original prompts")
    manifest = {
        "version": document["version"], "source_corpus_sha256": corpus_hash,
        "translation_file_sha256": hashlib.sha256((DATA / "strata_translations_es.json").read_bytes()).hexdigest(),
        "n": 96, "records": [
            {"annotation_id": identifier,
             "original_sha256": hashlib.sha256(texts_a[identifier].encode()).hexdigest(),
             "translation_sha256": hashlib.sha256(translations[identifier].encode()).hexdigest()}
            for identifier in sorted(translations)],
    }
    manifest_path = DATA / "translation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Translation version has changed; preserve the old round and version new workbooks first")
    for annotator, path, rows, fields in plans:
        write_csv(path, rows, fields)
        excel_path = DATA / f"strata_annotator_{annotator}_BLIND_ES.xlsx"
        if excel_path.exists():
            print(f"Preserved existing workbook: {excel_path.relative_to(ROOT)}")
        else:
            write_workbook(excel_path, rows, document["version"])
            print(f"Created: {excel_path.relative_to(ROOT)} ({len(rows)} requests)")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Original corpus unchanged; CSV IDs, row order and all human annotations preserved.")


if __name__ == "__main__":
    main()
