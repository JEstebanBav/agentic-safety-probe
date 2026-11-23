"""Summarise the frozen first human round and prepare a separate consensus workbook.

python scripts/week1/week1_review_annotations.py
Reads raw XLSX snapshots; never changes either annotator's decisions. The
consensus workbook is created only once, preserving subsequent human work.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import xlsxwriter

from week1_agreement import STRATA, agreement, read_sheet
from week1_audit import write_csv
from week1_excel import HEADERS, read_xlsx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/week1"
DOCS = ROOT / "docs/week1"
ROUND = DATA / "round1"
ORDER = ["explicit", "contextual", "framed"]
INTENT = ["yes", "no", "uncertain"]


def workbook(path, comparisons, notes, summary):
    if path.exists():
        print(f"Preserved existing consensus workbook: {path.relative_to(ROOT)}")
        return
    book = xlsxwriter.Workbook(str(path), {"strings_to_urls": False, "strings_to_formulas": False})
    book.set_properties({"title": "Revisión conjunta — primera ronda de anotación",
                         "comments": "Human initial labels are frozen. Empty consensus fields require author decisions."})
    header = book.add_format({"bold": True, "bg_color": "#244861", "font_color": "white", "text_wrap": True, "valign": "top"})
    text = book.add_format({"text_wrap": True, "valign": "top", "locked": True})
    gray = book.add_format({"text_wrap": True, "valign": "top", "bg_color": "#F2F5F7", "locked": True})
    editable = book.add_format({"text_wrap": True, "valign": "top", "bg_color": "#FFF4CC", "locked": False})
    number = book.add_format({"num_format": "0.000"})
    percent = book.add_format({"num_format": "0.0%"})
    info = book.add_worksheet("Resumen")
    info.set_column(0, 0, 44)
    info.set_column(1, 1, 105)
    info.write_row(0, 0, ["Resultado de ronda inicial", "Valor / explicación"], header)
    items = [
        ("Solicitudes por persona", 96),
        ("Coincidencias de estrato", summary["strata"]["matches"]),
        ("Desacuerdos de estrato", len(summary["strata"]["disagreements"])),
        ("Acuerdo observado de estrato", summary["strata"]["agreement"]),
        ("Kappa de estrato", summary["strata"]["kappa"]),
        ("Meta del mentor", 0.6),
        ("Estado", "Por debajo de la meta; revisar definiciones y registrar consenso separado."),
        ("Desacuerdos en claridad del daño", len(summary["intent_clear"]["disagreements"])),
        ("Notas humanas", len(notes)),
        ("Paso 1", "Leer las diferencias y notas en Comparacion y Comentarios. Las respuestas a las notas son apoyo de IA, no consenso humano."),
        ("Paso 2", "Discutir qué evidencia respalda cada decisión según una regla común. Separar estrato, claridad del daño, duplicidad, realismo y cambios al dataset."),
        ("Paso 3", "Completar las 96 decisiones finales y sus justificaciones en las columnas amarillas de Anotacion. No editar los archivos A/B originales."),
        ("Paso 4", "Guardar este libro. week1_agreement.py puede leerlo como consenso separado y conservar el Kappa inicial de A/B."),
        ("Kappa y consenso", "Un Kappa calculado después de ponerse de acuerdo no mide acuerdo independiente. Si se repite una ronda, conservar la primera y documentar el nuevo protocolo."),
        ("Datos experimentales", "No se modifica el dataset ni se incorporan etiquetas por IA. Los cambios de contenido se versionan y requieren nuevas activaciones de los casos afectados."),
    ]
    for index, (label, value) in enumerate(items, 1):
        info.write_string(index, 0, label, gray)
        info.set_row(index, 60 if isinstance(value, str) else 24)
        if isinstance(value, (int, float)):
            info.write_number(index, 1, value, percent if "Acuerdo observado" in label else number if isinstance(value, float) else None)
        else:
            info.write_string(index, 1, value, text)

    annotation = book.add_worksheet("Anotacion")
    fields = ["annotation_id", "prompt", "prompt_es", "stratum", "intent_clear", "evidence", "notes"]
    display = {value: key for key, value in HEADERS.items()}
    annotation.write_row(0, 0, [display[field] for field in fields], header)
    annotation.freeze_panes(1, 1)
    annotation.set_zoom(85)
    annotation.set_column(0, 0, 9)
    annotation.set_column(1, 2, 60)
    annotation.set_column(3, 4, 20)
    annotation.set_column(5, 6, 45)
    annotation.set_row(0, 36)
    for index, row in enumerate(comparisons, 1):
        lines = max(sum(max(1, math.ceil(len(line) / 56)) for line in row[field].split("\n")) for field in ("prompt", "prompt_es"))
        annotation.set_row(index, min(409, max(120, lines * 16 + 12)))
        for column, field in enumerate(fields):
            value = row.get(field, "") if column < 3 else ""
            annotation.write_string(index, column, value, gray if column < 3 else editable)
    annotation.autofilter(0, 0, len(comparisons), len(fields) - 1)
    annotation.data_validation(1, 3, len(comparisons), 3, {"validate": "list", "source": ORDER,
                               "error_message": "Elige explicit, contextual o framed.", "error_type": "stop"})
    annotation.data_validation(1, 4, len(comparisons), 4, {"validate": "list", "source": INTENT,
                               "error_message": "Elige yes, no o uncertain.", "error_type": "stop"})
    annotation.write_comment(0, 3, "Consenso humano después de preservar las dos decisiones iniciales. Todas las filas quedan en blanco hasta adjudicarlas.")
    annotation.write_comment(0, 6, "Justifica el acuerdo final y registra decisiones de revisión del dataset por separado.")
    annotation.protect("", {"autofilter": True, "select_locked_cells": True, "select_unlocked_cells": True})

    comparison = book.add_worksheet("Comparacion")
    columns = [
        ("annotation_id", "ID", 9), ("label_A", "Estrato A", 16), ("label_B", "Estrato B", 16),
        ("intent_A", "Daño claro A", 16), ("intent_B", "Daño claro B", 16),
        ("review_required", "Revisar", 12), ("review_reasons", "Motivos de revisión", 35),
        ("prompt_es", "Solicitud en español", 75), ("notes_A", "Notas A", 55), ("notes_B", "Notas B", 55),
        ("evidence_A", "Evidencia A", 32), ("evidence_B", "Evidencia B", 32),
        ("theme_group_proposed", "Grupo propuesto", 35),
    ]
    comparison.freeze_panes(1, 1)
    comparison.set_zoom(80)
    for column, (_, title, width) in enumerate(columns):
        comparison.write_string(0, column, title, header)
        comparison.set_column(column, column, width)
    comparison.set_row(0, 34)
    for index, row in enumerate(comparisons, 1):
        comparison.set_row(index, 130, None, {"hidden": not row["review_required"]})
        for column, (key, _, _) in enumerate(columns):
            value = ("sí" if row[key] else "no") if key == "review_required" else row[key]
            comparison.write_string(index, column, str(value), text)
    comparison.autofilter(0, 0, len(comparisons), len(columns) - 1)
    comparison.filter_column(5, 'x == "sí"')

    comments = book.add_worksheet("Comentarios")
    comment_columns = [
        ("annotation_id", "ID", 9), ("annotator", "Persona", 10), ("human_note", "Nota humana original", 65),
        ("themes", "Tema", 30), ("response", "Observación de IA para discutir; no es consenso", 100),
        ("related_ids", "IDs relacionados", 30),
    ]
    comments.freeze_panes(1, 2)
    comments.set_zoom(80)
    for column, (_, title, width) in enumerate(comment_columns):
        comments.write_string(0, column, title, header)
        comments.set_column(column, column, width)
    for index, row in enumerate(notes, 1):
        comments.set_row(index, 150)
        for column, (key, _, _) in enumerate(comment_columns):
            value = row[key]
            comments.write_string(index, column, "; ".join(value) if isinstance(value, list) else value, text)
    comments.autofilter(0, 0, len(notes), len(comment_columns) - 1)
    book.close()
    print(f"Created separate consensus workbook: {path.relative_to(ROOT)}")


def main():
    manifest = json.loads((ROUND / "manifest.json").read_text(encoding="utf-8"))
    by_annotator = {}
    for item in manifest["files"]:
        path = ROOT / item["snapshot"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Raw annotation snapshot has changed")
        rows = read_xlsx(path)
        by_annotator[item["annotator"]] = {r["annotation_id"]: r for r in rows}
    a, b = by_annotator["A"], by_annotator["B"]
    if set(a) != set(b) or len(a) != 96:
        raise ValueError("Round1 does not contain the same 96 annotation IDs")
    for who in ("A", "B"):
        path = ROOT / next(item["snapshot"] for item in manifest["files"] if item["annotator"] == who)
        read_sheet(path, set(a), "stratum", STRATA)
        read_sheet(path, set(a), "intent_clear", set(INTENT))
    if not (ROUND / "codebook_at_review.md").exists():
        shutil.copy2(DOCS / "codebook.md", ROUND / "codebook_at_review.md")
    metrics = {}
    for field, order in (("stratum", ORDER), ("intent_clear", INTENT)):
        result = agreement({k: a[k][field] for k in a}, {k: b[k][field] for k in b})
        if result["status"] != "complete":
            raise ValueError("This review requires a complete initial round; no partial scoring")
        result["matches"] = sum(a[k][field] == b[k][field] for k in a)
        result["counts_A"] = dict(Counter(a[k][field] for k in a))
        result["counts_B"] = dict(Counter(b[k][field] for k in b))
        result["confusion_matrix_rows_A_columns_B"] = {
            left: {right: sum(a[k][field] == left and b[k][field] == right for k in a) for right in order}
            for left in order}
        metrics["strata" if field == "stratum" else "intent_clear"] = result
    metrics["strata"].update(target_kappa=.6, target_met=metrics["strata"]["kappa"] >= .6)
    with (DATA / "annotation_key_PRIVATE.csv").open(encoding="utf-8-sig", newline="") as f:
        key = {r["annotation_id"]: r["pair_id"] for r in csv.DictReader(f)}
    with (DATA / "theme_groups_proposed.csv").open(encoding="utf-8-sig", newline="") as f:
        groups = {r["pair_id"]: r["theme_group_id"] for r in csv.DictReader(f)}
    reviews = json.loads((DATA / "round1_notes_review.json").read_text(encoding="utf-8"))["reviews"]
    review_by_key = {(r["annotator"], r["annotation_id"]): r for r in reviews}
    actual_notes = {(who, identifier) for who, rows in by_annotator.items()
                    for identifier, row in rows.items() if row["notes"].strip()}
    if set(review_by_key) != actual_notes:
        raise ValueError("Comment responses do not cover exactly the authors' supplied notes")
    notes = [dict(review_by_key[(who, identifier)], human_note=by_annotator[who][identifier]["notes"])
             for who, identifier in sorted(actual_notes)]
    comparisons = []
    for identifier in sorted(a):
        left, right = a[identifier], b[identifier]
        if (left["prompt"], left["prompt_es"]) != (right["prompt"], right["prompt_es"]):
            raise ValueError("Source text differs between raters")
        reasons = []
        if left["stratum"] != right["stratum"]:
            reasons.append("desacuerdo_estrato")
        if left["intent_clear"] != right["intent_clear"]:
            reasons.append("desacuerdo_claridad")
        if left["intent_clear"] != "yes" or right["intent_clear"] != "yes":
            reasons.append("daño_no_confirmado_por_ambos")
        if left["notes"].strip() or right["notes"].strip():
            reasons.append("comentario_humano")
        comparisons.append({
            "annotation_id": identifier, "pair_id": key[identifier],
            "theme_group_proposed": groups[key[identifier]],
            "label_A": left["stratum"], "label_B": right["stratum"],
            "intent_A": left["intent_clear"], "intent_B": right["intent_clear"],
            "evidence_A": left["evidence"], "evidence_B": right["evidence"],
            "notes_A": left["notes"], "notes_B": right["notes"],
            "review_required": bool(reasons), "review_reasons": "; ".join(reasons),
            "prompt": left["prompt"], "prompt_es": left["prompt_es"],
        })
    summary = {
        "round": "round1_initial_independent_labels", "source_manifest": "data/week1/round1/manifest.json",
        **metrics, "human_notes": len(notes), "items_with_notes": len({r["annotation_id"] for r in notes}),
        "items_to_discuss": sum(row["review_required"] for row in comparisons),
        "evidence_cells_nonempty": {who: sum(bool(r["evidence"].strip()) for r in rows.values())
                                    for who, rows in by_annotator.items()},
        "consensus_status_at_initial_review": "not_adjudicated", "original_labels_modified": False,
        "classifier_validation": "pending; these 96 request labels are not the 100 generated outputs",
    }
    adjudication_path = DATA / "strata_adjudication.json"
    if adjudication_path.exists():
        record = json.loads(adjudication_path.read_text(encoding="utf-8"))
        summary["subsequent_adjudication"] = {
            "decision_date": record["decision_date"], "method": record["method"],
            "report": "docs/week1/strata_adjudication.md",
            "original_kappa_unchanged": True,
        }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "round1_agreement.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(DATA / "round1_comparison.csv", comparisons, list(comparisons[0]))
    write_csv(DATA / "round1_comment_responses.csv", [{
        "annotation_id": r["annotation_id"], "annotator": r["annotator"], "human_note": r["human_note"],
        "themes": "; ".join(r["themes"]), "response": r["response"],
        "related_ids": "; ".join(r["related_ids"]),
    } for r in notes], ["annotation_id", "annotator", "human_note", "themes", "response", "related_ids"])
    workbook(DATA / "strata_consensus_round1.xlsx", comparisons, notes, summary)

    primary = metrics["strata"]
    clarity = metrics["intent_clear"]
    lines = ["# Resultado de la primera ronda de anotación", "",
             "Se conservaron copias y hashes de las hojas entregadas por A y B antes de la adjudicación. Cada una tiene 96 etiquetas de estrato y 96 respuestas de claridad del daño. Los textos ingleses y sus traducciones coinciden con el manifiesto de traducción.", "",
             f"**Kappa de estrato: {primary['kappa']:.6f}. Acuerdo observado: {primary['matches']}/96 ({primary['agreement']:.2%}). Hay {len(primary['disagreements'])} desacuerdos.** La meta del mentor, κ ≥ 0,6, no se alcanza en esta ronda.", "",
             "Este resultado mide las decisiones iniciales tal como se entregaron. No se corrigió ninguna etiqueta para aumentar el acuerdo. El Kappa no es el porcentaje simple de coincidencias ni una medida de exactitud frente a una etiqueta verdadera.", "",
             "## Matriz de estratos", "",
             "Filas: A; columnas: B.", "",
             "| A / B | explicit | contextual | framed | Total A |",
             "|---|---:|---:|---:|---:|"]
    matrix = primary["confusion_matrix_rows_A_columns_B"]
    for label in ORDER:
        lines.append(f"| {label} | {matrix[label]['explicit']} | {matrix[label]['contextual']} | {matrix[label]['framed']} | {primary['counts_A'].get(label, 0)} |")
    lines.append(f"| Total B | {primary['counts_B'].get('explicit', 0)} | {primary['counts_B'].get('contextual', 0)} | {primary['counts_B'].get('framed', 0)} | 96 |")
    lines += ["", "Los dos cruces más frecuentes son A=contextual/B=explicit (23 casos) y A=framed/B=contextual (14 casos). Esto señala diferencias de aplicación de las definiciones; no establece que un anotador sea la referencia correcta.", "",
              "## Claridad del daño y evidencia", "",
              f"La columna auxiliar intent_clear coincide en {clarity['matches']}/96 ({clarity['agreement']:.2%}), con κ={clarity['kappa']:.6f} y {len(clarity['disagreements'])} desacuerdos. Es una dimensión distinta de los estratos y no tiene aquí una meta del mentor fijada por separado.", "",
              "A marcó yes=86, no=2, uncertain=8; B marcó yes=57, no=31, uncertain=8. Algunas notas mezclan claridad del daño con realismo, repetición de temas o viabilidad de herramientas. Esas observaciones son útiles, pero conviene registrarlas por separado antes de adjudicar el ground truth.", "",
              "Las celdas de evidencia están vacías en A; en B solo hay una no vacía, con el texto 'c'. Esto no impide calcular Kappa, pero no documenta el razonamiento. Durante el consenso, citar el fragmento que respalda cada decisión, especialmente en los desacuerdos.", "",
              "## Qué hacer a continuación", "",
              "1. Revisar la regla de framed: exige un encuadre que legitime la acción cuestionada. Un tema profesional, un archivo académico o un FAQ por sí solos no lo establecen. La presencia del encuadre tampoco demuestra que la actividad sea dañina.",
              "2. Separar explicitud de probabilidad de rechazo, naturalidad y ejecutabilidad. Clasificar la versión escrita; cualquier reformulación se registra como otra versión.",
              "3. Discutir los 53 desacuerdos de estrato y las dudas de daño; registrar decisiones y evidencia en el Excel de consenso separado. Los 96 casos deben tener una etiqueta final acordada, incluidos los que ya coincidían.",
              "4. Conservar esta primera ronda. El consenso no eleva retrospectivamente el Kappa independiente. Si se necesita una segunda ronda para evaluar un codebook revisado, documentarla como tal y conservar el resultado inicial; no llamar ciega a una revisión informada por la discusión previa.",
              "5. Adoptar cambios del dataset solo después de adjudicar: las etiquetas y activaciones de textos originales no se transfieren automáticamente a textos cambiados.", "",
              f"El Excel [strata_consensus_round1.xlsx](../../data/week1/strata_consensus_round1.xlsx) contiene las cuatro hojas Resumen, Anotacion, Comparacion y Comentarios. Comparacion muestra inicialmente {summary['items_to_discuss']} casos con diferencias, daño no confirmado por ambos o notas; se puede quitar el filtro para ver los 96. El archivo se creó con etiquetas de consenso vacías; las decisiones posteriores se registran por separado.", "",
              "El detalle de cada comentario está en [round1_notes_review.md](round1_notes_review.md). Las observaciones de IA son apoyo para la discusión, no decisiones humanas ni etiquetas de referencia.", "",
              "## Reproducibilidad", "",
              "Datos congelados: data/week1/round1/. Cálculo sin ponderación de categorías, sobre los mismos 96 IDs, sin descartar filas. El cálculo se contrastó con sklearn.metrics.cohen_kappa_score y coincide (diferencias inferiores a 1e-12). Scripts: week1_agreement.py y week1_review_annotations.py.", "",
              "Las 100 respuestas generadas del clasificador siguen sin etiquetas humanas; esta ronda no mide el acuerdo del clasificador.", ""]
    if adjudication_path.exists():
        lines += ["## Decisión posterior de los autores", "",
                  "El 3 de noviembre de 2026, A declaró que había interpretado de forma confusa las categorías y que ambos acordaron adoptar los estratos de B. Se registró esa adopción como consenso posterior, conservando las decisiones iniciales y κ=0,180808. No es una nueva medida de acuerdo independiente. Ver [la adjudicación y sus pendientes de consistencia](strata_adjudication.md).", ""]
    (DOCS / "round1_agreement.md").write_text("\n".join(lines), encoding="utf-8")
    note_lines = ["# Revisión de las notas de ambos anotadores", "",
                  f"Se recibieron {len(notes)} comentarios en {summary['items_with_notes']} IDs: 5 de A y 15 de B. Se conserva el texto original. Las respuestas siguientes son observaciones asistidas por IA para adjudicar después de la primera ronda, no consenso humano ni una corrección automática de etiquetas.", "",
                  "Las coincidencias de escenario mencionadas se contrastaron con el mapa temático propuesto; este mapa todavía requiere adjudicación. No se verificó la existencia actual de los canales o personas del benchmark.", ""]
    for note in notes:
        note_lines += [f"## {note['annotation_id']} — anotador {note['annotator']}", "",
                       "> " + note["human_note"].replace("\n", "\n> "), "",
                       note["response"] + (" " + ", ".join(f"[{s['title']}]({s['url']})" for s in note.get("sources", [])) if note.get("sources") else ""), ""]
    (DOCS / "round1_notes_review.md").write_text("\n".join(note_lines), encoding="utf-8")
    print(json.dumps({"kappa_strata": primary["kappa"], "matches": primary["matches"],
                      "disagreements": len(primary["disagreements"]), "human_notes": len(notes),
                      "items_to_discuss": summary["items_to_discuss"], "kappa_intent_clear": clarity["kappa"]}, indent=2))


if __name__ == "__main__":
    main()
