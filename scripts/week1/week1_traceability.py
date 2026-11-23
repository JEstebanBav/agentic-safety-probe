"""Check the article's evidence against saved runs without rewriting the paper.

python scripts/week1/week1_traceability.py
This checks the evidence object and source hashes, and maps all 16 tables.
It does not claim a fresh cell-by-cell review of every rounded manuscript cell.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from build_paper import MODELS, model_metrics
from week1_audit import write_csv

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/week1"
MAPPING = {
    1: ("data/dataset_v2.jsonl", "category/subtlety counts", "Current labels; needs human adjudication"),
    2: ("data/dataset_v2.jsonl", "lexical_jaccard by subtlety", "Overlap does not establish a valid minimal pair"),
    3: ("results/ultimos_resultados/*/results.json; model architecture configuration", "model dimensions; layers_analyzed", "Architecture head/FFN dimensions not independently revalidated in this audit"),
    4: ("results/ultimos_resultados/*/results.json; causal_results.json", "config; best_layer; layers_analyzed; available stages", "Saved runs, not a new experiment"),
    5: ("results.json", "heldout.detection -> metrics.<model>.deteccion", "Needs scenario-group reevaluation after adjudication"),
    6: ("results.json", "format_effect.held_out; heldout -> desplazamiento", "Original corpus only"),
    7: ("results.json", "format_effect.held_out.per_layer -> escala", "Original corpus only"),
    8: ("results.json", "format_effect.held_out.by_stratum -> estratos", "Human stratum labels pending"),
    9: ("factorial_results.json", "by_layer.<reference>.interaction -> factorial_referencia", "Descriptive full-sample direction"),
    10: ("robust_results.json", "rotation.<reference> -> rotacion", "Original corpus only"),
    11: ("robust_results.json", "per_layer.<reference> -> monitores", "Needs scenario-group and external evaluation"),
    12: ("results.json", "pca_by_layer -> pca", "Descriptive in-sample geometry"),
    13: ("robust_results.json", "layer_selection -> seleccion", "Original pair-level split"),
    14: ("causal_results.json; behavior_generations.jsonl", "behavior; baseline records -> conducta", "Classifier correction changes Llama and Qwen counts; see separate rescore"),
    15: ("causal_results.json; behavior_generations.jsonl", "causal.<layer>.per_dose -> causal", "Reclassification affects eligibility; historical cohorts are not a new causal run"),
    16: ("results.json; causal_results.json", "config; layers_analyzed; intervention_layers", "Rendered tokenizer prompts are still pending"),
}


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((ROOT / "docs/paper/evidence/metrics.json").read_text(encoding="utf-8"))
    tables = json.loads((ROOT / "docs/paper/evidence/table_transcription.json").read_text(encoding="utf-8"))
    models, files = [], []
    for key, spec in MODELS.items():
        extracted = model_metrics(key)
        # JSON equality includes source metadata and all nested metric values.
        matches = json.dumps(extracted, sort_keys=True) == json.dumps(recorded[key], sort_keys=True)
        models.append({"model": key, "source": recorded[key]["fuente"],
                       "fresh_extraction_matches_recorded_evidence": matches})
        base = ROOT / recorded[key]["fuente"]
        for name, expected in recorded[key]["archivos_sha256"].items():
            path = base / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            relative = path.relative_to(ROOT).as_posix()
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                                     cwd=ROOT, capture_output=True).returncode == 0
            files.append({"path": relative, "hash_matches_recorded": actual == expected,
                          "tracked_in_local_git": tracked})
    mapped = []
    for key, table in tables.items():
        sources, selector, note = MAPPING[table["number"]]
        mapped.append({"table": table["number"], "key": key, "source_files": sources,
                       "selector": selector, "status_or_limit": note})
    write_csv(DOCS / "table_sources.csv", sorted(mapped, key=lambda r: r["table"]),
              ["table", "key", "source_files", "selector", "status_or_limit"])
    report = {"models": models, "source_files": files, "tables_mapped": len(mapped),
              "scope": "Re-extraction of the existing evidence plus source hashes and table-to-source mapping; not fresh model runs or an independent check of every rounded table cell.",
              "remote_github_checked": False}
    (DOCS / "results_traceability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Trazabilidad local de resultados", "",
             f"Se mapearon {len(mapped)} tablas del articulo a sus fuentes. Se reconstruyo el objeto de evidencia de los cuatro modelos usando model_metrics, sin regenerar figuras ni modificar el manuscrito.", "",
             "| Modelo | Extraccion coincide con evidencia guardada | Ruta |", "|---|---|---|"]
    for model in models:
        lines.append(f"| {model['model']} | {model['fresh_extraction_matches_recorded_evidence']} | `{model['source']}` |")
    lines += ["", f"Coinciden {sum(r['hash_matches_recorded'] for r in files)} de {len(files)} hashes de archivos fuente registrados; {sum(r['tracked_in_local_git'] for r in files)} estan versionados en el Git local. No se consulto el remoto de GitHub.", "",
              "La corrida Qwen3-8B reciente, incluidas sus generaciones y resultados causales, ya esta rastreada en el Git local de este directorio. La observacion del mentor sobre GitHub no debe asumirse automaticamente vigente para el remoto actual sin consultarlo.", "",
              "El mapa detallado esta en table_sources.csv. La tabla de arquitectura requiere comprobar configuraciones de cabezas y FFN por separado. Esta auditoria verifica la extraccion de evidencia; no se declara una nueva comprobacion celda por celda de todas las tablas redondeadas.", "",
              "Las tablas conductuales 14 y 15 necesitan considerar la correccion del clasificador. Los informes de week1_rescore conservan las diferencias sin reemplazar resultados originales. La coincidencia con la evidencia antigua acredita trazabilidad, no valida por si misma etiquetas, grupos, generalizacion ni causalidad.", ""]
    (DOCS / "results_traceability.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"models": models, "source_files": len(files),
                      "matching_source_hashes": sum(r["hash_matches_recorded"] for r in files),
                      "tables_mapped": len(mapped)}, indent=2))
    if not all(r["fresh_extraction_matches_recorded_evidence"] for r in models) or not all(r["hash_matches_recorded"] for r in files):
        raise SystemExit("Some source/evidence values do not match; inspect results_traceability.json")


if __name__ == "__main__":
    main()
