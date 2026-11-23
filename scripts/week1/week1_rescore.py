"""Re-score saved generations, preserving original results and cohort selection.

python scripts/week1/week1_rescore.py
No generations, activations, or original result files are overwritten. The
human sample is uniform over distinct saved texts in the three latest runs;
its seed, population and source hashes are recorded. Annotation sheets are
created only once so that re-running this script cannot erase human work.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.causal_test import (  # noqa: E402
    answer_has_tool_call, classify_response, classify_response_detailed,
    summarise_baseline, summarise_causal,
)
from week1_audit import write_csv  # noqa: E402

OUT = ROOT / "results/week1_rescore"
DATA = ROOT / "data/week1"
DOCS = ROOT / "docs/week1"


def strict_json(value):
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numeric_diff(before, after, path=""):
    out = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() & after.keys()):
            out += numeric_diff(before[key], after[key], f"{path}.{key}".lstrip("."))
    elif isinstance(before, list) and isinstance(after, list):
        for index, (a, b) in enumerate(zip(before, after)):
            out += numeric_diff(a, b, f"{path}[{index}]")
    elif (isinstance(before, (float, int)) and not isinstance(before, bool)
          and isinstance(after, (float, int)) and not isinstance(after, bool)):
        a, b = strict_json(before), strict_json(after)
        if a != b:
            out.append({"metric": path, "old": a, "new": b})
    return out


def summaries(rows):
    result = {}
    for layer in sorted({r.get("layer") for r in rows}, key=str):
        block = [r for r in rows if r.get("layer") == layer]
        baseline = [r for r in block if r["arm"] == "baseline"]
        causal = [r for r in block if r["arm"] != "baseline"]
        entry = {}
        if baseline:
            entry["baseline"] = summarise_baseline(baseline)
        if causal:
            doses = sorted({r["dose"] for r in causal})
            unit = next((r["alpha"] / r["dose"] for r in causal if r["dose"]), 0.)
            entry["causal_observed_cohort_only"] = summarise_causal(causal, doses, unit)
        result[str(layer)] = entry
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    # Freeze input discovery: never re-score this script's own output.
    paths = sorted(p for p in (ROOT / "results").rglob("behavior_generations.jsonl")
                   if OUT not in p.parents)
    changed, metrics, runs, sample_pool, baseline_rates = [], [], [], {}, []
    for path in paths:
        source = path.relative_to(ROOT).as_posix()
        source_bytes = path.read_bytes()
        old = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
        new = []
        run_changes = 0
        for line_no, row in enumerate(old, 1):
            detail = classify_response_detailed(row["text"])
            label = classify_response(row["text"])
            record = dict(row, behavior=label, behavior_detail=detail,
                          tool_call=answer_has_tool_call(row["text"]))
            new.append(record)
            if row["behavior"] != label:
                run_changes += 1
                changed.append({"source": source, "line": line_no,
                                "pair_id": row["pair_id"], "variant": row["variant"],
                                "layer": row.get("layer"), "arm": row["arm"], "dose": row["dose"],
                                "old": row["behavior"], "new": label, "detail": detail})
            if "ultimos_resultados" in path.parts:
                digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                sample_pool.setdefault(digest, {
                    "source": source, "line": line_no, "pair_id": row["pair_id"],
                    "variant": row["variant"], "arm": row["arm"], "layer": row.get("layer"),
                    "text": row["text"], "text_sha256": digest,
                    "old_label": row["behavior"], "new_label": label, "new_detail": detail,
                })
        destination = OUT / path.parent.relative_to(ROOT / "results")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "rescored_generations.jsonl").write_text("".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in new), encoding="utf-8")
        before, after = summaries(old), summaries(new)
        delta = numeric_diff(before, after)
        metrics += [dict(source=source, **d) for d in delta]
        report = {
            "source": source, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "n_records": len(old), "changed_binary_labels": run_changes,
            "mixed_response_count": sum(r["behavior_detail"] == "tool_call_with_refusal" for r in new),
            "old_summaries": before, "rescored_summaries": after,
            "numeric_changes": delta,
            "causal_limit": "Re-scored outputs from historical cohorts only. Baseline reclassification can change who was eligible for steering. These summaries do not reconstruct missing interventions and must not silently replace manuscript causal tables.",
        }
        (destination / "rescore_report.json").write_text(
            json.dumps(strict_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
        runs.append({k: v for k, v in report.items() if k not in {
            "old_summaries", "rescored_summaries", "numeric_changes", "causal_limit"}})
        if "ultimos_resultados" in path.parts:
            for layer in sorted({r.get("layer") for r in old if r["arm"] == "baseline"}, key=str):
                for variant in ("chat_harmful", "agent_harmful", "chat_benign", "agent_benign"):
                    left = [r for r in old if r["arm"] == "baseline" and r.get("layer") == layer and r["variant"] == variant]
                    right = [r for r in new if r["arm"] == "baseline" and r.get("layer") == layer and r["variant"] == variant]
                    if left:
                        baseline_rates.append({"source": source, "layer": layer, "variant": variant,
                                               "n": len(left), "old_refuse": sum(r["behavior"] == "refuse" for r in left),
                                               "new_refuse": sum(r["behavior"] == "refuse" for r in right),
                                               "mixed": sum(r["behavior_detail"] == "tool_call_with_refusal" for r in right)})
    write_csv(OUT / "label_changes.csv", changed, [
        "source", "line", "pair_id", "variant", "layer", "arm", "dose", "old", "new", "detail"])
    write_csv(OUT / "metric_changes.csv", metrics, ["source", "metric", "old", "new"])
    write_csv(OUT / "latest_baseline_counts.csv", baseline_rates, [
        "source", "layer", "variant", "n", "old_refuse", "new_refuse", "mixed"])

    population = sorted(sample_pool)
    sample_size = min(100, len(population))
    selected = random.Random(4104).sample(population, sample_size)
    sample = [{"annotation_id": f"O{i:03d}", **sample_pool[digest]}
              for i, digest in enumerate(selected, 1)]
    sample_meta = {
        "seed": 4104, "sampling": "uniform without replacement over unique full-text hashes",
        "population": "three latest runs with saved generations; baseline and all intervention arms",
        "population_unique_texts": len(population), "sample_size": sample_size,
        "population_sha256": hashlib.sha256("\n".join(population).encode()).hexdigest(),
        "source_hashes": {r["source"]: r["source_sha256"] for r in runs if "ultimos_resultados" in r["source"]},
    }
    key_path = DATA / "output_sample_key_PRIVATE.json"
    expected_key = {"metadata": sample_meta, "records": sample}
    if not key_path.exists():
        key_path.write_text(json.dumps(expected_key, ensure_ascii=False, indent=2), encoding="utf-8")
    elif json.loads(key_path.read_text(encoding="utf-8")) != expected_key:
        raise ValueError("Existing sample differs: preserve human sheets and version the new sample")
    for annotator, seed in (("A", 4105), ("B", 4106)):
        path = DATA / f"outputs_annotator_{annotator}_BLIND.csv"
        if path.exists():
            continue
        order = list(sample)
        random.Random(seed).shuffle(order)
        write_csv(path, [{"annotation_id": r["annotation_id"], "text": r["text"],
                          "label": "", "notes": ""} for r in order],
                  ["annotation_id", "text", "label", "notes"])
    consensus = DATA / "outputs_consensus.csv"
    if not consensus.exists():
        write_csv(consensus, [{"annotation_id": r["annotation_id"], "label": "", "notes": ""} for r in sample],
                  ["annotation_id", "label", "notes"])
    overview = {"runs": runs, "total_changed_binary_labels": len(changed),
                "changed_numeric_metrics": len(metrics), "human_sample": sample_meta,
                "human_agreement_status": "pending; labels must be supplied by the authors",
                "original_results_modified": False}
    (OUT / "summary.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Reclasificacion de respuestas guardadas", "",
             "Los resultados originales permanecen intactos. Se normalizan apostrofos tipograficos y se registra tool_call_with_refusal como etiqueta detallada. Esa etiqueta cuenta como comply en las estadisticas binarias, porque la llamada fue emitida.", "",
             "El detector sigue siendo heuristico: las comillas corregidas y las pruebas de regresion no sustituyen el acuerdo con personas.", "",
             "| Corrida | Variante | Rechazos antes | Rechazos ahora | Total | Mixtas |",
             "|---|---|---:|---:|---:|---:|"]
    for row in baseline_rates:
        name = row["source"].replace("results/ultimos_resultados/", "").replace("/behavior_generations.jsonl", "")
        lines.append(f"| {name} | {row['variant']} | {row['old_refuse']} | {row['new_refuse']} | {row['n']} | {row['mixed']} |")
    lines += ["", f"Se reclasificaron {sum(r['n_records'] for r in runs)} registros de {len(runs)} archivos (incluidas corridas historicas y una prueba). Cambian {len(changed)} etiquetas binarias. La lista completa de diferencias numericas recalculadas esta en results/week1_rescore/metric_changes.csv; no se han reemplazado las tablas del articulo.", "",
              "Las curvas causales recalculadas describen las cohortes historicas efectivamente generadas. Si cambia una etiqueta basal, tambien puede cambiar la elegibilidad para steering: puede faltar una intervencion o sobrar un caso. No se presenta la reclasificacion como una nueva corrida causal completa.", "",
              f"La muestra humana contiene {sample_size} textos distintos elegidos aleatoriamente entre {len(population)} textos de las tres corridas recientes con generaciones. Incluye respuestas basales y con intervencion. Qwen3-32B no tiene generaciones guardadas en la carpeta reciente, por lo que no se inventan resultados conductuales para ese modelo.", "",
              "Las hojas BLIND no contienen las etiquetas del clasificador. Completar A y B por separado; luego adjudicar outputs_consensus.csv. No abrir output_sample_key_PRIVATE.json hasta terminar la anotacion. La semilla es 4104 y la poblacion queda identificada por hash.", ""]
    (DOCS / "classifier_rescore.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "records": sum(r["n_records"] for r in runs),
                      "binary_label_changes": len(changed), "sample_size": sample_size,
                      "latest_baseline_counts": baseline_rates}, indent=2))


if __name__ == "__main__":
    main()
