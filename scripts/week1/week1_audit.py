"""Reproduce the Week 1 data audit without modifying the experimental corpus.

Run from the repository root: python scripts/week1/week1_audit.py
Only the Python standard library is required. Human annotation files are
created once and are never overwritten by a subsequent audit.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/week1"
DOCS = ROOT / "docs/week1"
SOURCE = ROOT / "data/dataset_v2.jsonl"
VARIANTS = {"chat_harmful", "agent_harmful", "chat_benign", "agent_benign"}
STRATA = {"explicit", "contextual", "framed"}
# These are cues for review, not automatic evidence that the capability is
# malicious. Tor, a VPN, and credential retrieval can have legitimate uses.
TOOL_CUES = {
    "query_onion_search": "context: Tor/Onion rather than a neutral shared search surface",
    "download_from_alpha_bay": "context: marketplace-specific download surface",
    "get_all_login_data": "sensitive capability: retrieving all stored credentials",
    "purchase_credit_card_details": "capability: purchasing card details",
    "add_hidden_malware": "explicitly harmful capability and description",
}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def tokens(text):
    return set(re.findall(r"[a-záéíóúñ0-9]+", re.sub(r"Hint:.*", "", text, flags=re.S).lower()))


def jaccard(left, right):
    a, b = tokens(left), tokens(right)
    return len(a & b) / max(1, len(a | b))


def unwrap(tool):
    return tool.get("function", tool)


def structural_checks(rows, pairs):
    errors = []
    ids = Counter(r.get("id") for r in rows)
    if any(count > 1 for count in ids.values()):
        errors.append("Duplicate entry IDs")
    if len(rows) != 384 or len(pairs) != 96:
        errors.append(f"Expected 384 rows/96 pairs, found {len(rows)}/{len(pairs)}")
    for pid, variants in sorted(pairs.items()):
        if set(variants) != VARIANTS:
            errors.append(f"{pid}: missing/extra variants")
            continue
        if any(len(records) != 1 for records in variants.values()):
            errors.append(f"{pid}: repeated variant")
            continue
        v = {name: records[0] for name, records in variants.items()}
        for key in ("category", "subtlety", "base_id", "source"):
            if len({r.get(key) for r in v.values()}) != 1:
                errors.append(f"{pid}: inconsistent {key}")
        if v["chat_harmful"]["subtlety"] not in STRATA:
            errors.append(f"{pid}: invalid stratum")
        for arm in ("harmful", "benign"):
            c, a = v[f"chat_{arm}"], v[f"agent_{arm}"]
            if c["prompt"] != a["prompt"]:
                errors.append(f"{pid}/{arm}: text changes across formats")
            if c.get("tools") or c.get("tool_definitions"):
                errors.append(f"{pid}/{arm}: chat declares tools")
        for variant, r in v.items():
            expected_format, arm = variant.split("_")
            if r.get("format") != expected_format or r.get("is_harmful") is not (arm == "harmful"):
                errors.append(f"{pid}/{variant}: inconsistent format/harm flag")
            if "\ufffd" in r["prompt"]:
                errors.append(f"{pid}/{variant}: Unicode replacement character")
            if expected_format == "agent":
                schemas = [unwrap(t) for t in r.get("tool_definitions", [])]
                names = [t.get("name") for t in schemas]
                if names != r.get("tools") or len(names) != len(set(names)):
                    errors.append(f"{pid}/{variant}: schema names/order differ from tool list")
                for tool in schemas:
                    params = tool.get("parameters", {})
                    if not isinstance(params, dict) or params.get("type") != "object":
                        errors.append(f"{pid}/{variant}/{tool.get('name')}: invalid input schema")
                    elif not set(params.get("required", [])) <= set(params.get("properties", {})):
                        errors.append(f"{pid}/{variant}/{tool.get('name')}: required property absent")
        for key in ("tools", "tool_definitions", "system_prompt"):
            if v["agent_harmful"][key] != v["agent_benign"][key]:
                errors.append(f"{pid}: agent arms differ in {key}")
        if v["chat_harmful"]["system_prompt"] != v["chat_benign"]["system_prompt"]:
            errors.append(f"{pid}: chat arms differ in system prompt")
        score = round(jaccard(v["chat_harmful"]["prompt"], v["chat_benign"]["prompt"]), 4)
        if any(abs(r.get("lexical_jaccard", score) - score) > .00011 for r in v.values()):
            errors.append(f"{pid}: stored lexical_jaccard differs from recomputation")
    for fmt in ("chat", "agent"):
        if len({r["system_prompt"] for r in rows if r["format"] == fmt}) != 1:
            errors.append(f"Multiple system prompts within {fmt}")
    return errors


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    raw_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    all_pairs = defaultdict(lambda: defaultdict(list))
    for r in raw:
        all_pairs[r["pair_id"]][r["variant"]].append(r)
    errors = structural_checks(raw, all_pairs)
    (DOCS / "structural_checks.json").write_text(json.dumps({
        "source": "data/dataset_v2.jsonl", "source_sha256": source_hash,
        "rows": len(raw), "pairs": len(all_pairs), "errors": errors,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise SystemExit("Structural audit failed: " + "; ".join(errors))
    pairs = {pid: {v: rs[0] for v, rs in variants.items()} for pid, variants in all_pairs.items()}
    manual = json.loads((DATA / "manual_review.json").read_text(encoding="utf-8"))
    if manual["source_sha256"] != source_hash:
        raise ValueError("Manual review belongs to a different corpus; version and re-review it first")
    if set(manual["issues"]) - set(pairs):
        raise ValueError("Manual review references unknown pairs")
    groups = {pid: f"theme_{pid}" for pid in pairs}
    assigned = set()
    for gid, members in manual["theme_groups"].items():
        for pid in members:
            if pid not in pairs or pid in assigned:
                raise ValueError(f"Unknown or multiply grouped pair: {pid}")
            assigned.add(pid)
            groups[pid] = gid

    overlap = []
    for left, right in itertools.combinations(sorted(pairs), 2):
        a, b = pairs[left]["chat_harmful"], pairs[right]["chat_harmful"]
        score = jaccard(a["prompt"], b["prompt"])
        if score >= .25:
            overlap.append({"pair_a": left, "pair_b": right,
                            "harmful_jaccard": round(score, 4),
                            "same_proposed_theme": groups[left] == groups[right],
                            "same_current_stratum": a["subtlety"] == b["subtlety"]})
    overlap.sort(key=lambda row: -row["harmful_jaccard"])
    write_csv(DATA / "similarity_candidates.csv", overlap, [
        "pair_a", "pair_b", "harmful_jaccard", "same_proposed_theme", "same_current_stratum"])

    review, issues, tool_flags = [], [], []
    for pid, v in sorted(pairs.items()):
        h, b, a = v["chat_harmful"], v["chat_benign"], v["agent_harmful"]
        findings = manual["issues"].get(pid, [])
        cues = [name for name in a["tools"] if name in TOOL_CUES]
        for finding in findings:
            issues.append({"pair_id": pid, **finding})
        for name in cues:
            schema = next(unwrap(t) for t in a["tool_definitions"] if unwrap(t)["name"] == name)
            tool_flags.append({"pair_id": pid, "tool": name, "cue": TOOL_CUES[name],
                               "description": schema.get("description", ""),
                               "decision": "author_review_required"})
        review.append({
            "pair_id": pid, "source": h["source"], "category": h["category"],
            "current_stratum": h["subtlety"], "proposed_theme_group_id": groups[pid],
            "review_status": "review_required" if findings or cues else "no_additional_issue_identified",
            "issue_codes": ";".join(sorted({f["code"] for f in findings})),
            "tool_cues": ";".join(cues),
            "harmful_benign_jaccard": round(jaccard(h["prompt"], b["prompt"]), 4),
            "harmful_prompt": h["prompt"], "benign_prompt": b["prompt"],
            "tools": ";".join(a["tools"]),
            "findings": " | ".join(f["finding"] for f in findings),
            "recommendations": " | ".join(f["recommendation"] for f in findings),
            "author_decision": "", "author_notes": "",
        })
    write_csv(DATA / "pair_review.csv", review, list(review[0]))
    write_csv(DATA / "issues.csv", issues, ["pair_id", "code", "priority", "finding", "recommendation"])
    write_csv(DATA / "tool_context_review.csv", tool_flags,
              ["pair_id", "tool", "cue", "description", "decision"])
    write_csv(DATA / "theme_groups_proposed.csv", [
        {"pair_id": pid, "theme_group_id": gid, "status": "proposed_not_adjudicated"}
        for pid, gid in sorted(groups.items())], ["pair_id", "theme_group_id", "status"])
    # A metadata-only candidate. No prompt, tool, label or original file changes.
    grouped = [dict(r, theme_group_id=groups[r["pair_id"]],
                    theme_group_status="proposed_not_adjudicated") for r in raw]
    (DATA / "dataset_v2_grouped_candidate.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in grouped), encoding="utf-8")

    # Opaque IDs and independently shuffled sheets: no previous label, source,
    # category, pair ID, tool name or AI suggestion is exposed in these sheets.
    ids = sorted(pairs)
    random.Random(4101).shuffle(ids)
    key = [{"annotation_id": f"H{i:03d}", "pair_id": pid, "source_sha256": source_hash}
           for i, pid in enumerate(ids, 1)]
    key_path = DATA / "annotation_key_PRIVATE.csv"
    if not key_path.exists():
        write_csv(key_path, key, ["annotation_id", "pair_id", "source_sha256"])
    else:
        with key_path.open(encoding="utf-8-sig", newline="") as f:
            if list(csv.DictReader(f)) != key:
                raise ValueError("Existing annotation key belongs to a different corpus; preserve it and version new sheets")
    for annotator, seed in (("A", 4102), ("B", 4103)):
        path = DATA / f"strata_annotator_{annotator}_BLIND.csv"
        if path.exists():
            continue
        order = list(key)
        random.Random(seed).shuffle(order)
        sheets = [{"annotation_id": r["annotation_id"],
                   "prompt": pairs[r["pair_id"]]["chat_harmful"]["prompt"],
                   "stratum": "", "intent_clear": "", "evidence": "", "notes": ""} for r in order]
        write_csv(path, sheets, list(sheets[0]))

    counts = Counter(f["code"] for f in issues)
    summary = {
        "review_date": manual["review_date"], "reviewer": manual["reviewer"],
        "source_sha256": source_hash, "pairs_reviewed": len(pairs),
        "structural_errors": len(errors), "pairs_with_semantic_findings": len(manual["issues"]),
        "findings_by_code": dict(counts), "pairs_with_tool_cues": len({r["pair_id"] for r in tool_flags}),
        "pairs_requiring_review": sum(r["review_status"] == "review_required" for r in review),
        "similarity_threshold": .25, "lexical_candidate_links": len(overlap),
        "proposed_theme_groups": len(set(groups.values())),
        "multi_pair_theme_groups": len(manual["theme_groups"]),
        "theme_mapping_status": "proposed_not_adjudicated",
        "human_annotations_complete": False,
        "note": "Lexical candidates are not proven duplicates. Theme groups include conservative scenario families. No semantic correction or stratum change has been applied to the original corpus.",
    }
    (DOCS / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Revision de los 96 pares — 2 de noviembre de 2026", "",
             "Revision asistida por IA. No equivale a etiquetado humano independiente ni a consenso.", "",
             f"Corpus: `{source_hash}` (SHA-256). Se revisaron los textos de ambos brazos y las herramientas de los 96 pares.", "",
             f"La comprobacion estructural no encontro errores. Hay {len(manual['issues'])} pares con observaciones semanticas y {summary['pairs_with_tool_cues']} con herramientas que requieren revisar el contexto; estos conjuntos se solapan.", "",
             "## Como interpretar los hallazgos", "",
             "- `benign_contradiction`: una instruccion contradice directamente la finalidad benigna declarada.",
             "- `benign_ambiguity` / `harmful_ambiguity`: faltan permisos, hechos o contexto para adjudicar la etiqueta con seguridad. No significa que el caso sea definitivamente danino o benigno.",
             "- `stratum_review`: la etiqueta actual merece adjudicacion segun el codebook. No es una nueva etiqueta humana.",
             "- `capability_gap`: alguna accion solicitada no corresponde a las herramientas declaradas. Un terminal puede suplir muchas capacidades; no se marca como faltante cuando ya permite la accion.",
             "- `pair_confound`: cambian tema, poblacion, datos o pasos ademas de la intencion. La similitud lexica no resuelve por si sola este problema.",
             "- `category_review`: desacuerdo posible sobre la categoria primaria, no prueba de que el par sea invalido.", "",
             "Los indicios de herramientas no convierten automaticamente una capacidad legitima (por ejemplo Tor o acceso a credenciales) en maliciosa. Tampoco basta renombrar add_hidden_malware: su descripcion y funcion siguen siendo daninas. La neutralizacion debe controlar nombre, descripcion, esquema y capacidad en ambos brazos.", "",
             "## Observaciones por par", ""]
    for row in review:
        pid = row["pair_id"]
        findings = manual["issues"].get(pid, [])
        if not findings and not row["tool_cues"]:
            continue
        lines += [f"### {pid}", "",
                  f"Categoria: {row['category']}. Estrato actual: {row['current_stratum']}. Grupo propuesto: `{groups[pid]}`.", ""]
        for finding in findings:
            lines += [f"- **{finding['code']} ({finding['priority']}):** {finding['finding']} Propuesta: {finding['recommendation']}"]
        if row["tool_cues"]:
            lines += [f"- **Contexto de herramientas:** `{row['tool_cues']}` tambien se ofrece al brazo benigno. Revisar la funcion y el esquema; conservar simetria entre brazos."]
        lines.append("")
    lines += ["## Duplicados, grupos y limites", "",
              f"El barrido reproducible usa Jaccard de conjuntos de palabras >= 0,25 y obtiene {len(overlap)} enlaces candidatos. Es un barrido propio: no se afirma que reproduzca los 24 cruces del mentor, cuya lista completa no fue adjuntada.", "",
              f"Se proponen {len(set(groups.values()))} grupos de escenario para los 96 pares. {len(manual['theme_groups'])} grupos contienen mas de un par. La agrupacion incluye reformulaciones y familias conservadoras; falta adjudicacion humana antes de congelar los folds.", "",
              "Dos variantes de un escenario pueden tener estratos diferentes cuando una agrega un encuadre legitimador real. Compartir grupo no obliga a compartir estrato. ah_test_66/hate_03 merece revision especial porque la formulacion del mecanismo es casi igual.", "",
              "No se ejecutaron modelos ni se recalcularon AUROC sobre el candidato. Las activaciones anteriores solo sirven para textos y herramientas identicos a los originales. Si se cambia contenido, hay que volver a generar sus activaciones antes de atribuir resultados a un corpus corregido.", "",
              "Los casos sin observaciones aparecen igualmente en pair_review.csv como no_additional_issue_identified: eso no significa validacion definitiva. Toda aceptacion o exclusion debe quedar en author_decision y author_notes.", ""]
    (DOCS / "pair_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
