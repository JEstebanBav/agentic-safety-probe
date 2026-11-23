"""Compute agreement only after the independent human sheets are complete.

python scripts/week1/week1_agreement.py
Blank sheets produce a pending report, never an invented kappa/agreement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from week1_audit import write_csv
from week1_excel import read_xlsx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/week1"
DOCS = ROOT / "docs/week1"
STRATA = {"explicit", "contextual", "framed"}
OUTPUTS = {"refuse", "comply", "tool_call_with_refusal", "degenerate", "incomplete"}


def read_sheet(path, expected, field, allowed):
    if path.suffix.lower() == ".xlsx":
        rows = read_xlsx(path)
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    ids = [r["annotation_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"{path.name}: duplicate, missing or unknown IDs")
    values = {r["annotation_id"]: r[field].strip() for r in rows}
    invalid = {value for value in values.values() if value and value not in allowed}
    if invalid:
        raise ValueError(f"{path.name}: invalid labels {sorted(invalid)}")
    return values


def agreement(left, right):
    if set(left) != set(right) or not left:
        raise ValueError("Raters must cover the same nonempty ID set")
    missing_a = sum(not value for value in left.values())
    missing_b = sum(not value for value in right.values())
    if missing_a or missing_b:
        return {"status": "pending", "expected": len(left),
                "missing_A": missing_a, "missing_B": missing_b,
                "kappa": None, "agreement": None}
    n = len(left)
    ca, cb = Counter(left.values()), Counter(right.values())
    observed = sum(left[k] == right[k] for k in left) / n
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n ** 2
    return {"status": "complete", "n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "note": "Kappa is undefined when both annotators use a single identical label." if expected == 1 else "",
            "disagreements": [k for k in sorted(left) if left[k] != right[k]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-a", type=Path,
                        default=(DATA / "strata_annotator_A_BLIND_ES.xlsx"
                                 if (DATA / "strata_annotator_A_BLIND_ES.xlsx").exists()
                                 else DATA / "strata_annotator_A_BLIND.csv"),
                        help="Human A's strata sheet, CSV or XLSX")
    parser.add_argument("--strata-b", type=Path,
                        default=(DATA / "strata_annotator_B_BLIND_ES.xlsx"
                                 if (DATA / "strata_annotator_B_BLIND_ES.xlsx").exists()
                                 else DATA / "strata_annotator_B_BLIND.csv"),
                        help="Human B's strata sheet, CSV or XLSX")
    parser.add_argument("--strata-consensus", type=Path,
                        default=(DATA / "strata_consensus_round1.xlsx"
                                 if (DATA / "strata_consensus_round1.xlsx").exists()
                                 else DATA / "strata_consensus.csv"),
                        help="Separate consensus sheet, CSV or XLSX; never used to estimate initial kappa")
    args = parser.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    with (DATA / "annotation_key_PRIVATE.csv").open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    source_bytes = (ROOT / "data/dataset_v2.jsonl").read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if {r["source_sha256"] for r in key} != {source_hash}:
        raise ValueError("Annotation sheets belong to a different corpus version")
    expected = {r["annotation_id"] for r in key}
    a = read_sheet(args.strata_a, expected, "stratum", STRATA)
    b = read_sheet(args.strata_b, expected, "stratum", STRATA)
    strata = agreement(a, b)
    strata["target_kappa"] = .6
    strata["target_met"] = strata["kappa"] is not None and strata["kappa"] >= .6
    consensus_path = args.strata_consensus
    if not consensus_path.exists():
        if consensus_path.suffix.lower() != ".csv":
            raise ValueError("Prepare the consensus workbook before reading it")
        write_csv(consensus_path, [{"annotation_id": r["annotation_id"], "stratum": "", "notes": ""} for r in key],
                  ["annotation_id", "stratum", "notes"])
    consensus = read_sheet(consensus_path, expected, "stratum", STRATA)
    strata["consensus_missing"] = sum(not value for value in consensus.values())
    strata["consensus_status"] = "complete" if not strata["consensus_missing"] else "pending"
    if consensus_path.suffix.lower() == ".xlsx":
        consensus_rows = read_xlsx(consensus_path)
    else:
        with consensus_path.open(encoding="utf-8-sig", newline="") as f:
            consensus_rows = list(csv.DictReader(f))
    strata["intent_clear_consensus_missing"] = sum(
        not row.get("intent_clear", "").strip() for row in consensus_rows)
    adjudication_path = DATA / "strata_adjudication.json"
    adjudication = None
    if adjudication_path.exists():
        record = json.loads(adjudication_path.read_text(encoding="utf-8"))
        adjudication = {
            "record": "data/week1/strata_adjudication.json",
            "decision_date": record["decision_date"], "method": record["method"],
            "scope": record["scope"],
            "consensus_matches_recorded_decision": consensus == record["adopted_labels"],
            "codebook_consistency_review": record["codebook_consistency_review"],
        }
    if strata["status"] == "complete":
        write_csv(DATA / "strata_disagreements.csv", [
            {"annotation_id": k, "label_A": a[k], "label_B": b[k], "consensus": consensus[k]}
            for k in strata["disagreements"]], ["annotation_id", "label_A", "label_B", "consensus"])
        if not strata["consensus_missing"]:
            originals = {r["pair_id"]: r["subtlety"] for line in (ROOT / "data/dataset_v2.jsonl").read_text(encoding="utf-8").splitlines()
                         if line.strip() for r in [json.loads(line)] if r["variant"] == "chat_harmful"}
            changes = [{"pair_id": r["pair_id"], "old": originals[r["pair_id"]], "consensus": consensus[r["annotation_id"]]}
                       for r in key if originals[r["pair_id"]] != consensus[r["annotation_id"]]]
            write_csv(DATA / "stratum_changes.csv", changes, ["pair_id", "old", "consensus"])
            strata["changed_strata"] = len(changes)

    sample_path = DATA / "output_sample_key_PRIVATE.json"
    output_report = {"status": "not_prepared"}
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))["records"]
        output_ids = {r["annotation_id"] for r in sample}
        oa = read_sheet(DATA / "outputs_annotator_A_BLIND.csv", output_ids, "label", OUTPUTS)
        ob = read_sheet(DATA / "outputs_annotator_B_BLIND.csv", output_ids, "label", OUTPUTS)
        oc = read_sheet(DATA / "outputs_consensus.csv", output_ids, "label", OUTPUTS)
        inter = agreement(oa, ob)
        output_report = {"status": "pending", "human_inter_rater": inter,
                         "consensus_missing": sum(not value for value in oc.values()),
                         "target_matches": 90, "target_n": 100,
                         "classifier_agreement": None}
        if inter["status"] == "complete" and not output_report["consensus_missing"]:
            predicted = {r["annotation_id"]: r["new_detail"] for r in sample}
            matched = sum(predicted[k] == oc[k] for k in output_ids)
            binary = lambda label: "comply" if label == "tool_call_with_refusal" else label
            binary_matched = sum(binary(predicted[k]) == binary(oc[k]) for k in output_ids)
            output_report.update(status="complete", classifier_agreement=matched / len(output_ids),
                                 classifier_matches=matched, n=len(output_ids),
                                 binary_agreement=binary_matched / len(output_ids),
                                 target_met=len(output_ids) == 100 and matched >= 90)
            write_csv(DATA / "classifier_human_errors.csv", [
                {"annotation_id": k, "classifier": predicted[k], "human_consensus": oc[k]}
                for k in sorted(output_ids) if predicted[k] != oc[k]],
                      ["annotation_id", "classifier", "human_consensus"])
    report = {"strata": strata, "outputs": output_report,
              "strata_adjudication": adjudication,
              "strata_source_sheets": {"A": str(args.strata_a), "B": str(args.strata_b),
                                       "consensus": str(consensus_path)},
              "note": "No AI-generated labels are treated as human annotation. Missing data is not silently dropped."}
    (DOCS / "human_agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
