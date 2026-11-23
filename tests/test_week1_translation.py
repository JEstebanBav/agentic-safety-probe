"""Check bilingual coverage, unchanged identifiers and usable Excel input."""
import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "week1"))
from week1_agreement import read_sheet
from week1_excel import read_xlsx, update_xlsx_text_cells
from week1_translate_sheets import write_workbook


class TranslationTests(unittest.TestCase):
    def test_coverage_and_reference_tokens(self):
        translations = json.loads((ROOT / "data/week1/strata_translations_es.json").read_text(encoding="utf-8"))["translations"]
        with (ROOT / "data/week1/strata_annotator_A_BLIND.csv").open(encoding="utf-8-sig", newline="") as f:
            originals = {r["annotation_id"]: r["prompt"] for r in csv.DictReader(f)}
        self.assertEqual(set(originals), set(translations))
        self.assertEqual(len(translations), 96)
        patterns = [
            r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]+",
            r"https?://[^\s)]+", r"0x[A-Fa-f0-9]+",
            r"(?<![\w@])@[A-Za-z0-9_]+", r"\$\d+(?:\.\d+)?",
            r"(?<![A-Za-z0-9])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+",
            r"\b8/2[13]\b",
        ]
        for identifier, original in originals.items():
            for pattern in patterns:
                for token in re.findall(pattern, original):
                    with self.subTest(identifier=identifier, token=token):
                        self.assertIn(token, translations[identifier])

    def test_excel_preserves_unicode_text_and_manual_decisions(self):
        rows = [{
            "annotation_id": "H001", "prompt": "A harmless example.\nSecond line.",
            "prompt_es": "Un ejemplo inocuo.\nSegunda línea.", "stratum": "contextual",
            "intent_clear": "uncertain", "evidence": 'Una frase con "comillas" y tildes.',
            "notes": "=este texto no es una fórmula",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.xlsx"
            write_workbook(path, rows, "test")
            self.assertEqual(read_xlsx(path), rows)
            self.assertEqual(read_sheet(path, {"H001"}, "stratum", {"contextual"}), {"H001": "contextual"})

    def test_formulas_are_rejected_instead_of_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.xlsx"
            workbook = xlsxwriter.Workbook(str(path))
            sheet = workbook.add_worksheet("Anotacion")
            sheet.write(0, 0, "ID")
            sheet.write_formula(1, 0, '=CONCAT("H", "001")')
            workbook.close()
            with self.assertRaises(ValueError):
                read_xlsx(path)

    def test_targeted_consensus_update_keeps_original_text_and_guide(self):
        import zipfile
        rows = [{
            "annotation_id": "H001", "prompt": "Original text.", "prompt_es": "Texto original.",
            "stratum": "", "intent_clear": "", "evidence": "", "notes": "",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "consensus.xlsx"
            write_workbook(path, rows, "test")
            with zipfile.ZipFile(path) as archive:
                guide_before = archive.read("xl/worksheets/sheet2.xml")
            update_xlsx_text_cells(path, {"Anotacion": {"D2": "explicit", "G2": "Consenso humano: sí."}})
            updated = read_xlsx(path)
            self.assertEqual(updated[0]["stratum"], "explicit")
            self.assertEqual(updated[0]["notes"], "Consenso humano: sí.")
            for field in ("annotation_id", "prompt", "prompt_es", "intent_clear", "evidence"):
                self.assertEqual(updated[0][field], rows[0][field])
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.read("xl/worksheets/sheet2.xml"), guide_before)


if __name__ == "__main__":
    unittest.main()
