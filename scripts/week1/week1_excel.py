"""Read annotation XLSX files without requiring openpyxl or evaluating formulas."""
from __future__ import annotations

import posixpath
import re
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
HEADERS = {
    "ID": "annotation_id",
    "Original (inglés)": "prompt",
    "Traducción (español)": "prompt_es",
    "Estrato": "stratum",
    "¿El daño está claro?": "intent_clear",
    "Fragmento de evidencia": "evidence",
    "Notas": "notes",
}


def _column_index(reference):
    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", reference)
    if not match:
        raise ValueError(f"Invalid cell reference: {reference}")
    column = 0
    for char in match.group(1):
        column = column * 26 + ord(char) - ord("A") + 1
    return column, int(match.group(2))


def update_xlsx_text_cells(path: Path, updates):
    """Update explicitly named text cells; preserve all other workbook members.

    Caller must preserve the prior workbook and authorise the changes. Values
    are stored as literal inline strings, never interpreted as formulas.
    """
    path = path.resolve(strict=True)
    replacements = {}
    with zipfile.ZipFile(path) as source:
        wb = ET.fromstring(source.read("xl/workbook.xml"))
        rels = ET.fromstring(source.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.findall(f"{{{REL_NS}}}Relationship")}
        names = {s.get("name"): s.get(f"{{{DOC_REL_NS}}}id") for s in wb.findall("m:sheets/m:sheet", NS)}
        ET.register_namespace("", NS["m"])
        for name, cells in updates.items():
            if name not in names:
                raise ValueError(f"Worksheet not found: {name}")
            target = targets[names[name]]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            root = ET.fromstring(source.read(member))
            data = root.find("m:sheetData", NS)
            rows = {int(row.get("r")): row for row in data.findall("m:row", NS)}
            for reference, value in cells.items():
                if not isinstance(value, str):
                    raise ValueError("Cell updates must be literal strings")
                _, row_number = _column_index(reference)
                row = rows.get(row_number)
                if row is None:
                    row = ET.Element(f"{{{NS['m']}}}row", {"r": str(row_number)})
                    data.append(row)
                    rows[row_number] = row
                cell = next((c for c in row.findall("m:c", NS) if c.get("r") == reference), None)
                if cell is None:
                    cell = ET.SubElement(row, f"{{{NS['m']}}}c", {"r": reference})
                for child in list(cell):
                    cell.remove(child)
                cell.set("t", "inlineStr")
                inline = ET.SubElement(cell, f"{{{NS['m']}}}is")
                node = ET.SubElement(inline, f"{{{NS['m']}}}t")
                node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                node.text = value
                row[:] = sorted(list(row), key=lambda c: _column_index(c.get("r"))[0])
            data[:] = sorted(list(data), key=lambda row: int(row.get("r")))
            dimension = root.find("m:dimension", NS)
            references = [cell.get("r") for row in data for cell in row.findall("m:c", NS)]
            if dimension is not None and references:
                max_col = max(_column_index(r)[0] for r in references)
                max_row = max(_column_index(r)[1] for r in references)
                letters, number = "", max_col
                while number:
                    number, digit = divmod(number - 1, 26)
                    letters = chr(ord("A") + digit) + letters
                dimension.set("ref", f"A1:{letters}{max_row}")
            replacements[member] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".xlsx.tmp", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            with zipfile.ZipFile(temp_path, "w") as destination:
                destination.comment = source.comment
                for item in source.infolist():
                    destination.writestr(item, replacements.get(item.filename, source.read(item.filename)))
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def read_xlsx(path: Path):
    """Return records from the Anotacion sheet; retain blank human decisions."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", NS)
        sheet = next((s for s in sheets if s.get("name") == "Anotacion"), None)
        if sheet is None:
            raise ValueError(f"{path.name}: the Anotacion sheet is missing")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_id = sheet.get(f"{{{DOC_REL_NS}}}id")
        target = next(r.get("Target") for r in relationships.findall(f"{{{REL_NS}}}Relationship")
                      if r.get("Id") == relation_id)
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
                       for si in root.findall("m:si", NS)]
        sheet_xml = ET.fromstring(archive.read(member))
        matrix = []
        for row in sheet_xml.findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                letters = re.match(r"[A-Z]+", cell.get("r", "")).group()
                index = 0
                for char in letters:
                    index = index * 26 + ord(char) - ord("A") + 1
                if cell.find("m:f", NS) is not None:
                    raise ValueError(f"{path.name}: formulas are not allowed in annotation cells")
                value = cell.find("m:v", NS)
                text = value.text if value is not None and value.text is not None else ""
                if cell.get("t") == "s":
                    text = strings[int(text)]
                elif cell.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                values[index - 1] = text
            if values:
                matrix.append(values)
    if not matrix:
        raise ValueError(f"{path.name}: empty annotation sheet")
    headers = {index: HEADERS.get(value, value) for index, value in matrix[0].items()}
    if "annotation_id" not in headers.values():
        raise ValueError(f"{path.name}: ID header is missing")
    records = [{header: row.get(index, "") for index, header in headers.items()} for row in matrix[1:]]
    return [row for row in records if row.get("annotation_id")]
