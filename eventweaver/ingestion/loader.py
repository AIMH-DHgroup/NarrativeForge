from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .models import DocumentPage, LoadedDocument
from .pdf_loader import load_pdf

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def discover_documents(inputs: list[Path]) -> list[Path]:
    """Resolve once, deduplicate overlapping directories, never silently omit explicit files."""
    found: set[Path] = set()
    for item in inputs:
        path = item.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported format: {path}")
            found.add(path)
        else:
            found.update(p.resolve() for p in path.rglob("*") if p.is_file()
                         and p.suffix.lower() in SUPPORTED_EXTENSIONS
                         and not p.name.startswith("~$"))
    if not found:
        raise ValueError("No PDF, DOCX, TXT, MD or CSV documents found")
    return sorted(found)


def iter_documents(paths: list[Path], *, ocr_mode: str = "auto",
                   ocr_language: str = "ita+eng", ocr_dpi: int = 200,
                   cache_dir: Path | None = None, refresh_ocr: bool = False,
                   progress=lambda message: None) -> Iterator[LoadedDocument]:
    for i, path in enumerate(paths, 1):
        doc_id = f"D{i:05d}"
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            doc = load_pdf(path, document_id=doc_id, ocr_mode=ocr_mode,
                           ocr_language=ocr_language, ocr_dpi=ocr_dpi, cache_dir=cache_dir,
                           refresh_ocr=refresh_ocr, progress=progress)
        else:
            if suffix == ".docx":
                from docx import Document
                from docx.oxml.ns import qn
                from docx.table import Table
                from docx.text.paragraph import Paragraph
                word = Document(str(path))
                parts = []
                # Preserve paragraph/table order instead of appending all tables at the end.
                for child in word.element.body:
                    if child.tag == qn("w:p"):
                        parts.append(Paragraph(child, word).text)
                    elif child.tag == qn("w:tbl"):
                        for row in Table(child, word).rows:
                            parts.append(" | ".join(cell.text for cell in row.cells))
                pages = [DocumentPage(None, "\n".join(parts), "docx")]
            elif suffix == ".csv":
                # Each row is a distinct source unit; keep every column and row number.
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    sample = handle.read(8192)
                    handle.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    rows = csv.reader(handle, dialect)
                    header = next(rows, [])
                    pages = []
                    for number, row in enumerate(rows, 2):
                        if any(cell.strip() for cell in row):
                            text = "\n".join(f"{header[j] if j < len(header) else f'column_{j+1}'}: {value}"
                                             for j, value in enumerate(row))
                            pages.append(DocumentPage(number, text, "csv-row"))
            else:
                pages = [DocumentPage(None, path.read_text(encoding="utf-8-sig"), "text")]
            doc = LoadedDocument(doc_id, path, path.name, suffix[1:], pages)
        if not doc.text:
            raise ValueError(f"Empty document or no extractable text: {path}")
        yield doc
