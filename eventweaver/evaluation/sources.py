from __future__ import annotations

import hashlib
import re
from pathlib import Path

from eventweaver.ingestion.loader import discover_documents, iter_documents

from .models import EvaluationInput, content_hash


def load_inputs(narrative_path: Path, source_paths: list[Path], *,
                narrative_id: str | None = None, original_prompt: str | None = None) -> EvaluationInput:
    if narrative_path.suffix.lower() != ".md" or not narrative_path.is_file():
        raise ValueError("Narrative input must be an existing Markdown file")
    raw = narrative_path.read_bytes()
    text = raw.decode("utf-8-sig")
    lines = text.splitlines(keepends=True)
    heading = re.match(r"^ {0,3}#{1,6}\s+\S", lines[0]) if lines else None
    body_start = len(lines[0]) if heading else 0
    body = text[body_start:]
    if not body.strip():
        raise ValueError("Narrative body must not be empty")

    paths = discover_documents(source_paths)
    if narrative_path.resolve() in paths:
        raise ValueError("The evaluated narrative must not be included in its source corpus")
    documents = []
    # Evaluation does not run OCR, write OCR caches, or silently skip blank pages.
    for document in iter_documents(paths, ocr_mode="never", cache_dir=None):
        digest = hashlib.sha256(document.source_path.read_bytes()).hexdigest()
        pages = [
            {"index": index, "page_or_row": page.page_number,
             "text": page.text, "extraction_method": page.extraction_method}
            for index, page in enumerate(document.pages)
        ]
        if not pages or any(not page["text"].strip() for page in pages):
            raise ValueError(f"Source has unreadable pages; provide reviewed extracted text: {document.source_path}")
        documents.append({"id": "D-" + digest, "sha256": digest,
                          "text_sha256": content_hash(pages), "path": str(document.source_path),
                          "format": document.source_type, "pages": pages})
    if len({doc["id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate source content: supply each document only once")
    documents.sort(key=lambda doc: doc["id"])
    digest = content_hash([
        {"id": doc["id"], "text_sha256": doc["text_sha256"]} for doc in documents
    ])
    return EvaluationInput(
        narrative={"id": narrative_id or narrative_path.stem,
                   "path": str(narrative_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                   "text": text, "body": body, "body_start": body_start,
                   "offset_convention": "Unicode code points in UTF-8-sig decoded text; end exclusive"},
        corpus={"id": "C-" + digest, "sha256": digest, "documents": documents,
                "selection": "all extracted pages; no retrieval or truncation; OCR disabled"},
        original_prompt=original_prompt,
    )
