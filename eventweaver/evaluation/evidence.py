from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import EvaluationInput


def read_json(path: str) -> Any:
    return parse_json(Path(path).read_bytes())


def parse_json(raw: bytes) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(raw.decode("utf-8"), parse_constant=invalid)


def artifact(inputs: EvaluationInput, path: str) -> Any:
    return inputs.artifacts[path] if path in inputs.artifacts else read_json(path)


def require_text_fields(value: dict, fields: tuple[str, ...]) -> None:
    if any(not isinstance(value.get(field), str) or not value[field].strip() for field in fields):
        raise ValueError("Required non-empty text fields: " + ", ".join(fields))


def check_span(text: str, evidence: dict[str, Any]) -> None:
    start, end = evidence.get("start"), evidence.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Invalid evidence span; offsets must be Unicode code points, end exclusive")
    if text[start:end] != evidence.get("quote"):
        raise ValueError("Evidence quote does not match the referenced source span")


def check_evidence(corpus: dict[str, Any], evidence: dict[str, Any]) -> None:
    documents = {document["id"]: document for document in corpus["documents"]}
    document = documents.get(evidence.get("document_id"))
    page_index = evidence.get("page_index")
    if document is None or type(page_index) is not int or not 0 <= page_index < len(document["pages"]):
        raise ValueError("Unknown evidence document or page index")
    check_span(document["pages"][page_index]["text"], evidence)


def load_annotations(inputs: EvaluationInput, config: dict[str, Any]) -> dict[str, Any]:
    annotations = artifact(inputs, config["annotations"])
    if annotations.get("narrative_sha256") != inputs.narrative["sha256"]:
        raise ValueError("Annotations belong to a different narrative")
    if annotations.get("corpus_sha256") != inputs.corpus["sha256"]:
        raise ValueError("Annotations belong to a different corpus")
    require_text_fields(annotations, ("annotator", "annotated_at"))
    return annotations
