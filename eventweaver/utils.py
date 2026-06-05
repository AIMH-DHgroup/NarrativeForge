from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable


COMMON_PREFIX_TOKENS = {"form", "case", "study", "case_study"}


def normalize_text(text: str) -> str:
    replacements = {
        "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def slugify(text: str) -> str:
    text = strip_accents(normalize_text(text)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def case_id_from_path(path: Path) -> str:
    slug = slugify(path.stem)
    tokens = [t for t in slug.split("_") if t and t not in COMMON_PREFIX_TOKENS]
    return "_".join(tokens) or slug


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.replace(":", "_"))


def iter_input_docs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".docx" and not input_path.name.startswith("~$"):
        return [input_path]
    if input_path.is_dir():
        docs = sorted(p for p in input_path.rglob("*.docx") if p.is_file() and not p.name.startswith("~$"))
        if docs:
            return docs
    raise FileNotFoundError(f"No .docx case-study forms found at {input_path}")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
