from __future__ import annotations

import re
import string
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_TEXT_COLUMNS = [
    "source_text",
    "source_document",
    "document",
    "input_text",
    "reference_text",
    "case_text",
    "original_text",
]
GENERATED_TEXT_COLUMNS = [
    "generated_text",
    "generated_narrative",
    "output_text",
    "narrative",
    "response",
    "model_output",
]
SOURCE_PATH_COLUMNS = ["source_path", "benchmark_source_file", "source_file", "source", "reference_path", "document_path"]
GENERATED_PATH_COLUMNS = ["output_path", "output", "generated_path", "narrative_path", "response_path"]


@dataclass
class CoverageOptions:
    enabled: bool = False
    thresholds: list[float] = field(default_factory=lambda: [0.70, 0.75, 0.80])
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    skip_entity_coverage: bool = False
    skip_keyphrase_coverage: bool = False


@dataclass
class CoverageResult:
    df: pd.DataFrame
    warnings: list[str]
    available: bool


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) > 2]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))


def normalize_phrase(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def detect_text_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    source = next((lower_to_original[col] for col in SOURCE_TEXT_COLUMNS if col in lower_to_original), None)
    generated = next((lower_to_original[col] for col in GENERATED_TEXT_COLUMNS if col in lower_to_original), None)
    return source, generated


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    return next((lower_to_original[col] for col in candidates if col in lower_to_original), None)


def _candidate_paths(value: object, row: pd.Series) -> list[Path]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        module_path = Path(__file__).resolve()
        for base in module_path.parents:
            candidates.append(base / path)
    runs_file = row.get("nrs_runs_file")
    if runs_file is not None and not pd.isna(runs_file):
        runs_path = Path(str(runs_file))
        for base in [runs_path.parent, *runs_path.parents]:
            candidates.append(base / text)
            if path.name:
                candidates.append(base / path.name)
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return "\n".join(paragraphs)


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_text_from_path_value(value: object, row: pd.Series) -> str:
    for candidate in _candidate_paths(value, row):
        if candidate.exists() and candidate.is_file():
            try:
                return _read_text_file(candidate)
            except Exception:
                continue
    return ""


def _resolve_text_sources(df: pd.DataFrame, source_col: str | None, generated_col: str | None) -> tuple[pd.Series | None, pd.Series | None, list[str]]:
    warnings: list[str] = []
    if source_col:
        source_texts = df[source_col].fillna("").astype(str)
    else:
        source_path_col = _first_existing_column(df, SOURCE_PATH_COLUMNS)
        source_texts = None
        if source_path_col:
            source_texts = df.apply(lambda row: _read_text_from_path_value(row.get(source_path_col), row), axis=1)
            if source_texts.astype(str).str.strip().eq("").all():
                warnings.append(f"Source text could not be reconstructed from path column: {source_path_col}")
        else:
            warnings.append(f"No source text column or source path column found. Accepted path columns: {', '.join(SOURCE_PATH_COLUMNS)}")

    if generated_col:
        generated_texts = df[generated_col].fillna("").astype(str)
    else:
        generated_path_col = _first_existing_column(df, GENERATED_PATH_COLUMNS)
        generated_texts = None
        if generated_path_col:
            generated_texts = df.apply(lambda row: _read_text_from_path_value(row.get(generated_path_col), row), axis=1)
            if generated_texts.astype(str).str.strip().eq("").all():
                warnings.append(f"Generated text could not be reconstructed from path column: {generated_path_col}")
        else:
            warnings.append(f"No generated text column or generated path column found. Accepted path columns: {', '.join(GENERATED_PATH_COLUMNS)}")
    return source_texts, generated_texts, warnings


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def sentence_coverage_metrics(source_sentences: list[str], generated_sentences: list[str], embedder: Any, thresholds: list[float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for threshold in thresholds:
        suffix = f"{int(round(threshold * 100)):03d}"
        metrics[f"source_coverage_{suffix}"] = np.nan
        metrics[f"generation_support_{suffix}"] = np.nan
    if not source_sentences or not generated_sentences or embedder is None:
        return metrics
    source_emb = np.asarray(embedder.encode(source_sentences), dtype="float64")
    generated_emb = np.asarray(embedder.encode(generated_sentences), dtype="float64")
    similarities = _cosine_matrix(source_emb, generated_emb)
    source_max = similarities.max(axis=1)
    generated_max = similarities.max(axis=0)
    for threshold in thresholds:
        suffix = f"{int(round(threshold * 100)):03d}"
        metrics[f"source_coverage_{suffix}"] = float(np.mean(source_max >= threshold))
        metrics[f"generation_support_{suffix}"] = float(np.mean(generated_max >= threshold))
    return metrics


def _load_sentence_transformer(model_name: str, warnings: list[str], force: bool):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        warnings.append(
            "Semantic sentence coverage unavailable because sentence-transformers could not be loaded. "
            "Install optional dependencies with: pip install -e \".[coverage]\". "
            f"Details: {exc}"
        )
        return None


def _load_spacy(warnings: list[str]):
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            warnings.append("spaCy model en_core_web_sm is unavailable; entity coverage is skipped because blank spaCy has no NER pipeline.")
            return None
    except Exception as exc:
        warnings.append(f"Entity coverage unavailable because spaCy is not installed: {exc}")
        return None


def _load_keybert(warnings: list[str]):
    try:
        from keybert import KeyBERT

        return KeyBERT()
    except Exception as exc:
        warnings.append(f"Keyphrase coverage unavailable because KeyBERT is not installed: {exc}")
        return None


def entity_coverage(source_text: str, generated_text: str, nlp: Any) -> float:
    if nlp is None:
        return np.nan
    source_entities = {normalize_phrase(ent.text) for ent in nlp(str(source_text)).ents if normalize_phrase(ent.text)}
    if not source_entities:
        return np.nan
    generated_norm = normalize_phrase(generated_text)
    covered = sum(1 for entity in source_entities if entity in generated_norm)
    return covered / len(source_entities)


def keyphrase_coverage(source_text: str, generated_text: str, kw_model: Any, top_n: int = 20) -> float:
    if kw_model is None:
        return np.nan
    phrases = kw_model.extract_keywords(str(source_text), keyphrase_ngram_range=(1, 3), top_n=top_n)
    normalized = [normalize_phrase(item[0] if isinstance(item, tuple) else item) for item in phrases]
    normalized = [item for item in normalized if item]
    if not normalized:
        return np.nan
    generated_norm = normalize_phrase(generated_text)
    return sum(1 for phrase in normalized if phrase in generated_norm) / len(normalized)


def compute_coverage_for_row(
    source_text: str,
    generated_text: str,
    *,
    thresholds: list[float],
    embedder: Any = None,
    nlp: Any = None,
    kw_model: Any = None,
) -> dict[str, float]:
    source_words = word_count(source_text)
    generated_words = word_count(generated_text)
    source_sentences = split_sentences(source_text)
    generated_sentences = split_sentences(generated_text)
    metrics: dict[str, float] = {
        "source_word_count": source_words,
        "generated_word_count": generated_words,
        "compression_ratio": generated_words / source_words if source_words else np.nan,
        "source_sentence_count": len(source_sentences),
        "generated_sentence_count": len(generated_sentences),
    }
    metrics.update(sentence_coverage_metrics(source_sentences, generated_sentences, embedder, thresholds))
    metrics["entity_coverage"] = entity_coverage(source_text, generated_text, nlp)
    metrics["keyphrase_coverage"] = keyphrase_coverage(source_text, generated_text, kw_model)
    coverage_075 = metrics.get("source_coverage_075", np.nan)
    return metrics | {
        "coverage_adjusted_bertscore_075": np.nan,
        "coverage_adjusted_semantic_similarity_075": np.nan,
        "omission_risk_075": np.nan,
    }


def apply_coverage_diagnostics(df: pd.DataFrame, options: CoverageOptions, tables_dir: Path) -> CoverageResult:
    out = df.copy()
    warnings: list[str] = []
    source_col, generated_col = detect_text_columns(out)
    source_texts, generated_texts, path_warnings = _resolve_text_sources(out, source_col, generated_col)
    warnings.extend(path_warnings)
    unavailable_path = tables_dir / "coverage_diagnostics_unavailable.txt"
    if source_texts is None or generated_texts is None or source_texts.astype(str).str.strip().eq("").all() or generated_texts.astype(str).str.strip().eq("").all():
        message = (
            "Coverage diagnostics were skipped because source/generated text could not be found or reconstructed.\n"
            f"Detected source column: {source_col or 'missing'}\n"
            f"Detected generated column: {generated_col or 'missing'}\n"
            f"Accepted source columns: {', '.join(SOURCE_TEXT_COLUMNS)}\n"
            f"Accepted generated columns: {', '.join(GENERATED_TEXT_COLUMNS)}\n"
            f"Accepted source path columns: {', '.join(SOURCE_PATH_COLUMNS)}\n"
            f"Accepted generated path columns: {', '.join(GENERATED_PATH_COLUMNS)}\n"
            + ("\n".join(warnings) + "\n" if warnings else "")
        )
        tables_dir.mkdir(parents=True, exist_ok=True)
        unavailable_path.write_text(message, encoding="utf-8")
        warnings.append(message.strip())
        return CoverageResult(out, warnings, False)

    embedder = _load_sentence_transformer(options.model, warnings, options.enabled)
    nlp = None if options.skip_entity_coverage else _load_spacy(warnings)
    kw_model = None if options.skip_keyphrase_coverage else _load_keybert(warnings)

    rows = []
    for idx, row in out.iterrows():
        metrics = compute_coverage_for_row(
            source_texts.loc[idx],
            generated_texts.loc[idx],
            thresholds=options.thresholds,
            embedder=embedder,
            nlp=nlp,
            kw_model=kw_model,
        )
        coverage_075 = metrics.get("source_coverage_075", np.nan)
        bert = pd.to_numeric(pd.Series([row.get("bertscore_f1", np.nan)]), errors="coerce").iloc[0]
        semantic = pd.to_numeric(pd.Series([row.get("semantic_similarity", np.nan)]), errors="coerce").iloc[0]
        metrics["coverage_adjusted_bertscore_075"] = bert * coverage_075 if not pd.isna(bert) and not pd.isna(coverage_075) else np.nan
        metrics["coverage_adjusted_semantic_similarity_075"] = semantic * coverage_075 if not pd.isna(semantic) and not pd.isna(coverage_075) else np.nan
        metrics["omission_risk_075"] = bert - coverage_075 if not pd.isna(bert) and not pd.isna(coverage_075) else np.nan
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows, index=out.index)
    for col in metrics_df.columns:
        out[col] = metrics_df[col]
    if warnings:
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / "coverage_diagnostics_warnings.txt").write_text("\n\n".join(warnings), encoding="utf-8")
    return CoverageResult(out, warnings, True)


def has_coverage_metrics(df: pd.DataFrame) -> bool:
    return "source_coverage_075" in df.columns and df["source_coverage_075"].notna().any()
