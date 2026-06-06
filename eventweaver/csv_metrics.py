from __future__ import annotations

import re

from .text_metrics import broken_sentence_count, forbidden_formatting_count, semantic_similarity_with_method, word_count
from .utils import normalize_text, strip_accents


FIELD_GROUPS: list[tuple[str, float, list[str]]] = [
    ("value_chain_descriptor", 3.0, ["Descriptor of the value chain"]),
    ("territorial_context", 2.0, ["Member State", "District / Sub-region", "NUTS 3", "NUTS 2", "Reference mountain chain", "Reference mountain landscape", "Mountain Reference Landscape", "LAU"]),
    ("issues_challenges", 3.0, ["Issues of interest to the MOVING project", "Challenges of the VC in light of the MOVING project", "Type of challenges"]),
    ("value_chain_and_innovation_type", 2.0, ["Type of VC", "Type of Innovation"]),
    ("innovation_description", 3.0, ["Brief description of the innovation in the VC"]),
    ("innovation_linked_to", 1.0, ["Innovation linked to"]),
    ("land_use_systems", 2.0, ["Land use systems"]),
    ("local_assets", 3.0, ["Local Assets", "Description of the key local assets"]),
    ("synthetic_description", 4.0, ["Synthetic description of the value chain"]),
    ("reasons_for_selection", 2.0, ["Reasons for selection"]),
    ("protected_landscape", 1.0, ["Protected Areas", "Mountain Reference Landscape", "Reference mountain landscape"]),
    ("socioeconomic", 1.0, ["socioeconomic indicators"]),
]


def _get_case_insensitive(row: dict[str, str], column: str) -> str:
    target = column.strip().lower()
    for key, value in row.items():
        if key.strip().lower() == target:
            return str(value).strip()
    return ""


def _normalize_for_match(text: str) -> str:
    return strip_accents(normalize_text(text)).lower()


def _important_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"\b\w+[\w/-]*\b", _normalize_for_match(text)):
        if len(token) < 3:
            continue
        if token in {"and", "the", "for", "with", "from", "into", "that", "this", "project", "value", "chain", "type", "area", "areas"}:
            continue
        terms.append(token)
    return terms


def _field_score(field_text: str, output_text: str) -> float:
    field_text = str(field_text).strip()
    if not field_text:
        return 0.0

    output_norm = _normalize_for_match(output_text)
    field_norm = _normalize_for_match(field_text)
    field_terms = _important_terms(field_text)
    if not field_terms:
        return 0.0

    short_field = len(field_terms) <= 4 or word_count(field_text) <= 6
    direct_hit = field_norm in output_norm
    term_hits = sum(1 for term in field_terms if term in output_norm)
    hit_ratio = term_hits / max(len(field_terms), 1)

    if short_field:
        if direct_hit:
            return 1.0
        if hit_ratio >= 0.75:
            return 1.0
        if hit_ratio >= 0.4:
            return 0.5
        if semantic_similarity_with_method(field_text, output_text, "auto")[0] >= 0.75:
            return 0.5
        return 0.0

    if direct_hit or hit_ratio >= 0.8:
        return 1.0

    semantic = semantic_similarity_with_method(field_text, output_text, "auto")[0]
    if semantic >= 0.55 or hit_ratio >= 0.45:
        return 1.0
    if semantic >= 0.32 or hit_ratio >= 0.2:
        return 0.5
    return 0.0


def compute_field_coverage(row: dict[str, str], output_text: str) -> float:
    weighted_scores: list[float] = []
    available_weights: list[float] = []

    for _, weight, columns in FIELD_GROUPS:
        values = [
            _get_case_insensitive(row, column)
            for column in columns
            if _get_case_insensitive(row, column).strip()
        ]
        if not values:
            continue
        field_text = " ".join(values)
        score = _field_score(field_text, output_text)
        weighted_scores.append(score * weight)
        available_weights.append(weight)

    if not available_weights:
        return 0.0
    return sum(weighted_scores) / sum(available_weights)


def compute_format_score(output_text: str) -> float:
    score = 1.0
    if word_count(output_text) < 250 or word_count(output_text) > 600:
        score -= 0.25
    if len([p for p in re.split(r"\n\s*\n", output_text.strip()) if p.strip()]) < 3 or len([p for p in re.split(r"\n\s*\n", output_text.strip()) if p.strip()]) > 8:
        score -= 0.25
    if broken_sentence_count(output_text) > 0:
        score -= 0.25
    if forbidden_formatting_count(output_text) > 0:
        score -= 0.25
    return max(0.0, min(1.0, score))


def csv_failed_output(output_text: str) -> bool:
    text = output_text or ""
    lower = text.lower()
    if not text.strip():
        return True
    if word_count(text) < 150 or word_count(text) > 900:
        return True
    if len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]) < 2:
        return True
    if broken_sentence_count(text) > 0:
        return True
    if forbidden_formatting_count(text) > 0:
        return True
    if re.search(r"^\s*[-*•]\s+", text, flags=re.MULTILINE):
        return True
    if "{" in text or "}" in text or "[" in text or "]" in text:
        return True
    if "|" in text and "---" in text:
        return True
    if re.search(r"\b(according to|the csv record|the dataset|prompt|source record|value-chain record says)\b", lower):
        return True
    return False


def compute_q_score(bertscore: float | None, semantic_similarity_value: float, field_coverage: float, format_score: float) -> float:
    semantic_similarity_value = max(0.0, min(1.0, semantic_similarity_value))
    field_coverage = max(0.0, min(1.0, field_coverage))
    format_score = max(0.0, min(1.0, format_score))
    if bertscore is None:
        return 0.45 * semantic_similarity_value + 0.35 * field_coverage + 0.20 * format_score
    return 0.30 * max(0.0, min(1.0, float(bertscore))) + 0.25 * semantic_similarity_value + 0.30 * field_coverage + 0.15 * format_score
