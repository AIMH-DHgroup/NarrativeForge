from __future__ import annotations

import statistics

from .text_metrics import bertscore_f1, semantic_similarity_with_method


def _clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def nrs_no_r(bertscore: float | None, semantic_similarity_value: float) -> float:
    """Single-run Narrative Reliability Score without robustness."""

    bertscore = _clip01(bertscore)
    semantic_similarity_value = max(0.0, min(1.0, semantic_similarity_value))
    if bertscore is None:
        return 100.0 * semantic_similarity_value
    return 100.0 * (0.50 * bertscore + 0.50 * semantic_similarity_value)


def nrs(mean_bertscore: float | None, mean_semantic_similarity: float, robustness: float) -> float:
    """Case/model Narrative Reliability Score.

    Formula:
    - with BERTScore: 100 * (0.35 * BERTScore + 0.35 * semantic + 0.30 * R)
    - without BERTScore: 100 * (0.70 * semantic + 0.30 * R)
    """

    robustness = max(0.0, min(1.0, robustness))
    mean_semantic_similarity = max(0.0, min(1.0, mean_semantic_similarity))
    mean_bertscore = _clip01(mean_bertscore)
    if mean_bertscore is None:
        return 100.0 * (0.70 * mean_semantic_similarity + 0.30 * robustness)
    return 100.0 * (0.35 * mean_bertscore + 0.35 * mean_semantic_similarity + 0.30 * robustness)


def robustness_from_runs(run_rows: list[dict]) -> dict:
    """Compute R_stab, R_struct, and R_fail from repeated runs."""

    if len(run_rows) < 2:
        return {"robustness_available": False, "R_stab": None, "R_struct": None, "R_fail": None, "R": None}

    qualities: list[float] = []
    paragraph_counts: list[float] = []
    failed_runs = 0

    for row in run_rows:
        bert = row.get("bertscore_f1")
        semantic = float(row.get("semantic_similarity") or 0.0)
        if bert is None or bert == "":
            qualities.append(semantic)
        else:
            qualities.append(0.50 * max(0.0, min(1.0, float(bert))) + 0.50 * max(0.0, min(1.0, semantic)))
        paragraph_counts.append(float(row.get("paragraph_count") or 0.0))
        if bool(row.get("failed")):
            failed_runs += 1

    sigma_quality = statistics.pstdev(qualities) if len(qualities) > 1 else 0.0
    r_stab = 1.0 - min(sigma_quality / 0.15, 1.0)

    mean_paragraphs = statistics.mean(paragraph_counts) if paragraph_counts else 0.0
    std_paragraphs = statistics.pstdev(paragraph_counts) if len(paragraph_counts) > 1 else 0.0
    r_struct = 1.0 - min(std_paragraphs / max(mean_paragraphs, 1.0), 1.0)

    r_fail = 1.0 - (failed_runs / len(run_rows))
    r = 0.50 * r_stab + 0.25 * r_struct + 0.25 * r_fail

    return {
        "robustness_available": True,
        "R_stab": max(0.0, min(1.0, r_stab)),
        "R_struct": max(0.0, min(1.0, r_struct)),
        "R_fail": max(0.0, min(1.0, r_fail)),
        "R": max(0.0, min(1.0, r)),
    }


def single_run_quality(bertscore: float | None, semantic_similarity_value: float) -> float:
    return 0.50 * (bertscore if bertscore is not None else semantic_similarity_value) + 0.50 * semantic_similarity_value
