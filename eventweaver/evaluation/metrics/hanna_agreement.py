"""Ordinal two-rater agreement across narratives, separately per HANNA dimension."""
from collections import Counter

from ..evidence import require_text_fields


def quadratic_kappa(pairs):
    if len(pairs) < 2:
        return None
    n = len(pairs)
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum((a - b) ** 2 / 16 for a, b in pairs) / n
    expected = sum(count_a * count_b * (a - b) ** 2 / 16
                   for a, count_a in first.items() for b, count_b in second.items()) / n ** 2
    return 1 - observed / expected if expected else None


def agreement_report(panel, inputs, rubric):
    require_text_fields(panel, ("annotator", "annotated_at"))
    if panel.get("corpus_sha256") != inputs.corpus["sha256"] or panel.get("rubric_version") != rubric["version"]:
        raise ValueError("HANNA agreement panel corpus or rubric mismatch")
    items = panel.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Agreement panel requires independently rated narrative items")
    rows = {}
    annotators = set()
    for item in items:
        digest = item.get("narrative_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) or digest in rows:
            raise ValueError("Agreement panel needs unique narrative SHA-256 identities")
        ratings = {}
        for rating in item.get("ratings", []):
            require_text_fields(rating, ("annotator", "annotated_at"))
            rater = rating["annotator"]
            if rater in ratings:
                raise ValueError("Duplicate rater for a narrative in agreement panel")
            scores = rating.get("scores")
            if not isinstance(scores, dict) or set(scores) != set(rubric["dimensions"]):
                raise ValueError("Agreement ratings require all six dimensions (null for missing)")
            if any(value is not None and (type(value) is not int or not 1 <= value <= 5) for value in scores.values()):
                raise ValueError("Agreement scores must be ordinal integers 1..5 or null")
            ratings[rater] = scores
            annotators.add(rater)
        rows[digest] = ratings
    if inputs.narrative["sha256"] not in rows:
        raise ValueError("Agreement panel must include the evaluated narrative")
    if len(annotators) != 2:
        raise ValueError("Quadratic weighted Cohen kappa requires exactly two annotators; multi-rater alpha is not implemented")
    first, second = sorted(annotators)
    dimensions = {}
    for dimension in rubric["dimensions"]:
        pairs = []
        for ratings in rows.values():
            a, b = ratings.get(first, {}).get(dimension), ratings.get(second, {}).get(dimension)
            if a is not None and b is not None:
                pairs.append((a, b))
        value = quadratic_kappa(pairs)
        dimensions[dimension] = {"value": value, "paired_narratives": len(pairs),
                                 "excluded_incomplete_pairs": len(rows) - len(pairs),
                                 "status": "completed" if value is not None else "uncertain"}
    return {"statistic": "quadratic_weighted_cohen_kappa", "version": "hanna-agreement-v1",
            "annotators": [first, second], "dimensions": dimensions, "panel": panel,
            "assumptions": ["Exactly two fixed raters score the same independent narrative units on ordered 1..5 categories.",
                            "Quadratic weights assume squared ordinal-category distance; no interval-scale claim is made about the underlying quality.",
                            "Complete pairs are used separately by dimension; missingness may bias agreement.",
                            "At least two paired narratives and nonzero expected disagreement are required; otherwise agreement is undefined.",
                            "No cross-dimension or cross-metric agreement average is computed; this is not the original HANNA ICC protocol."]}
