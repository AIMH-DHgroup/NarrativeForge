import math

import pytest

from eventweaver.evaluation.metrics.factscore import FactScoreEvaluator


def test_human_atomic_precision_and_reference_brevity_penalty(evaluation_case):
    inputs, _, _, evidence, span, annotations, save = evaluation_case
    claim = {"id": "claim-1", "text": "The church was built in 1200.", "narrative_span": span,
             "verdict": "supported", "evidence": [evidence], "explanation": "Explicit date in source"}
    annotations.update(decomposition_reviewed=True, claims=[claim])
    path = save("claims.json", annotations)
    result = FactScoreEvaluator().evaluate(inputs, {"annotations": path})
    assert result["score"] == pytest.approx(math.exp(-9))
    assert result["raw_precision"] == 1
    assert result["annotations"]["claims"][0]["evidence"] == [evidence]
    assert FactScoreEvaluator().evaluate(inputs, {"annotations": path, "gamma": 0})["score"] == 1


def test_unverifiable_is_not_silently_excluded(evaluation_case):
    inputs, _, _, _, span, annotations, save = evaluation_case
    annotations.update(decomposition_reviewed=True, claims=[{
        "id": "c1", "text": "Claim", "narrative_span": span, "verdict": "unverifiable",
        "explanation": "No determination possible", "evidence": [],
    }])
    result = FactScoreEvaluator().evaluate(inputs, {"annotations": save("claims.json", annotations)})
    assert result["status"] == "uncertain"
    assert result["score"] is None
    assert result["precision_bounds"] == [0, 1]
    assert result["denominator"] == 1


def test_claim_evidence_cannot_reference_another_corpus(evaluation_case):
    inputs, _, _, evidence, span, annotations, save = evaluation_case
    evidence["document_id"] = "forged"
    annotations.update(claims=[{"id": "c1", "text": "Claim", "narrative_span": span,
                               "verdict": "supported", "explanation": "x", "evidence": [evidence]}])
    with pytest.raises(ValueError, match="Unknown evidence"):
        FactScoreEvaluator().evaluate(inputs, {"annotations": save("claims.json", annotations)})


def test_duplicate_claims_cannot_inflate_length_penalty(evaluation_case):
    inputs, _, _, evidence, span, annotations, save = evaluation_case
    annotations.update(decomposition_reviewed=True, claims=[
        {"id": str(index), "text": "Construction date is 1200", "narrative_span": span,
         "verdict": "supported", "explanation": "Source date", "evidence": [evidence]}
        for index in range(10)
    ])
    with pytest.raises(ValueError, match="Duplicate atomic claim"):
        FactScoreEvaluator().evaluate(inputs, {"annotations": save("duplicates.json", annotations)})
