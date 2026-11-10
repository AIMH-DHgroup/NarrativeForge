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


def automated_judge(monkeypatch, extraction, verdict):
    from eventweaver.evaluation.metrics import factscore_automatic

    class Judge:
        def __init__(self, config):
            self.calls = []

        def ask(self, instruction, data):
            self.calls.append({"instruction": instruction, "data": data})
            return extraction if len(self.calls) == 1 else verdict(data)

    monkeypatch.setattr(factscore_automatic, "JSONJudge", Judge)


@pytest.mark.parametrize("verdict,score,status", [("supported", 1, "completed"), ("unsupported", 0, "completed"),
                                                ("unverifiable", None, "uncertain")])
def test_automatic_atomic_extraction_and_verification(evaluation_case, monkeypatch, verdict, score, status):
    quote = evaluation_case[4]["quote"]
    extraction = {"claims": [{"text": "The church was built in 1200.", "quote": quote}]}

    def decision(data):
        context = data["contexts"][0]
        return {"verdict": verdict, "explanation": "Exact assertion or documented contradiction / ambiguity",
                "evidence": [{"context_id": context["id"], "quote": context["text"]}]}

    automated_judge(monkeypatch, extraction, decision)
    result = FactScoreEvaluator().evaluate(evaluation_case[0], {"mode": "automatic", "model": "judge", "gamma": 0})
    assert result["score"] == score
    assert result["status"] == status
    assert result["mode"] == "automatic"
    assert result["claims"][0]["evidence"][0] == evaluation_case[3]
    assert result["claims"][0]["id"].startswith("A-")


def test_automatic_claim_deduplication_and_empty_sets(evaluation_case, monkeypatch):
    item = {"text": "Built in 1200", "quote": evaluation_case[4]["quote"]}
    automated_judge(monkeypatch, {"claims": [item, item]}, lambda data: {
        "verdict": "unsupported", "explanation": "No support", "evidence": [],
    })
    result = FactScoreEvaluator().evaluate(evaluation_case[0], {"mode": "automatic", "model": "judge"})
    assert result["denominator"] == 1
    assert len(result["duplicates_removed"]) == 1
    automated_judge(monkeypatch, {"claims": []}, lambda data: None)
    assert FactScoreEvaluator().evaluate(evaluation_case[0], {"mode": "automatic", "model": "judge"})["score"] is None


def test_automatic_claim_evidence_cannot_be_invented(evaluation_case, monkeypatch):
    automated_judge(monkeypatch, {"claims": [{"text": "Built in 1200", "quote": "Not in the narrative"}]}, lambda data: None)
    result = FactScoreEvaluator().evaluate(evaluation_case[0], {"mode": "automatic", "model": "judge"})
    assert result["status"] == "failed"
    assert result["score"] is None
    assert "not present" in result["error"]
