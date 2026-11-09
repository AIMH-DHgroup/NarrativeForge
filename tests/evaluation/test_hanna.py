import hashlib

import pytest

from eventweaver.evaluation.metrics.hanna import HannaEvaluator


def test_human_hanna_retains_independent_scores(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    scores = dict.fromkeys(("coherence", "relevance", "empathy", "surprise", "engagement", "complexity"), 4)
    annotations.update(rubric_version="hanna-2022-table7-v1",
                       prompt_sha256=hashlib.sha256(inputs.original_prompt.encode()).hexdigest(),
                       ratings=[{"annotator": "r1", "annotated_at": "2026-11-09", "scores": scores}])
    result = HannaEvaluator().evaluate(inputs, {"annotations": save("hanna.json", annotations)})
    assert result["status"] == "completed"
    assert result["dimension_scores"] == scores
    assert result["score"] is None
    assert all(len(value["anchors"]) == 5 for value in result["rubric"]["dimensions"].values())


def test_hanna_partial_and_out_of_range(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    scores = dict.fromkeys(("coherence", "relevance", "empathy", "surprise", "engagement", "complexity"))
    scores["coherence"] = 3
    annotations.update(rubric_version="hanna-2022-table7-v1",
                       prompt_sha256=hashlib.sha256(inputs.original_prompt.encode()).hexdigest(),
                       ratings=[{"annotator": "r1", "annotated_at": "2026-11-09", "scores": scores}])
    path = save("hanna.json", annotations)
    assert HannaEvaluator().evaluate(inputs, {"annotations": path})["status"] == "partial"
    scores["coherence"] = 6
    with pytest.raises(ValueError, match="1 to 5"):
        HannaEvaluator().evaluate(inputs, {"annotations": save("bad.json", annotations)})


def test_hanna_rejects_unbound_rubric(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    annotations["rubric_version"] = "other-rubric"
    with pytest.raises(ValueError, match="matching rubric"):
        HannaEvaluator().evaluate(inputs, {"annotations": save("bad.json", annotations)})
