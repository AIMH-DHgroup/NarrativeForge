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


@pytest.mark.parametrize("pairs,expected", [([(1, 1), (5, 5)], 1.0), ([(1, 5), (5, 1)], -1.0),
                                          ([(1, 1), (2, 3), (3, 2), (4, 4)], 0.8), ([(3, 3), (3, 3)], None),
                                          ([(1, 1)], None)])
def test_ordinal_agreement_known_contingency_tables(pairs, expected):
    from eventweaver.evaluation.metrics.hanna_agreement import quadratic_kappa

    actual = quadratic_kappa(pairs)
    assert actual is None if expected is None else actual == pytest.approx(expected)


def test_agreement_across_narratives_not_across_dimensions(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    rubric = HannaEvaluator().evaluate(inputs, {})["rubric"]
    dimensions = tuple(rubric["dimensions"])
    panel = dict(annotations, rubric_version=rubric["version"], items=[])
    for digest, score in [(inputs.narrative["sha256"], 1), ("a" * 64, 5)]:
        panel["items"].append({"narrative_sha256": digest, "ratings": [
            {"annotator": rater, "annotated_at": "now", "scores": dict.fromkeys(dimensions, score)}
            for rater in ("r1", "r2")
        ]})
    result = HannaEvaluator().evaluate(inputs, {"agreement_annotations": save("panel.json", panel)})
    assert result["score"] is None
    assert result["status"] == "partial"
    assert all(item["value"] == 1 for item in result["agreement"]["dimensions"].values())
    assert all(item["paired_narratives"] == 2 for item in result["agreement"]["dimensions"].values())
    panel["items"][0]["ratings"][1]["scores"]["empathy"] = None
    result = HannaEvaluator().evaluate(inputs, {"agreement_annotations": save("panel2.json", panel)})
    assert result["agreement"]["dimensions"]["empathy"]["value"] is None


def test_hanna_multiple_raters_dimension_means(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    dimensions = tuple(HannaEvaluator().evaluate(inputs, {})["rubric"]["dimensions"])
    annotations.update(rubric_version="hanna-2022-table7-v1",
                       prompt_sha256=hashlib.sha256(inputs.original_prompt.encode()).hexdigest(),
                       ratings=[{"annotator": f"r{score}", "annotated_at": "now", "scores": dict.fromkeys(dimensions, score)}
                                for score in (1, 3, 5)])
    result = HannaEvaluator().evaluate(inputs, {"annotations": save("a.json", annotations)})
    assert all(score == 3 for score in result["dimension_scores"].values())
    assert result["score"] is None


def test_hanna_missing_dimensions_are_not_imputed(evaluation_case):
    inputs, _, _, _, _, annotations, save = evaluation_case
    annotations.update(rubric_version="hanna-2022-table7-v1",
                       prompt_sha256=hashlib.sha256(inputs.original_prompt.encode()).hexdigest(),
                       ratings=[{"annotator": "r1", "annotated_at": "now", "scores": {"coherence": 5}}])
    with pytest.raises(ValueError, match="six rubric dimensions"):
        HannaEvaluator().evaluate(inputs, {"annotations": save("missing.json", annotations)})
