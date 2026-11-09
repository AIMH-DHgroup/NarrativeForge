from copy import deepcopy

import pytest

from eventweaver.evaluation.inventory import freeze_inventory, validate_inventory
from eventweaver.evaluation.metrics.source_coverage import SourceCoverageEvaluator


def test_frozen_inventory_reused_without_narrative_dependency(evaluation_case):
    inputs, _, _, evidence, span, annotations, save = evaluation_case
    draft = {"relevance_criteria": "Facts about church construction", "reviewed_by": "curator",
             "reviewed_at": "2026-11-09", "units": [{"text": "Construction date is 1200", "evidence": [evidence]}]}
    inventory = freeze_inventory(draft, inputs.corpus)
    assert freeze_inventory(draft, inputs.corpus) == inventory
    path = save("inventory.json", inventory)
    annotations.update(inventory_sha256=inventory["sha256"], decisions=[{
        "unit_id": inventory["units"][0]["id"], "verdict": "covered",
        "narrative_evidence": [span], "explanation": "Date is retained",
    }])
    config = {"inventory": path, "annotations": save("coverage.json", annotations)}
    evaluator = SourceCoverageEvaluator()
    assert evaluator.evaluate(inputs, config)["score"] == 1
    assert evaluator.evaluate(inputs, config)["inventory"]["sha256"] == inventory["sha256"]
    changed = deepcopy(inventory)
    changed["units"][0]["text"] = "altered"
    with pytest.raises(ValueError, match="hash"):
        validate_inventory(changed, inputs.corpus)


def test_coverage_uncertainty_and_missing_decisions(evaluation_case):
    inputs, _, _, evidence, _, annotations, save = evaluation_case
    inventory = freeze_inventory({"relevance_criteria": "History", "reviewed_by": "r", "reviewed_at": "now",
                                  "units": [{"text": "Date", "evidence": [evidence]}]}, inputs.corpus)
    annotations.update(inventory_sha256=inventory["sha256"], decisions=[])
    config = {"inventory": save("inventory.json", inventory), "annotations": save("a.json", annotations)}
    result = SourceCoverageEvaluator().evaluate(inputs, config)
    assert result["score"] is None
    assert result["status"] == "partial"
    annotations["decisions"] = [{"unit_id": inventory["units"][0]["id"], "verdict": "uncertain",
                                 "explanation": "Ambiguous paraphrase"}]
    config["annotations"] = save("b.json", annotations)
    result = SourceCoverageEvaluator().evaluate(inputs, config)
    assert result["score"] == 0
    assert result["status"] == "uncertain"
    assert result["coverage_bounds"] == [0, 1]


@pytest.mark.parametrize("field,value", [("relevance_criteria", "  "), ("reviewed_by", True), ("reviewed_at", 123)])
def test_inventory_requires_textual_review_provenance(evaluation_case, field, value):
    inputs, _, _, evidence, *_ = evaluation_case
    draft = {"relevance_criteria": "History", "reviewed_by": "r", "reviewed_at": "now",
             "units": [{"text": "Date", "evidence": [evidence]}]}
    draft[field] = value
    with pytest.raises(ValueError, match="text fields"):
        freeze_inventory(draft, inputs.corpus)
