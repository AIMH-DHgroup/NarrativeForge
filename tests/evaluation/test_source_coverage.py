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


def test_v2_inventory_requires_relevance_and_detects_source_changes(evaluation_case):
    inputs, _, _, evidence, *_ = evaluation_case
    draft = {"version": "source-inventory-v2", "relevance_criteria": "History", "reviewed_by": "r", "reviewed_at": "now",
             "units": [{"text": "Date", "evidence": [evidence]}]}
    with pytest.raises(ValueError, match="reviewed relevance"):
        freeze_inventory(draft, inputs.corpus)
    draft["units"][0]["relevance"] = "relevant"
    inventory = freeze_inventory(draft, inputs.corpus)
    assert inventory["units"][0]["relevance"] == "relevant"
    other_corpus = dict(inputs.corpus, sha256="modified")
    with pytest.raises(ValueError, match="another source"):
        validate_inventory(inventory, other_corpus)
    draft["units"].append(dict(draft["units"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_inventory(draft, inputs.corpus)


def test_source_only_inventory_proposals_never_receive_narrative(evaluation_case, monkeypatch):
    from eventweaver.evaluation.metrics import source_coverage_automatic as module

    class Judge:
        def __init__(self, config):
            self.calls = []

        def ask(self, instruction, data):
            assert "narrative" not in data
            self.calls.append(data)
            return {"units": [{"text": "Date is 1200", "quote": data["source_passage"]["text"]}]}

    monkeypatch.setattr(module, "JSONJudge", Judge)
    draft = module.prepare_inventory(evaluation_case[0].corpus, "Construction history", {"model": "judge"})
    assert draft["frozen"] is False
    assert draft["units"][0]["relevance"] == "uncertain"
    assert draft["units"][0]["evidence"][0] == evaluation_case[3]
    with pytest.raises(ValueError):
        freeze_inventory(draft, evaluation_case[0].corpus)  # No human approval.


@pytest.mark.parametrize("verdict,score,status", [("covered", 1, "completed"), ("not_covered", 0, "completed"),
                                                ("uncertain", 0, "uncertain")])
def test_automatic_coverage_matching_is_independent_and_traceable(evaluation_case, monkeypatch, verdict, score, status):
    from eventweaver.evaluation.metrics import source_coverage_automatic as module

    inputs, _, _, evidence, span, _, save = evaluation_case
    inventory = freeze_inventory({"version": "source-inventory-v2", "relevance_criteria": "History", "reviewed_by": "r", "reviewed_at": "now",
                                  "units": [{"text": "Date", "evidence": [evidence], "relevance": "relevant"}]}, inputs.corpus)

    class Judge:
        def __init__(self, config):
            self.calls = []

        def ask(self, instruction, data):
            self.calls.append(data)
            assert data["source_information_unit"]["id"] == inventory["units"][0]["id"]
            return {"verdict": verdict, "explanation": "Complete meaning / absent / partial meaning",
                    "narrative_evidence": [{"quote": span["quote"]}] if verdict == "covered" else []}

    monkeypatch.setattr(module, "JSONJudge", Judge)
    result = SourceCoverageEvaluator().evaluate(inputs, {"mode": "automatic", "model": "judge", "inventory": save("i.json", inventory)})
    assert result["score"] == score
    assert result["status"] == status
    assert result["inventory"]["sha256"] == inventory["sha256"]
    if verdict == "covered":
        assert result["annotations"]["decisions"][0]["narrative_evidence"] == [span]


def test_invalid_automatic_coverage_preserves_trace(evaluation_case, monkeypatch):
    from eventweaver.evaluation.metrics import source_coverage_automatic as module

    inputs, _, _, evidence, _, _, save = evaluation_case
    inventory = freeze_inventory({"relevance_criteria": "History", "reviewed_by": "r", "reviewed_at": "now",
                                  "units": [{"text": "Date", "evidence": [evidence]}]}, inputs.corpus)

    class Judge:
        def __init__(self, config):
            self.calls = []

        def ask(self, instruction, data):
            self.calls.append(data)
            return {"verdict": "covered", "explanation": "Claimed covered without any citation"}

    monkeypatch.setattr(module, "JSONJudge", Judge)
    result = SourceCoverageEvaluator().evaluate(inputs, {"inventory": save("i.json", inventory), "mode": "automatic", "model": "judge"})
    assert result["status"] == "failed"
    assert result["score"] is None
    assert len(result["model_calls"]) == 1
