import os

import pytest

from eventweaver.evaluation.inventory import freeze_inventory
from eventweaver.evaluation.metrics.factscore import FactScoreEvaluator
from eventweaver.evaluation.metrics.source_coverage import SourceCoverageEvaluator


@pytest.mark.skipif(not os.getenv("EVALUATION_JSON_JUDGE_MODEL"), reason="Opt-in model-backed evaluation, may incur costs")
def test_real_automatic_factscore_supported_fixture(evaluation_case):
    config = {"mode": "automatic", "model": os.environ["EVALUATION_JSON_JUDGE_MODEL"], "gamma": 0,
              "base_url": os.getenv("EVALUATION_JSON_JUDGE_BASE_URL", "http://127.0.0.1:11434/v1")}
    if os.getenv("EVALUATION_JSON_JUDGE_KEY_ENV"):
        config["api_key_env"] = os.environ["EVALUATION_JSON_JUDGE_KEY_ENV"]
    result = FactScoreEvaluator().evaluate(evaluation_case[0], config)
    assert result["status"] == "completed", result
    assert result["raw_precision"] == 1, result


@pytest.mark.skipif(not os.getenv("EVALUATION_JSON_JUDGE_MODEL"), reason="Opt-in independent coverage judge")
def test_real_automatic_coverage_supported_fixture(evaluation_case):
    inputs, _, _, evidence, _, _, save = evaluation_case
    inventory = freeze_inventory({"version": "source-inventory-v2", "relevance_criteria": "Construction date",
                                  "reviewed_by": "controlled-test-fixture", "reviewed_at": "2026-11-09",
                                  "units": [{"text": evidence["quote"], "evidence": [evidence], "relevance": "relevant"}]}, inputs.corpus)
    config = {"mode": "automatic", "model": os.environ["EVALUATION_JSON_JUDGE_MODEL"],
              "base_url": os.getenv("EVALUATION_JSON_JUDGE_BASE_URL", "http://127.0.0.1:11434/v1"),
              "inventory": save("inventory.json", inventory)}
    if os.getenv("EVALUATION_JSON_JUDGE_KEY_ENV"):
        config["api_key_env"] = os.environ["EVALUATION_JSON_JUDGE_KEY_ENV"]
    result = SourceCoverageEvaluator().evaluate(inputs, config)
    assert result["status"] == "completed", result
    assert result["score"] == 1, result
