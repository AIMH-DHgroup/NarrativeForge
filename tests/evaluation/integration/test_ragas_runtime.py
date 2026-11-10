"""Opt-in tests: real structured-output judge, credentials and network required."""
import os

import pytest

from eventweaver.evaluation.metrics.ragas_faithfulness import RagasFaithfulnessEvaluator


@pytest.mark.skipif(not os.getenv("EVALUATION_RAGAS_MODEL"), reason="Set EVALUATION_RAGAS_MODEL to opt in")
def test_real_ragas_reference_fixture(evaluation_case):
    pytest.importorskip("ragas")
    config = {"model": os.environ["EVALUATION_RAGAS_MODEL"]}
    if os.getenv("EVALUATION_RAGAS_BASE_URL"):
        config["base_url"] = os.environ["EVALUATION_RAGAS_BASE_URL"]
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], config)
    assert result["status"] == "completed", result
    assert result["score"] == 1.0, result  # A controlled supported date, not a domain validation.
