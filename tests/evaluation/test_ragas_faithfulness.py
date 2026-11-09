from eventweaver.evaluation.metrics.ragas_faithfulness import RagasFaithfulnessEvaluator


def test_no_ragas_dependency_or_proxy_score(evaluation_case):
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {})
    assert result["status"] == "unsupported"
    assert result["score"] is None
