from eventweaver.evaluation.metrics.alignscore import AlignScoreEvaluator


def test_no_checkpoint_download_or_similarity_substitute(evaluation_case):
    result = AlignScoreEvaluator().evaluate(evaluation_case[0], {})
    assert result["status"] == "unsupported"
    assert result["score"] is None
