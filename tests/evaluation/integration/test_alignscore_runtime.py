import os

import pytest

from eventweaver.evaluation.metrics.alignscore import AlignScoreEvaluator


@pytest.mark.skipif(not os.getenv("EVALUATION_ALIGNSCORE_PYTHON"), reason="Requires isolated AlignScore assets")
def test_real_alignscore_repeatable_fixture(evaluation_case):
    config = {"python": os.environ["EVALUATION_ALIGNSCORE_PYTHON"],
              "checkpoint": os.environ["EVALUATION_ALIGNSCORE_CHECKPOINT"],
              "backbone": os.environ["EVALUATION_ALIGNSCORE_BACKBONE"], "device": "cpu"}
    first = AlignScoreEvaluator().evaluate(evaluation_case[0], config)
    second = AlignScoreEvaluator().evaluate(evaluation_case[0], config)
    assert first["status"] == second["status"] == "completed", (first, second)
    assert first["score"] == pytest.approx(second["score"], abs=1e-6)
