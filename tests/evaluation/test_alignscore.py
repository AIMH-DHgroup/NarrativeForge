from eventweaver.evaluation.metrics.alignscore import AlignScoreEvaluator

import json
from types import SimpleNamespace

import pytest

from eventweaver.evaluation.metrics import alignscore as module


def test_no_checkpoint_download_or_similarity_substitute(evaluation_case):
    result = AlignScoreEvaluator().evaluate(evaluation_case[0], {})
    assert result["status"] == "unsupported"
    assert result["score"] is None


def resources(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test-checkpoint-not-real-weights")
    python = tmp_path / "python.exe"
    python.write_bytes(b"test-executable")
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text("{}")
    return {"python": str(python), "checkpoint": str(checkpoint), "backbone": str(backbone)}


def test_reference_fork_output_contract_and_checkpoint_identity(evaluation_case, tmp_path, monkeypatch):
    config = resources(tmp_path)
    jobs = []

    def run(command, **kwargs):
        assert isinstance(command, list) and "shell" not in kwargs
        jobs.append(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "raw_scores": [[0.75, 0.4, 0.2]], "implementation_revision": module.FACTIVERSE_REVISION,
        }), stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    result = AlignScoreEvaluator().evaluate(evaluation_case[0], config)
    assert result["score"] == 0.75
    assert result["raw_class_scores"] == [0.75, 0.4, 0.2]  # Not a probability distribution to average.
    assert result["checkpoint_sha256"] == module.file_hash(config["checkpoint"])
    assert jobs[0]["narrative"] == evaluation_case[0].narrative["body"]


def test_long_input_is_not_silently_discarded(evaluation_case, tmp_path, monkeypatch):
    config = resources(tmp_path)
    config["max_input_chars"] = 1
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("Must not run truncated evidence"))
    assert AlignScoreEvaluator().evaluate(evaluation_case[0], config)["score"] is None


@pytest.mark.parametrize("output", [[0.7], [[0.7]], [[float("nan"), 0, 0]]])
def test_invalid_scoring_output_is_rejected(evaluation_case, tmp_path, monkeypatch, output):
    config = resources(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({"raw_scores": output, "implementation_revision": module.FACTIVERSE_REVISION}), stderr=""))
    with pytest.raises(ValueError):
        AlignScoreEvaluator().evaluate(evaluation_case[0], config)


def test_missing_checkpoint_and_incompatible_environment(evaluation_case, tmp_path, monkeypatch):
    config = resources(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({"error": "ImportError: optional runtime unavailable"}), stderr=""))
    assert AlignScoreEvaluator().evaluate(evaluation_case[0], config)["status"] == "unsupported"
    config["checkpoint"] = str(tmp_path / "absent.ckpt")
    assert AlignScoreEvaluator().evaluate(evaluation_case[0], config)["status"] == "unsupported"


def test_worker_overrides_saved_checkpoint_backbone_without_hub_loading(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from eventweaver.evaluation.alignscore_worker import make_scorer

    backbone = tmp_path / "roberta-base"
    backbone.mkdir()
    (backbone / "config.json").write_text('{"model_type":"roberta"}')
    calls = []

    class Weights:
        @classmethod
        def load_from_checkpoint(cls, **kwargs):
            calls.append(kwargs)
            return "loaded-local-weights"

    package = ModuleType("alignscore")
    inference = ModuleType("alignscore.inference")
    model = ModuleType("alignscore.model")
    inference.BERTAlignModel = model.BERTAlignModel = Weights

    def alignscore(**kwargs):
        assert inference.BERTAlignModel is not Weights
        loaded = inference.BERTAlignModel(model=kwargs["model"]).load_from_checkpoint(checkpoint_path=kwargs["ckpt_path"], strict=False)
        return loaded

    package.AlignScore = alignscore
    package.inference = inference
    monkeypatch.setitem(sys.modules, "alignscore", package)
    monkeypatch.setitem(sys.modules, "alignscore.inference", inference)
    monkeypatch.setitem(sys.modules, "alignscore.model", model)
    assert make_scorer({"backbone": str(backbone), "checkpoint": "trusted.ckpt", "device": "cpu", "batch_size": 16}) == "loaded-local-weights"
    assert calls == [{"model": str(backbone), "using_pretrained": False, "strict": True, "checkpoint_path": "trusted.ckpt"}]
    assert inference.BERTAlignModel is Weights


def test_long_corpus_is_scored_in_one_reference_document_operation(evaluation_case, tmp_path, monkeypatch):
    from copy import deepcopy

    inputs = deepcopy(evaluation_case[0])
    original = "Long source sentence with a date 1200. " * 500
    inputs.corpus["documents"][0]["pages"][0]["text"] = original
    observed = []

    def run(command, **kwargs):
        observed.append(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "raw_scores": [[0.8, 0.2, 0.1]], "implementation_revision": module.FACTIVERSE_REVISION,
        }))

    monkeypatch.setattr(module.subprocess, "run", run)
    result = AlignScoreEvaluator().evaluate(inputs, resources(tmp_path))
    assert result["score"] == 0.8
    assert len(observed) == 1
    assert observed[0]["context"].split() == original.split()
    assert len(result["contexts_used"]) > 1


def test_empty_alignscore_narrative_is_not_given_a_score(evaluation_case, tmp_path):
    from dataclasses import replace

    inputs = replace(evaluation_case[0], narrative=dict(evaluation_case[0].narrative, body=""))
    with pytest.raises(ValueError, match="non-empty"):
        AlignScoreEvaluator().evaluate(inputs, resources(tmp_path))
