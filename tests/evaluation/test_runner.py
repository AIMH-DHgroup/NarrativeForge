import json

import pytest

from eventweaver.cli import main
from eventweaver.evaluation import runner
from eventweaver.evaluation.config import METRIC_IDS
from eventweaver.evaluation.sources import load_inputs


@pytest.mark.parametrize("metric", METRIC_IDS)
def test_individual_metrics_produce_only_requested_results(evaluation_case, tmp_path, metric):
    inputs, source, narrative, *_ = evaluation_case
    before = narrative.read_bytes()
    output = tmp_path / "evaluation"
    statuses = runner.run_evaluation(narrative, [source], output, [metric])
    result = json.loads((output / f"{metric}.json").read_text(encoding="utf-8"))
    assert statuses == {metric: "unsupported"}
    assert result["score"] is None
    assert result["narrative_sha256"] == inputs.narrative["sha256"]
    assert result["corpus_sha256"] == inputs.corpus["sha256"]
    assert sorted(path.name for path in output.glob("*.json")) == sorted([
        "evaluation_metadata.json", "sources.json", f"{metric}.json",
    ])
    assert narrative.read_bytes() == before


def test_failure_is_isolated_and_previous_results_are_never_overwritten(evaluation_case, tmp_path, monkeypatch):
    _, source, narrative, *_ = evaluation_case

    class Broken:
        reference, method = "fixture", "fixture"

        def evaluate(self, inputs, config):
            raise ImportError("missing optional dependency")

    monkeypatch.setitem(runner.EVALUATORS, "factscore", Broken)
    output = tmp_path / "evaluation"
    statuses = runner.run_evaluation(narrative, [source], output, ["factscore", "hanna"])
    assert statuses == {"factscore": "failed", "hanna": "unsupported"}
    assert json.loads((output / "factscore.json").read_text())["score"] is None
    with pytest.raises(FileExistsError):
        runner.run_evaluation(narrative, [source], output, ["hanna"])


def test_missing_dependency_during_evaluator_initialization_is_isolated(evaluation_case, tmp_path, monkeypatch):
    _, source, narrative, *_ = evaluation_case

    class MissingDependency:
        reference, method = "fixture", "fixture"

        def __init__(self):
            raise ImportError("optional runtime missing")

    monkeypatch.setitem(runner.EVALUATORS, "alignscore", MissingDependency)
    statuses = runner.run_evaluation(narrative, [source], tmp_path / "out", ["alignscore", "hanna"])
    assert statuses == {"alignscore": "failed", "hanna": "unsupported"}


def test_corpus_identity_is_independent_of_location(evaluation_case, tmp_path):
    inputs, source, narrative, *_ = evaluation_case
    renamed = tmp_path / "renamed.txt"
    renamed.write_bytes(source.read_bytes())
    assert load_inputs(narrative, [renamed]).corpus["sha256"] == inputs.corpus["sha256"]
    renamed.write_text("Changed fact.")
    assert load_inputs(narrative, [renamed]).corpus["sha256"] != inputs.corpus["sha256"]


@pytest.mark.parametrize("content", ["", "# Title only\n", " \n"])
def test_empty_narrative_rejected(evaluation_case, content):
    _, source, narrative, *_ = evaluation_case
    narrative.write_text(content)
    with pytest.raises(ValueError, match="empty"):
        load_inputs(narrative, [source])


def test_invalid_sources_and_identifiers(evaluation_case, tmp_path):
    _, source, narrative, *_ = evaluation_case
    with pytest.raises(FileNotFoundError):
        load_inputs(narrative, [tmp_path / "missing"])
    invalid = tmp_path / "source.exe"
    invalid.write_text("Unsupported")
    with pytest.raises(ValueError, match="Unsupported"):
        load_inputs(narrative, [invalid])
    with pytest.raises(ValueError, match="known metric"):
        runner.run_evaluation(narrative, [source], tmp_path / "out", ["similarity"])
    with pytest.raises(ValueError, match="source corpus"):
        load_inputs(narrative, [narrative])


def test_cli_dispatch_does_not_generate_or_reorganize(evaluation_case, tmp_path, monkeypatch):
    _, source, narrative, *_ = evaluation_case

    def forbidden(*args, **kwargs):
        raise AssertionError("Generation must never run during evaluation")

    monkeypatch.setattr("eventweaver.cli.generate_narratives", forbidden)
    monkeypatch.setattr("eventweaver.cli.reorganize_markdown", forbidden)
    result = main(["evaluate", str(narrative), "--sources", str(source), "--metrics", "hanna,alignscore",
                   "--output-dir", str(tmp_path / "out")])
    assert result == 1  # Honest unsupported outcomes are not completed evaluations.
    assert (tmp_path / "out" / "hanna.json").exists()


def test_missing_narrative_cli_returns_clear_error(tmp_path, capsys):
    assert main(["evaluate", str(tmp_path / "missing.md"), "--sources", str(tmp_path),
                 "--output-dir", str(tmp_path / "out")]) == 1
    assert "existing Markdown" in capsys.readouterr().err


def test_invalid_annotation_path_does_not_block_another_metric(evaluation_case, tmp_path):
    _, source, narrative, *_ = evaluation_case
    statuses = runner.run_evaluation(narrative, [source], tmp_path / "out", ["factscore", "hanna"],
                                     config={"factscore": {"annotations": 123}})
    assert statuses == {"factscore": "failed", "hanna": "unsupported"}


def test_annotation_hash_matches_consumed_snapshot(evaluation_case, tmp_path, monkeypatch):
    inputs, source, narrative, _, _, annotations, save = evaluation_case
    path = save("a.json", annotations)

    class SnapshotEvaluator:
        reference, method = "fixture", "fixture"

        def evaluate(self, supplied_inputs, config):
            from pathlib import Path

            from eventweaver.evaluation.evidence import load_annotations

            Path(path).write_text("{}")  # Simulate file mutation after the runner snapshot.
            loaded = load_annotations(supplied_inputs, config)
            assert loaded["narrative_sha256"] == inputs.narrative["sha256"]
            return {"status": "completed", "score": None}

    monkeypatch.setitem(runner.EVALUATORS, "factscore", SnapshotEvaluator)
    assert runner.run_evaluation(narrative, [source], tmp_path / "out", ["factscore"],
                                 config={"factscore": {"annotations": path}}) == {"factscore": "completed"}
    result = json.loads((tmp_path / "out" / "factscore.json").read_text())
    assert result["input_artifacts"]["annotations"]["data"] == annotations


def test_cli_completes_a_reviewed_human_metric(evaluation_case, tmp_path):
    inputs, source, narrative, evidence, span, annotations, save = evaluation_case
    annotations.update(decomposition_reviewed=True, claims=[{
        "id": "c1", "text": "Construction date is 1200", "narrative_span": span,
        "verdict": "supported", "evidence": [evidence], "explanation": "Explicit source date",
    }])
    config_path = save("config.json", {"factscore": {
        "annotations": save("claims.json", annotations), "gamma": 0,
    }})
    assert main(["evaluate", str(narrative), "--sources", str(source), "--metrics", "factscore",
                 "--config", config_path, "--output-dir", str(tmp_path / "out")]) == 0
    result = json.loads((tmp_path / "out" / "factscore.json").read_text())
    assert result["score"] == 1
    assert result["narrative_sha256"] == inputs.narrative["sha256"]
    assert result["input_artifacts"]["annotations"]["data"] == annotations


def test_human_modes_do_not_silently_ignore_judge_settings(evaluation_case, tmp_path):
    _, source, narrative, *_ = evaluation_case
    with pytest.raises(ValueError, match="require mode"):
        runner.run_evaluation(narrative, [source], tmp_path / "out", ["factscore"],
                              config={"factscore": {"model": "accidentally-configured-judge"}})
