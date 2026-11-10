from eventweaver.evaluation.metrics.ragas_faithfulness import RagasFaithfulnessEvaluator

import pytest

from eventweaver.evaluation.metrics import ragas_faithfulness as module


def test_no_ragas_dependency_or_proxy_score(evaluation_case):
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {})
    assert result["status"] == "unsupported"
    assert result["score"] is None


@pytest.mark.parametrize("verdicts,score", [([1], 1.0), ([0], 0.0), ([1, 0], 0.5)])
def test_official_score_and_independent_statement_trace(evaluation_case, monkeypatch, verdicts, score):
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.4.3")
    monkeypatch.setenv("EVAL_TEST_KEY", "test-not-a-real-key")
    statements = [f"Fact {index}" for index in range(len(verdicts))]

    async def official(config, question, response, contexts):
        assert question == "Describe the church"
        assert contexts == [evaluation_case[0].corpus["documents"][0]["pages"][0]["text"]]
        return score, statements, [{"statement": text, "verdict": verdict, "reason": "Test judgment"}
                                    for text, verdict in zip(statements, verdicts)], {}

    monkeypatch.setattr(module, "run_official", official)
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {"model": "judge", "api_key_env": "EVAL_TEST_KEY"})
    assert result["score"] == score
    assert result["statements"] == statements
    assert result["contexts_used"][0]["document_id"] == evaluation_case[3]["document_id"]


def test_context_chunking_preserves_all_evidence(evaluation_case):
    from eventweaver.evaluation.contexts import source_contexts

    contexts = source_contexts(evaluation_case[0].corpus, chunk_chars=5)
    assert "".join(item["text"] for item in contexts) == evaluation_case[0].corpus["documents"][0]["pages"][0]["text"]
    assert len(contexts) > 1
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {"model": "judge", "max_input_chars": 1})
    assert result["status"] == "unsupported"
    assert result["score"] is None


def test_dependency_error_is_actionable(evaluation_case, monkeypatch):
    def missing(name):
        raise module.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(module.importlib.metadata, "version", missing)
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {"model": "judge"})
    assert result["status"] == "unsupported"


def test_missing_verdict_does_not_produce_inflated_score(evaluation_case, monkeypatch):
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.4.3")
    monkeypatch.setenv("EVAL_TEST_KEY", "test")

    async def official(*args):
        return 1.0, ["Fact 1", "Fact 2"], [{"statement": "Fact 1", "verdict": 1}], {}

    monkeypatch.setattr(module, "run_official", official)
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {"model": "judge", "api_key_env": "EVAL_TEST_KEY"})
    assert result["status"] == "failed"
    assert result["score"] is None


def test_ragas_failure_retains_statements_and_effective_parameters(evaluation_case, monkeypatch):
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.4.3")
    monkeypatch.setenv("EVAL_TEST_KEY", "test")

    async def official(*args):
        return None, ["Fact extracted"], [], {"effective_model_parameters": {"temperature": 1.0}, "error": "NLI stage timed out"}

    monkeypatch.setattr(module, "run_official", official)
    result = RagasFaithfulnessEvaluator().evaluate(evaluation_case[0], {"model": "o3", "api_key_env": "EVAL_TEST_KEY"})
    assert result["status"] == "failed"
    assert result["statements"] == ["Fact extracted"]
    assert result["runtime_configuration"]["effective_model_parameters"]["temperature"] == 1.0
    assert result["contexts_used"]


def test_runtime_records_effective_factory_parameters_and_second_stage_failure(monkeypatch):
    import asyncio
    import sys
    from types import ModuleType, SimpleNamespace

    captured = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def factory(model, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(_map_provider_params=lambda: {"temperature": 1.0, "max_completion_tokens": 4096})

    class Metric:
        extracted_statements = ["An extracted fact"]
        verdict_records = []

        async def ascore(self, **kwargs):
            raise TimeoutError("NLI failure")

    openai = ModuleType("openai")
    openai.AsyncOpenAI = Client
    ragas = ModuleType("ragas")
    llms = ModuleType("ragas.llms")
    llms.llm_factory = factory
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)
    monkeypatch.setenv("TEST_JUDGE_KEY", "placeholder")
    monkeypatch.setattr(module, "make_metric", lambda llm: Metric())
    score, statements, verdicts, runtime = asyncio.run(module.run_official(
        {"model": "o3", "api_key_env": "TEST_JUDGE_KEY"}, "question", "answer", ["context"]))
    assert captured[0]["temperature"] == 0.0  # Requested setting.
    assert runtime["effective_model_parameters"]["temperature"] == 1.0
    assert score is None and statements == ["An extracted fact"] and verdicts == []
    assert "NLI failure" in runtime["error"]
