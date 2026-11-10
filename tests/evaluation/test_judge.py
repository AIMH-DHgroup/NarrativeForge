from types import SimpleNamespace

import pytest

from eventweaver.evaluation.judge import JSONJudge


def test_judge_records_exact_request_without_generation_defaults(monkeypatch):
    captured = []

    def post(url, **kwargs):
        captured.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "model": "judge-version", "choices": [{"finish_reason": "stop", "message": {"content": '{"claims": []}'}}],
        })

    monkeypatch.setattr("eventweaver.evaluation.judge.requests.post", post)
    judge = JSONJudge({"model": "judge", "base_url": "http://localhost:11434/v1"})
    assert judge.ask("Instruction", {"narrative": "Text"}) == {"claims": []}
    assert captured[0][0].endswith("/v1/chat/completions")
    assert "frequency_penalty" not in captured[0][1]["json"]
    assert judge.calls[0]["response"]["model"] == "judge-version"


def test_oversized_judge_input_is_not_truncated():
    judge = JSONJudge({"model": "judge", "max_input_chars": 1})
    with pytest.raises(ValueError, match="no input.*truncated"):
        judge.ask("Instruction", {"narrative": "Long input"})
