from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from eventweaver.config import GenerateConfig
from eventweaver.ollama_client import generate_ollama
from eventweaver.providers.factory import build_provider


def test_penalties_are_none_by_default() -> None:
    config = GenerateConfig()
    assert config.frequency_penalty is None
    assert config.presence_penalty is None


def test_ollama_payload_omits_unset_penalties() -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"response": "{}"}
    response.raise_for_status.return_value = None
    with patch("requests.post", return_value=response) as post:
        generate_ollama("prompt", "model")
    options = post.call_args.kwargs["json"]["options"]
    assert "frequency_penalty" not in options
    assert "presence_penalty" not in options


def test_explicit_penalties_are_rejected_by_current_backends() -> None:
    with pytest.raises(ValueError, match="does not expose"):
        build_provider("ollama", "model", frequency_penalty=0.0)
    with pytest.raises(ValueError, match="does not expose"):
        build_provider("huggingface-local", "model", presence_penalty=0.8)


def test_explicit_zero_is_not_treated_as_omitted() -> None:
    with pytest.raises(ValueError, match="does not expose"):
        generate_ollama("prompt", "model", frequency_penalty=0.0)
