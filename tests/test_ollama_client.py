from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from eventweaver.ollama_client import generate_ollama, list_ollama_models, preflight_ollama, resolve_ollama_host


class OllamaClientTests(unittest.TestCase):
    def test_resolve_ollama_host_defaults_and_normalizes(self) -> None:
        self.assertEqual(resolve_ollama_host(None), "http://localhost:11434")
        self.assertEqual(resolve_ollama_host("localhost:11434/"), "http://localhost:11434")
        self.assertEqual(resolve_ollama_host("http://example.test:11434/"), "http://example.test:11434")

    def test_list_models_reports_connection_refused_clearly(self) -> None:
        with patch("eventweaver.ollama_client.requests.get", side_effect=requests.exceptions.ConnectionError("[WinError 10061] refused")):
            with self.assertRaisesRegex(RuntimeError, "Windows refused the connection to Ollama"):
                list_ollama_models("http://localhost:11434")

    def test_preflight_reports_missing_models(self) -> None:
        response = Mock()
        response.json.return_value = {"models": [{"name": "qwen3:4b"}]}
        response.raise_for_status.return_value = None
        with patch("eventweaver.ollama_client.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "qwen3:32b"):
                preflight_ollama(["qwen3:4b", "qwen3:32b"], "http://localhost:11434")

    def test_generate_does_not_retry_connection_refused(self) -> None:
        with patch("eventweaver.ollama_client.requests.post", side_effect=requests.exceptions.ConnectionError("[WinError 10061] refused")) as post:
            result = generate_ollama("prompt", "qwen3:4b", ollama_host="http://localhost:11434", retries=3)
        self.assertIn("Windows refused the connection to Ollama", result.error or "")
        self.assertEqual(post.call_count, 1)

    def test_generate_retries_transient_5xx(self) -> None:
        first = Mock(status_code=500)
        second = Mock(status_code=200)
        second.json.return_value = {"response": "ok"}
        second.raise_for_status.return_value = None
        with patch("eventweaver.ollama_client.requests.post", side_effect=[first, second]) as post, patch("eventweaver.ollama_client.time.sleep"):
            result = generate_ollama("prompt", "qwen3:4b", ollama_host="http://localhost:11434", retries=1)
        self.assertEqual(result.text, "ok")
        self.assertEqual(post.call_count, 2)

    def test_generate_can_request_bounded_json_output(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"response": "{}"}
        response.raise_for_status.return_value = None
        with patch("eventweaver.ollama_client.requests.post", return_value=response) as post:
            result = generate_ollama(
                "prompt",
                "qwen3:8b",
                max_new_tokens=512,
                json_mode=True,
                think=False,
            )
        self.assertEqual(result.text, "{}")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 512)

    def test_generate_omits_penalties_by_default(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"response": "ok"}
        response.raise_for_status.return_value = None
        with patch("eventweaver.ollama_client.requests.post", return_value=response) as post:
            generate_ollama("prompt", "qwen3:8b")
        options = post.call_args.kwargs["json"]["options"]
        self.assertNotIn("frequency_penalty", options)
        self.assertNotIn("presence_penalty", options)

    def test_generate_rejects_explicit_zero_for_unsupported_ollama_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not expose"):
            generate_ollama("prompt", "qwen3:8b", frequency_penalty=0.0)


if __name__ == "__main__":
    unittest.main()
