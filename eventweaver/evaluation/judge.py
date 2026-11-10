"""Generic JSON inference transport; no shared extraction or scoring logic."""
import json
import os

import requests

from .contexts import positive_integer


class JSONJudge:
    def __init__(self, config):
        if not isinstance(config.get("model"), str) or not config["model"].strip():
            raise ValueError("Automatic evaluation requires a judge model")
        self.config = config
        self.calls = []

    def ask(self, instruction, data):
        payload = {"model": self.config["model"], "temperature": 0.0,
                   "max_tokens": positive_integer(self.config, "max_tokens", 4096),
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": instruction},
                                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]}
        if len(json.dumps(payload, ensure_ascii=False)) > positive_integer(self.config, "max_input_chars", 60000):
            raise ValueError("Full judge input exceeds max_input_chars; no input is silently truncated")
        headers = {"Content-Type": "application/json"}
        if self.config.get("api_key_env"):
            key = os.getenv(self.config["api_key_env"])
            if not key:
                raise ValueError("Configured judge API-key environment variable is missing")
            headers["Authorization"] = f"Bearer {key}"
        record = {"request": payload, "response": None}
        self.calls.append(record)
        response = requests.post(self.config.get("base_url", "http://127.0.0.1:11434/v1").rstrip("/") + "/chat/completions",
                                 json=payload, headers=headers,
                                 timeout=positive_integer(self.config, "timeout_seconds", 300))
        response.raise_for_status()
        body = response.json()
        try:
            json.dumps(body, allow_nan=False)
        except (ValueError, TypeError):
            record["response"] = repr(body)
            raise ValueError("Judge returned non-finite or non-JSON response data") from None
        record["response"] = body
        if body["choices"][0].get("finish_reason") != "stop":
            raise ValueError("Judge output did not finish normally; partial JSON is not accepted")
        def nonfinite(value):
            raise ValueError("Judge JSON contains a non-finite value: " + value)

        result = json.loads(body["choices"][0]["message"]["content"], parse_constant=nonfinite)
        if not isinstance(result, dict):
            raise ValueError("Judge must return one JSON object")
        return result


def quoted_span(text, quote, occurrence=0):
    if not isinstance(quote, str) or not quote.strip() or type(occurrence) is not int or occurrence < 0:
        raise ValueError("Evidence needs an exact nonempty quote and nonnegative occurrence")
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start < 0:
            raise ValueError("Judge quoted text not present in the evidence")
    return {"start": start, "end": start + len(quote), "quote": quote}
