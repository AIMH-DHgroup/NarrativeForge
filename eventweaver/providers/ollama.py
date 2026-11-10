from __future__ import annotations

import math

from ..ollama_client import generate_ollama, list_ollama_models, resolve_ollama_host
from .base import GenerationResult


class OllamaProvider:
    supports_penalties = True
    supports_text_mode = True

    def __init__(
        self,
        model: str,
        *,
        num_ctx: int = 8192,
        host: str | None = None,
        timeout: int = 900,
        retries: int = 1,
        preflight: bool = True,
        json_mode: bool = False,
        think: bool | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        if frequency_penalty is not None or presence_penalty is not None:
            raise ValueError(
                "Ollama does not expose frequency_penalty/presence_penalty in this provider; "
                "use its native defaults instead"
            )
        self.model = model
        self.num_ctx = num_ctx
        self.host = resolve_ollama_host(host)
        self.timeout = timeout
        self.retries = retries
        self.json_mode = json_mode
        self.think = think
        if preflight:
            available = list_ollama_models(self.host)
            if model not in available:
                raise RuntimeError(f"Ollama model '{model}' is not installed locally. Install it with 'ollama pull {model}'.")

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048,
                  frequency_penalty: float | None = None, presence_penalty: float | None = None,
                  json_mode: bool | None = None) -> GenerationResult:
        if frequency_penalty is not None or presence_penalty is not None:
            raise ValueError(
                "Ollama does not expose frequency_penalty/presence_penalty in this provider"
            )
        result = generate_ollama(
            prompt,
            self.model,
            temperature=temperature,
            num_ctx=self.num_ctx,
            max_new_tokens=max_new_tokens,
            json_mode=self.json_mode if json_mode is None else json_mode,
            think=self.think,
            timeout=self.timeout,
            ollama_host=self.host,
            retries=self.retries,
            frequency_penalty=None,
            presence_penalty=None,
        )
        return GenerationResult(text=result.text, runtime_seconds=result.runtime_seconds, error=result.error, truncated=result.truncated)

    def validate_temperature(self, temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Ollama temperature must be a finite number >= 0")
