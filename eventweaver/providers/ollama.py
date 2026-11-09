from __future__ import annotations

import math

from ..ollama_client import generate_ollama, list_ollama_models, resolve_ollama_host
from .base import GenerationResult
from ..config import validate_penalty


class OllamaProvider:
    supports_penalties = True

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
        frequency_penalty: float = 1.0,
        presence_penalty: float = 0.8,
    ) -> None:
        self.frequency_penalty = validate_penalty(frequency_penalty, "frequency_penalty")
        self.presence_penalty = validate_penalty(presence_penalty, "presence_penalty")
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
                 frequency_penalty: float | None = None, presence_penalty: float | None = None) -> GenerationResult:
        result = generate_ollama(
            prompt,
            self.model,
            temperature=temperature,
            num_ctx=self.num_ctx,
            max_new_tokens=max_new_tokens,
            json_mode=self.json_mode,
            think=self.think,
            timeout=self.timeout,
            ollama_host=self.host,
            retries=self.retries,
            frequency_penalty=self.frequency_penalty if frequency_penalty is None else frequency_penalty,
            presence_penalty=self.presence_penalty if presence_penalty is None else presence_penalty,
        )
        return GenerationResult(text=result.text, runtime_seconds=result.runtime_seconds, error=result.error, truncated=result.truncated)

    def validate_temperature(self, temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Ollama temperature must be a finite number >= 0")
