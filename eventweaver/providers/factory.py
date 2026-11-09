from __future__ import annotations

from .huggingface import HuggingFaceApiProvider, HuggingFaceLocalProvider
from .ollama import OllamaProvider


def build_provider(
    provider_name: str,
    model: str,
    *,
    num_ctx: int = 8192,
    ollama_host: str | None = None,
    ollama_timeout: int = 900,
    ollama_retries: int = 1,
    skip_ollama_preflight: bool = False,
    json_mode: bool = False,
    think: bool | None = None,
    frequency_penalty: float = 1.0,
    presence_penalty: float = 0.8,
    hf_token_env: str = "HF_TOKEN",
    hf_provider: str = "auto",
    hf_timeout: int = 120,
):
    name = provider_name.strip().lower()
    if name == "ollama":
        return OllamaProvider(
            model,
            num_ctx=num_ctx,
            host=ollama_host,
            timeout=ollama_timeout,
            retries=ollama_retries,
            preflight=not skip_ollama_preflight,
            json_mode=json_mode,
            think=think,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    if name == "huggingface-api":
        return HuggingFaceApiProvider(model, token_env=hf_token_env, provider=hf_provider, timeout=hf_timeout)
    if name == "huggingface-local":
        return HuggingFaceLocalProvider(model)
    raise ValueError("provider must be one of: ollama, huggingface-api, huggingface-local")
