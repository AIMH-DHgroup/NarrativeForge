from __future__ import annotations

import os
import re
import time
import math

from .base import GenerationResult


class HuggingFaceApiProvider:
    def __init__(
        self,
        model: str,
        *,
        token_env: str = "HF_TOKEN",
        provider: str = "auto",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.token_env = token_env
        self.provider = provider
        self.timeout = timeout
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError("Hugging Face API support requires huggingface_hub. Install it with 'pip install huggingface_hub'.") from exc
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(f"Environment variable {token_env} is not set. Put a Hugging Face token in that variable.")
        self.client = InferenceClient(api_key=token, provider=provider, timeout=timeout)

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            text = completion.choices[0].message.content or ""
            return GenerationResult(text=str(text).strip(), runtime_seconds=time.perf_counter() - started)
        except Exception as exc:
            return GenerationResult(text="", runtime_seconds=time.perf_counter() - started, error=str(exc))

    def validate_temperature(self, temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Hugging Face API temperature must be a finite number > 0")


class HuggingFaceLocalProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local Hugging Face support requires transformers, torch, and accelerate. "
                "Install it with 'pip install -e .[hf-local]'."
            ) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
                dtype="auto",
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load local Hugging Face model '{model}': {exc}") from exc

    def _format_prompt(self, prompt: str) -> str:
        """Use the model chat template when available, without hidden reasoning."""

        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (AttributeError, TypeError, ValueError):
            return prompt

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        started = time.perf_counter()
        try:
            kwargs = {"max_new_tokens": max_new_tokens, "return_full_text": False}
            if temperature > 0:
                kwargs.update({"do_sample": True, "temperature": temperature})
            else:
                kwargs.update({"do_sample": False})
            output = self.pipeline(self._format_prompt(prompt), **kwargs)
            text = output[0].get("generated_text", "") if output else ""
            text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
            return GenerationResult(text=str(text).strip(), runtime_seconds=time.perf_counter() - started)
        except Exception as exc:
            return GenerationResult(text="", runtime_seconds=time.perf_counter() - started, error=str(exc))

    def validate_temperature(self, temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Local Hugging Face temperature must be a finite number >= 0")
