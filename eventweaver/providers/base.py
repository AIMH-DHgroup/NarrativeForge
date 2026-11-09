from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    text: str
    runtime_seconds: float
    error: str | None = None
    truncated: bool = False


class LLMProvider(Protocol):
    model: str

    def generate(self, prompt: str, *, temperature: float = 0.1, max_new_tokens: int = 2048) -> GenerationResult:
        ...

    def validate_temperature(self, temperature: float) -> None:
        ...
