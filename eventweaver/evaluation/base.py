from __future__ import annotations

from typing import Any, Protocol

from .models import EvaluationInput


class Evaluator(Protocol):
    reference: str
    method: str

    def evaluate(self, inputs: EvaluationInput, config: dict[str, Any]) -> dict[str, Any]: ...


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unsupported", "score": None, "warnings": [reason]}
